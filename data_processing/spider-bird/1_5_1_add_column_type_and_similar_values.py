#!/usr/bin/env python3
"""
Enrich each document in mats.<dataset>_<split>_samples with:
    column_info = {
        "<table>.<column>": {
            "type": "INTEGER" | "TEXT" | …,
            "similar_values": [...]
        }, ...
    }

The script:
1. Loads the docs from MongoDB (skip if column_info already exists)
2. Reads column types from the split's SQLite database
3. Calls your /search_column_content service for two "similar" values
4. Bulk-updates the documents.

Usage
-----
python add_column_info.py train                    # default paths / 8 procs for bird
python add_column_info.py dev --dataset spider     # spider dataset
python add_column_info.py dev  --processes 4       # dev split, 4 workers
python add_column_info.py train --base-url http://api:8000
"""
from __future__ import annotations
import argparse, os, json, sqlite3, requests
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, Any, List

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


# ────────────────────── CLI & ENV ──────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate docs with column_info.")
    p.add_argument("split", choices=["train", "dev"],
                   help="Dataset split (train | dev)")
    p.add_argument("--dataset", choices=["bird", "spider"], default="bird",
                   help="Which dataset to process (bird | spider, default: %(default)s)")
    p.add_argument("--base-dir", type=Path, 
                   help="Root folder that contains the dataset files "
                        "(default: ../data/bird-062024 for bird, ../data/sft_data_collections/spider for spider)")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB connection string")
    p.add_argument("--base-url", default="http://localhost:8005",
                   help="URL of the /search_column_content service")
    p.add_argument("--processes", type=int, default=8,
                   help="Worker processes (default: %(default)s)")
    return p.parse_args()


def get_default_base_dir(dataset: str) -> Path:
    """Get the default base directory for the specified dataset."""
    if dataset == "bird":
        return Path("../data/bird-062024")
    elif dataset == "spider":
        return Path("../data/sft_data_collections/spider")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


args = parse_args()
load_dotenv("../.env")           # .env can still override MONGODB_URI
MONGO_URI = args.mongo_uri


# ────────────────────── HELPERS ──────────────────────
def get_column_types(db_path: Path) -> Dict[str, str]:
    """Return {table.column: column_type} from SQLite schema."""
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    tables = [r[0] for r in cur.fetchall()]

    col_types: Dict[str, str] = {}
    for tbl in tables:
        cur.execute(f"PRAGMA table_info('{tbl}')")
        for row in cur.fetchall():
            col_types[f"{tbl}.{row[1]}"] = row[2]
    con.close()
    return col_types


def get_similar(base_url: str, source: str,
                query: str, db_id: str, table: str, column: str) -> List[Any]:
    """Hit /search_column_content and return the `results` list."""
    payload = {
        "source":   source,
        "db_id":    db_id,
        "table":    table,
        "column":   column,
        "query":    query,
        "k":        5,
    }
    try:
        r = requests.post(f"{base_url}/search_column_content", json=payload, timeout=1000)
        if r.ok:
            return r.json().get("results", [])
    except Exception as exc:
        print(f"[svc] {table}.{column}: {exc}")
    return []


def build_db_path(root: Path, dataset: str, split: str, db_id: str) -> Path:
    """Build database path based on dataset structure."""
    if dataset == "bird":
        # Bird dataset: ../data/bird-062024/train/train_databases/db_id/db_id.sqlite
        return root / split / f"{split}_databases" / db_id / f"{db_id}.sqlite"
    elif dataset == "spider":
        # Spider dataset: ../data/sft_data_collections/spider/database/db_id/db_id.sqlite
        return root / "database" / db_id / f"{db_id}.sqlite"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ────────────────────── WORKER ──────────────────────
def process_doc(task):
    idx, doc, root, dataset, split, base_url = task
    db_id   = doc["db_id"]
    # question     = f"{doc['question']} {doc['evidence']}"
    question = doc.get("question", "")

    schema  = doc.get("schema", [])
    

    db_path = build_db_path(root, dataset, split, db_id)
    try:
        col_types = get_column_types(db_path)
    except Exception as exc:
        print(f"[{idx}] get_column_types ERROR: {exc}")
        return None

    col_info = {}
    for col in schema:
        col_type = col_types.get(col, "unknown")
        tbl, col_name = col.split(".", 1)
        similar_vals = get_similar(
            base_url, source=f"{dataset}-{split}", 
            query=question,
            db_id=db_id, 
            table=tbl.lower(), 
            column=col_name.lower()
        )
        print(f"[{idx}] {col}: {similar_vals}")
        col_info[col] = {"type": col_type, "similar_values": similar_vals}

    return UpdateOne({"_id": idx}, {"$set": {"column_info": col_info}})


# ────────────────────── MAIN ──────────────────────
def main() -> None:
    # Set default base directory if not provided
    if args.base_dir is None:
        args.base_dir = get_default_base_dir(args.dataset)

    split = args.split
    dataset = args.dataset
    root  = args.base_dir
    base_url = args.base_url

    mongo = MongoClient(MONGO_URI)
    # Use dataset-specific collection names
    collection_name = f"{dataset}_{split}_samples"
    coll  = mongo["mats"][collection_name]

    # docs = list(
    #     coll.find({"column_info": {"$exists": False}},
    #               {"db_id": 1, "question": 1, "evidence": 1, "schema": 1})
    # )
    docs = list(
        coll.find({},
                  {"db_id": 1, "question": 1, "evidence": 1, "schema": 1})
    )
    print(f"[{dataset}_{split}] {len(docs)} documents to enrich with column_info")

    if not docs:
        mongo.close()
        return

    # prepare tasks list for Pool
    tasks = [(d["_id"], d, root, dataset, split, base_url) for d in docs]

    with Pool(processes=args.processes) as pool:
        updates = pool.map(process_doc, tasks)

    updates = [u for u in updates if u]   # drop None

    if updates:
        res = coll.bulk_write(updates, ordered=False)
        print(f"Updated {res.modified_count} documents.")
    else:
        print("No valid updates generated.")

    mongo.close()


if __name__ == "__main__":
    main()
