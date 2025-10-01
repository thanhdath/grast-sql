#!/usr/bin/env python3
"""
Annotate MongoDB documents with primary-key information.

For every document in  <dataset>_<split>_samples  we
    • open its <db_id>.sqlite file,
    • run  PRAGMA table_info(table)  on each table,
    • collect columns where  pk > 0  (composite-key order preserved),
    • write   {table: ["col1", "col2", …]}   to  primary_keys.

Usage
-----
python annotate_primary_keys.py train
python annotate_primary_keys.py dev --dataset spider --processes 4
"""

from __future__ import annotations
import argparse, os, sqlite3
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, List, Any

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


# ───────────────────────────── CLI ──────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate docs with primary-key info.")
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
    p.add_argument("--processes", type=int, default=8,
                   help="Worker processes")
    return p.parse_args()


def get_default_base_dir(dataset: str) -> Path:
    if dataset == "bird":
        return Path("../data/bird-062024")
    elif dataset == "spider":
        return Path("../data/sft_data_collections/spider")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ─────────────────────── path helpers ───────────────────────────
def build_db_path(root: Path, dataset: str, split: str, db_id: str) -> Path:
    if dataset == "bird":
        return root / split / f"{split}_databases" / db_id / f"{db_id}.sqlite"
    elif dataset == "spider":
        return root / "database" / db_id / f"{db_id}.sqlite"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ──────────────────── primary-key extractor ─────────────────────
def get_primary_keys(db_path: Path, tables: List[str]) -> Dict[str, List[str]]:
    """
    Return dict  {table: [pk_col1, pk_col2, …]}.

    Uses  PRAGMA table_info(table)  and selects rows where pk > 0,
    ordering by pk (1-based) to preserve composite-key column order.
    """
    pk_info: Dict[str, List[str]] = {}
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        for tbl in tables:
            try:
                cur.execute(f"PRAGMA table_info('{tbl}')")
                rows = cur.fetchall()        # cid, name, type, notnull, dflt, pk
                cols = [row[1] for row in sorted(rows, key=lambda r: r[5]) if row[5] > 0]
                if cols:
                    pk_info[tbl] = cols
            except Exception as exc_tbl:
                print(f"Error fetching PKs for table {tbl}: {exc_tbl}")
        con.close()
    except Exception as exc:
        print(f"Error opening DB {db_path}: {exc}")
    return pk_info


# ────────────────────── per-document job ────────────────────────
def process_doc(task):
    _id, doc, root, dataset, split = task
    db_id = doc.get("db_id")
    if not db_id:
        print(f"[{_id}] Missing db_id")
        return None

    schema = doc.get("schema", [])
    tables = sorted({c.split(".", 1)[0] for c in schema if "." in c})

    db_path = build_db_path(root, dataset, split, db_id)
    pk_info = get_primary_keys(db_path, tables)

    return UpdateOne({"_id": _id}, {"$set": {"primary_keys": pk_info or {}}})


# ─────────────────────────── main ───────────────────────────────
def main() -> None:
    args = parse_args()
    load_dotenv("../.env")
    if args.base_dir is None:
        args.base_dir = get_default_base_dir(args.dataset)
    split = args.split
    dataset = args.dataset
    root  = args.base_dir

    client = MongoClient(args.mongo_uri)
    # Use dataset-specific collection names
    collection_name = f"{dataset}_{split}_samples"
    coll   = client["mats"][collection_name]

    print(f"[{dataset}_{split}] Removing existing primary_keys fields…")
    coll.update_many({}, {"$unset": {"primary_keys": ""}})

    docs = list(coll.find({}, {"db_id": 1, "schema": 1}))
    print(f"[{dataset}_{split}] {len(docs)} documents to enrich with primary_keys")

    if not docs:
        client.close()
        return

    tasks = [(d["_id"], d, root, dataset, split) for d in docs]

    with Pool(processes=args.processes) as pool:
        updates = pool.map(process_doc, tasks)

    updates = [u for u in updates if u]
    if updates:
        res = coll.bulk_write(updates, ordered=False)
        print(f"Updated {res.modified_count} documents with primary_keys.")
    else:
        print("No primary-key info found to update.")

    client.close()


if __name__ == "__main__":
    main()
