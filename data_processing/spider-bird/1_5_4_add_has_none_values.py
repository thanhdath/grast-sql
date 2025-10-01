#!/usr/bin/env python3
"""
Update only the `has_null` field in `column_info` for each column.

This script:
1. Loads documents from MongoDB.
2. Connects to the corresponding SQLite database.
3. For each column in the schema:
   - Checks if the column has any NULL values.
   - Updates only the `has_null` field in `column_info`.
4. Preserves all existing fields (e.g., type, similar_values).
"""

from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv


# ────────────────────── CLI & ENV ──────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update only has_null in column_info.")
    p.add_argument("split", choices=["train", "dev"],
                   help="Dataset split (train | dev)")
    p.add_argument("--dataset", choices=["bird", "spider"], default="bird",
                   help="Which dataset to process (bird | spider, default: %(default)s)")
    p.add_argument("--base-dir", type=Path, 
                   help="Root folder that contains the dataset files "
                        "(default: ../data/bird-062024 for bird, ../data/sft_data_collections/spider for spider)")
    p.add_argument("--mongo-uri",
                   default="mongodb://192.168.1.108:27017",
                   help="MongoDB connection string")
    p.add_argument("--processes", type=int, default=8,
                   help="Number of worker processes (default: %(default)s)")
    return p.parse_args()


def get_default_base_dir(dataset: str) -> Path:
    if dataset == "bird":
        return Path("../data/bird-062024")
    elif dataset == "spider":
        return Path("../data/sft_data_collections/spider")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def build_db_path(root: Path, dataset: str, split: str, db_id: str) -> Path:
    if dataset == "bird":
        return root / split / f"{split}_databases" / db_id / f"{db_id}.sqlite"
    elif dataset == "spider":
        return root / "database" / db_id / f"{db_id}.sqlite"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ────────────────────── WORKER ──────────────────────
def process_doc(task):
    idx, doc, root, dataset, split = task
    db_id = doc["db_id"]
    schema = doc.get("schema", [])
    existing_col_info = doc.get("column_info", {})

    db_path = build_db_path(root, dataset, split, db_id)

    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
    except Exception as exc:
        print(f"[{idx}] DB connection error: {exc}")
        return None

    col_info = dict(existing_col_info)  # Start with existing data

    for col in schema:
        try:
            tbl, col_name = col.split(".", 1)
        except ValueError:
            print(f"[{idx}] Invalid column name: {col}")
            continue

        # Check for NULL values
        has_null: Optional[bool] = None
        try:
            cur.execute(f'SELECT 1 FROM "{tbl}" WHERE "{col_name}" IS NULL LIMIT 1')
            result = cur.fetchone()
            has_null = bool(result)
        except Exception as exc:
            print(f"[{idx}] Null check error {tbl}.{col_name}: {exc}")

        # Preserve existing column data, update only has_null
        col_data = dict(existing_col_info[col]) if col in existing_col_info else {}
        col_data['has_null'] = has_null
        col_info[col] = col_data

    con.close()
    return UpdateOne({"_id": idx}, {"$set": {"column_info": col_info}})


# ────────────────────── MAIN ──────────────────────
def main() -> None:
    args = parse_args()
    load_dotenv("../.env")
    if args.base_dir is None:
        args.base_dir = get_default_base_dir(args.dataset)
    split = args.split
    dataset = args.dataset
    root = args.base_dir
    mongo = MongoClient(args.mongo_uri)
    # Use dataset-specific collection names
    collection_name = f"{dataset}_{split}_samples"
    coll = mongo["mats"][collection_name]

    # Include column_info in the fetched documents
    docs = list(
        coll.find({}, {"db_id": 1, "schema": 1, "column_info": 1})
    )

    print(f"[{dataset}_{split}] {len(docs)} documents to update with has_null info")

    if not docs:
        mongo.close()
        return

    tasks = [(d["_id"], d, root, dataset, split) for d in docs]

    with Pool(processes=args.processes) as pool:
        updates = list(
            tqdm(
                pool.imap(process_doc, tasks),
                total=len(tasks),
                desc=f"Processing {dataset}_{split} documents",
                unit="doc"
            )
        )

    updates = [u for u in updates if u]  # Remove None results

    if updates:
        res = coll.bulk_write(updates, ordered=False)
        print(f"Updated {res.modified_count} documents.")
    else:
        print("No valid updates generated.")

    mongo.close()


if __name__ == "__main__":
    main()