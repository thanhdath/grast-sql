#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sqlite3
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, List, Any

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate docs with foreign key info.")
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


def build_db_path(root: Path, dataset: str, split: str, db_id: str) -> Path:
    if dataset == "bird":
        return root / split / f"{split}_databases" / db_id / f"{db_id}.sqlite"
    elif dataset == "spider":
        return root / "database" / db_id / f"{db_id}.sqlite"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_foreign_keys(db_path: Path, tables: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    fk_info = {}
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        for tbl in tables:
            try:
                cur.execute(f"PRAGMA foreign_key_list('{tbl}')")
                rows = cur.fetchall()
                fk_list = []
                for row in rows:
                    to_col = row[4]
                    if to_col is None or to_col.strip().lower() == "none" or to_col.strip() == "":
                        continue
                    fk_list.append({
                        "from": row[3],
                        "to": to_col,
                        "ref_table": row[2],
                        "on_update": row[5],
                        "on_delete": row[6],
                        "match": row[7]
                    })
                if fk_list:
                    fk_info[tbl] = fk_list
            except Exception as exc_tbl:
                print(f"Error fetching foreign keys for table {tbl}: {exc_tbl}")
        con.close()
    except Exception as exc:
        print(f"Error opening DB {db_path}: {exc}")
    return fk_info


def process_doc(task):
    _id, doc, root, dataset, split = task
    db_id = doc.get("db_id")
    if not db_id:
        print(f"[{_id}] Missing db_id")
        return None

    schema = doc.get("schema", [])
    tables = sorted(set(c.split(".", 1)[0] for c in schema if "." in c))

    db_path = build_db_path(root, dataset, split, db_id)
    fk_info = get_foreign_keys(db_path, tables)

    if not fk_info:
        fk_info = {}

    return UpdateOne({"_id": _id}, {"$set": {"foreign_keys": fk_info}})


def main() -> None:
    args = parse_args()
    load_dotenv("../.env")
    if args.base_dir is None:
        args.base_dir = get_default_base_dir(args.dataset)
    split = args.split
    dataset = args.dataset
    root = args.base_dir

    client = MongoClient(args.mongo_uri)
    # Use dataset-specific collection names
    collection_name = f"{dataset}_{split}_samples"
    coll = client["mats"][collection_name]

    print(f"[{dataset}_{split}] Removing existing foreign_keys fields...")
    coll.update_many({}, {"$unset": {"foreign_keys": ""}})

    docs = list(coll.find({}, {"db_id": 1, "schema": 1}))
    print(f"[{dataset}_{split}] {len(docs)} documents to enrich with foreign_keys")

    if not docs:
        client.close()
        return

    tasks = [(d["_id"], d, root, dataset, split) for d in docs]

    with Pool(processes=args.processes) as pool:
        updates = pool.map(process_doc, tasks)

    updates = [u for u in updates if u]

    if updates:
        res = coll.bulk_write(updates, ordered=False)
        print(f"Updated {res.modified_count} documents with foreign_keys.")
    else:
        print("No foreign key info found to update.")

    client.close()


if __name__ == "__main__":
    main()
