#!/usr/bin/env python3
"""
repair_columns.py
─────────────────
Repair documents in the MongoDB collections  mats.<dataset>_<split>_samples  whose
`used_columns` list is empty, missing, or contains items that do not exist
in the database schema (case-insensitive).

Typical usage
-------------
python repair_columns.py dev                     # default paths for bird
python repair_columns.py train --dataset spider  # spider dataset
python repair_columns.py train --processes 8
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
from multiprocessing import Pool
from pathlib import Path
from typing import List

from openai import OpenAI
from dotenv import load_dotenv
from pymongo import MongoClient


# ───────────────────────────────────────────────────────────────
# 0.  CLI & ENV
# ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repair used_columns mismatches.")
    p.add_argument("split", choices=["train", "dev"],
                   help="Dataset split to repair (train | dev).")
    p.add_argument("--dataset", choices=["bird", "spider"], default="bird",
                   help="Which dataset to process (bird | spider, default: %(default)s)")
    p.add_argument("--base-dir", type=Path, 
                   help="Root folder that contains the dataset files "
                        "(default: ../data/bird-062024 for bird, ../data/sft_data_collections/spider for spider)")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides).")
    p.add_argument("--processes", type=int, default=16,
                   help="Worker processes (default: %(default)s).")
    return p.parse_args()


def get_default_base_dir(dataset: str) -> Path:
    """Get the default base directory for the specified dataset."""
    if dataset == "bird":
        return Path("../data/bird-062024")
    elif dataset == "spider":
        return Path("../data/sft_data_collections/spider")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_sql_field(dataset: str) -> str:
    """Get the correct SQL field name for the dataset."""
    if dataset == "bird":
        return "SQL"
    elif dataset == "spider":
        return "query"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def normalize_columns_case(used_cols: List[str], schema: List[str]) -> List[str]:
    """
    Normalize the case of used_columns to match the schema case.
    Returns only columns that exist in the schema (case-insensitive match).
    """
    # Create a case-insensitive mapping from schema
    schema_lower_to_original = {col.lower(): col for col in schema}
    
    normalized_cols = []
    for col in used_cols:
        col_lower = col.lower()
        if col_lower in schema_lower_to_original:
            # Use the original case from schema
            normalized_cols.append(schema_lower_to_original[col_lower])
    
    return normalized_cols


args = parse_args()
load_dotenv("../.env")
MONGO_URI = args.mongo_uri

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───────────────────────────────────────────────────────────────
# 1.  UTILITIES
# ───────────────────────────────────────────────────────────────
def get_schema(db_path: Path) -> List[str]:
    """Return a sorted list of fully-qualified column names:  table.column"""
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    tables = [r[0] for r in cur.fetchall()]

    cols = []
    for t in tables:
        cur.execute(f"PRAGMA table_info('{t}');")
        cols.extend(f"{t}.{row[1]}" for row in cur.fetchall())

    con.close()
    return sorted(cols)


def extract_used_columns(sql: str, model: str = "o1-mini") -> List[str]:
    """
    Ask an LLM to resolve table aliases and return every column in
    <table>.<column> form.  Return a unique list.
    """
    # chat = client.chat.completions.create(
    #     model=model,
    #     temperature=0.1,
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": (
    #                 "You are SQL-aware. Resolve table aliases and list every "
    #                 "column as <table>.<column>. Return ONLY "
    #                 '{"columns": [...]} with unique strings.'
    #             ),
    #         },
    #         {"role": "user", "content": sql},
    #     ],
    # )
    # return json.loads(chat.choices[0].message.content)["columns"]

    response = client.responses.create(
        model="o4-mini",
        reasoning={"effort": "medium"},
        input=[
            {
                "role": "user", 
                "content": f"""You are SQL-aware. Resolve table aliases and list every column as <table>.<column>.
Return ONLY {{"columns": [...]}} with unique strings.

{sql}"""
            }
        ]
    )

    return json.loads(response.output_text)["columns"]


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


def needs_fix(doc: dict) -> bool:
    """
    Return True if the document should be repaired.

    • `schema` or `used_columns` field is missing
    • `used_columns` is empty or None
    • Any entry in `used_columns` is absent from `schema`
      (case-insensitive comparison)
    """
    if "schema" not in doc or "used_columns" not in doc:
        return True

    # Empty list or None → needs repair
    if not doc["used_columns"]:
        return True

    exact = set(doc["schema"])
    lower = {c.lower() for c in exact}

    for col in doc["used_columns"]:
        if col in exact or col.lower() in lower:
            continue
        return True  # mismatch detected

    return False


# ───────────────────────────────────────────────────────────────
# 2.  WORKER
# ───────────────────────────────────────────────────────────────
def repair_document(task):
    idx, doc, root_dir, dataset, split = task
    client = MongoClient(MONGO_URI)
    # Use dataset-specific collection names
    collection_name = f"{dataset}_{split}_samples"
    coll = client["mats"][collection_name]

    try:
        db_id = doc["db_id"]
        sql_field = get_sql_field(dataset)
        sql_query = doc[sql_field]
        db_path = build_db_path(root_dir, dataset, split, db_id)

        schema = get_schema(db_path)
        used_cols = extract_used_columns(sql_query)
        print(used_cols)
        # Normalize case to match schema
        used_cols = normalize_columns_case(used_cols, schema)
        print(f"Used columns after filtering: {used_cols}")
    except Exception as exc:
        print(f"[{idx}] ERROR: {exc}")
        client.close()
        return

    coll.update_one(
        {"_id": idx},
        {
            "$set": {
                "schema": schema,
                "used_columns": used_cols,
                "updated_at": dt.datetime.utcnow(),
            },
            "$unset": {
                # clear any previous audit flags
                "audit_not_in_schema": "",
                "audit_case_mismatch": "",
            },
        },
    )
    print(f"[{idx}] fixed → {len(used_cols)} columns")
    client.close()


# ───────────────────────────────────────────────────────────────
# 3.  MAIN
# ───────────────────────────────────────────────────────────────
def main() -> None:
    # Set default base directory if not provided
    if args.base_dir is None:
        args.base_dir = get_default_base_dir(args.dataset)

    root_dir = args.base_dir
    dataset = args.dataset
    split = args.split

    client = MongoClient(MONGO_URI)
    # Use dataset-specific collection names
    collection_name = f"{dataset}_{split}_samples"
    coll = client["mats"][collection_name]

    # Get the correct SQL field name for the dataset
    sql_field = get_sql_field(dataset)

    to_fix = [
        (doc["_id"], doc, root_dir, dataset, split)
        for doc in coll.find(
            {}, {"db_id": 1, sql_field: 1, "schema": 1, "used_columns": 1}
        )
        if needs_fix(doc)
    ]
    client.close()

    print(f"Dataset: {dataset}")
    print(f"Split: {split}")
    print(f"{len(to_fix)} documents need repair")

    if not to_fix:
        return

    with Pool(processes=args.processes) as pool:
        pool.map(repair_document, to_fix, chunksize=4)


if __name__ == "__main__":
    main()
