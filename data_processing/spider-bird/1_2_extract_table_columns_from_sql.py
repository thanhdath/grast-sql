#!/usr/bin/env python3
"""
Annotate each SQL query with the list of columns it touches.
First compute total tokens and estimated price for GPT-4.1-mini calls,
then ask user to confirm before proceeding.
"""
from __future__ import annotations

import argparse, datetime as dt, json, os, sqlite3
from multiprocessing import Pool
from pathlib import Path
from typing import List

import openai
from dotenv import load_dotenv
from pymongo import MongoClient, errors


# ───────────────────────────── CLI & ENV ──────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate SQL with used columns.")
    p.add_argument("split", choices=["train", "dev"],
                   help="Dataset split (train | dev).")
    p.add_argument("--dataset", choices=["bird", "spider"], default="bird",
                   help="Which dataset to process (bird | spider, default: %(default)s)")
    p.add_argument("--base-dir", type=Path, 
                   help="Root folder that contains the dataset files "
                        "(default: ../data/bird-062024 for bird, ../data/sft_data_collections/spider for spider)")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI",
                                     "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides).")
    p.add_argument("--processes", type=int, default=16,
                   help="Worker processes (default: %(default)s).")
    p.add_argument("--price-per-1k-tokens", type=float, default=0.03,
                   help="GPT-4.1-mini price per 1000 tokens (USD).")
    return p.parse_args()


def get_default_base_dir(dataset: str) -> Path:
    """Get the default base directory for the specified dataset."""
    if dataset == "bird":
        return Path("../data/bird-062024")
    elif dataset == "spider":
        return Path("../data/sft_data_collections/spider")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_json_path(base_dir: Path, dataset: str, split: str) -> Path:
    """Get the correct JSON file path based on dataset and split."""
    if dataset == "bird":
        # Bird dataset: ../data/bird-062024/train/train.json
        return base_dir / split / f"{split}.json"
    elif dataset == "spider":
        # Spider dataset: ../data/sft_data_collections/spider/train.json
        return base_dir / f"{split}.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


args = parse_args()
load_dotenv("../.env")
MONGO_URI = args.mongo_uri


# ───────────────────────────── HELPERS ────────────────────────────────
def get_schema(db_path: Path) -> List[str]:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    tables = [r[0] for r in cur.fetchall()]
    schema = []
    for t in tables:
        cur.execute(f"PRAGMA table_info('{t}');")
        schema.extend(f"{t}.{row[1]}" for row in cur.fetchall())
    con.close()
    return sorted(schema)


def count_tokens(text: str) -> int:
    # Basic approximation: 1 token ~ 4 characters (this is heuristic)
    # You can replace this with tiktoken or official tokenizer for accuracy
    return max(1, len(text) // 4)


def extract_used_columns(sql: str, model: str) -> List[str]:
    chat = openai.OpenAI().chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are SQL-aware. Resolve table aliases and list every "
                    "column as <table>.<column>. Return ONLY "
                    '{"columns": [...]} with unique strings.'
                ),
            },
            {"role": "user", "content": sql},
        ],
        max_tokens=2048,
    )
    return json.loads(chat.choices[0].message.content)["columns"]


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


# ───────────────────────────── WORKER ─────────────────────────────────
def process_sample(task):
    idx, sample, root_dir, dataset, split = task
    client = MongoClient(MONGO_URI)
    # Use dataset-specific collection names
    collection_name = f"{dataset}_{split}_samples"
    coll = client["mats"][collection_name]

    model = 'gpt-4.1-mini'

    db_id = sample["db_id"]
    sql_field = get_sql_field(dataset)
    sql_query = sample[sql_field]
    db_path = build_db_path(root_dir, dataset, split, db_id)

    try:
        schema = get_schema(db_path)
        used_cols = extract_used_columns(sql_query, model)
        # Normalize case to match schema
        used_cols = normalize_columns_case(used_cols, schema)
        used_cols = [c for c in used_cols if c in schema]
    except Exception as exc:
        print(f"[{idx}] ERROR: {exc}")
        client.close()
        return

    try:
        coll.update_one(
            {"_id": idx},
            {"$set": {
                "used_columns": used_cols,
                "schema": schema,
                "sql_parsed_at": dt.datetime.utcnow(),
            }},
            upsert=False,
        )
        print(f"[{idx}] updated ({len(used_cols)} cols, model={model})")
    except errors.PyMongoError as exc:
        print(f"[{idx}] Mongo ERROR: {exc}")
    finally:
        client.close()


# ───────────────────────────── MAIN ──────────────────────────────────
def main() -> None:
    # Set default base directory if not provided
    if args.base_dir is None:
        args.base_dir = get_default_base_dir(args.dataset)

    root_dir = args.base_dir
    dataset = args.dataset
    split = args.split

    json_path = get_json_path(root_dir, dataset, split)
    if not json_path.is_file():
        raise FileNotFoundError(f"Cannot find {json_path}")

    with json_path.open(encoding="utf-8") as fp:
        data = json.load(fp)

    total_tokens = 0
    sql_field = get_sql_field(dataset)
    for sample in data:
        # count tokens in SQL text as proxy for cost
        total_tokens += count_tokens(sample.get(sql_field, ""))

    # Calculate estimated price for GPT calls
    # Price per 1k tokens provided by args.price_per_1k_tokens (default 0.03 USD)
    estimated_cost = (total_tokens / 1000) * args.price_per_1k_tokens

    print(f"Dataset: {dataset}")
    print(f"Split: {split}")
    print(f"Total samples: {len(data)}")
    print(f"Estimated total tokens: {total_tokens}")
    print(f"Estimated cost for GPT-4.1-mini calls: ${estimated_cost:.4f} USD")

    confirm = input("Proceed with processing all samples? (yes/no): ").strip().lower()
    if confirm not in ("yes", "y"):
        print("Aborted by user.")
        return

    tasks = [(idx, sample, root_dir, dataset, split) for idx, sample in enumerate(data)]
    print(f"Processing {len(tasks)} samples...")

    from multiprocessing import Pool

    with Pool(processes=args.processes) as pool:
        pool.map(process_sample, tasks, chunksize=4)


if __name__ == "__main__":
    main()
