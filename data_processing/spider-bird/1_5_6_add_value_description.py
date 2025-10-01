#!/usr/bin/env python3
"""
Attach per-column *value descriptions* to every document in
mats.<dataset>_<split>_samples.

Directory layout
----------------
<base-dir>/<split>/<split>_databases/<db_id>/database_description/*.csv (Bird)
<base-dir>/database/<db_id>/database_description/*.csv (Spider)

For every CSV row with a non-blank `value_description`, we write:

    {
        "column_value_desc": {
            "table.col_orig": "<value description>",
            ...
        }
    }

Encoding handling
-----------------
• Tries 'utf-8', 'utf-8-sig', 'cp1252', 'latin1'.
• If all fail, guesses with charset-normalizer (or chardet).

Usage
-----
python add_value_descriptions.py train                    # bird dataset
python add_value_descriptions.py dev --dataset spider     # spider dataset
python add_value_descriptions.py dev --base-dir ./data/bird
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateMany

# ───────────────────────────── CLI & ENV ─────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add column value-description fields.")
    p.add_argument("split", choices=["train", "dev"])
    p.add_argument("--dataset", choices=["bird", "spider"], default="bird",
                   help="Which dataset to process (bird | spider, default: %(default)s)")
    p.add_argument("--base-dir", type=Path, 
                   help="Root folder that contains the dataset files "
                        "(default: ../data/bird-062024 for bird, ../data/sft_data_collections/spider for spider)")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides)")
    return p.parse_args()


def get_default_base_dir(dataset: str) -> Path:
    """Get the default base directory for the specified dataset."""
    if dataset == "bird":
        return Path("../data/bird-062024")
    elif dataset == "spider":
        return Path("../data/sft_data_collections/spider")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_db_root(base_dir: Path, dataset: str, split: str) -> Path:
    """Get the database root directory based on dataset structure."""
    if dataset == "bird":
        # Bird dataset: ../data/bird-062024/train/train_databases/
        return base_dir / split / f"{split}_databases"
    elif dataset == "spider":
        # Spider dataset: ../data/sft_data_collections/spider/database/
        return base_dir / "database"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


args = parse_args()
load_dotenv("../.env")

# Set default base directory if not provided
if args.base_dir is None:
    args.base_dir = get_default_base_dir(args.dataset)

# ­────────────────────────── Paths ­──────────────────────────
db_root = get_db_root(args.base_dir, args.dataset, args.split)
if not db_root.is_dir():
    sys.exit(f"❌  Cannot find database root: {db_root}")

# ­────────────────────────── Mongo connection ­────────────────────────
client = MongoClient(args.mongo_uri)
# Use dataset-specific collection names
collection_name = f"{args.dataset}_{args.split}_samples"
col = client["mats"][collection_name]

# ­───────────────────── Robust CSV reader helper ­────────────────────
_ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "cp1252", "latin1")

def read_csv_any(path: Path) -> pd.DataFrame:
    """Try a handful of encodings, then guess with charset-normalizer / chardet."""
    for enc in _ENCODING_CANDIDATES:
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, engine="python")
        except UnicodeDecodeError:
            continue

    # Last-chance automatic detection
    try:
        try:
            import charset_normalizer as cn
            with open(path, "rb") as fh:
                raw = fh.read(32768)
            guess = cn.from_bytes(raw).best().encoding
        except ImportError:
            import chardet
            with open(path, "rb") as fh:
                raw = fh.read(32768)
            guess = chardet.detect(raw)["encoding"]

        if guess:
            return pd.read_csv(path, dtype=str, encoding=guess, engine="python")
    except Exception:
        pass

    raise UnicodeDecodeError("csv", b"", 0, 1, "unable to detect encoding")

# ­────────────────────────── Build updates ­───────────────────────────
bulk: List[UpdateMany] = []

for db_dir in db_root.iterdir():
    if not db_dir.is_dir():
        continue
    db_id = db_dir.name
    desc_dir = db_dir / "database_description"
    if not desc_dir.is_dir():
        continue

    column_value_desc: Dict[str, str] = {}

    for csv_path in desc_dir.glob("*.csv"):
        table_orig = csv_path.stem  # e.g. frpm.csv → "frpm"
        try:
            df = read_csv_any(csv_path)
        except Exception as e:
            print(f"⚠️  Skipping {csv_path} ({e})")
            continue

        if "original_column_name" not in df.columns:
            print(f"⚠️  {csv_path} missing 'original_column_name' – skipped.")
            continue

        val_desc_col = "value_description"
        if val_desc_col not in df.columns:
            # Nothing to gather; go to next file
            continue

        for _, row in df.iterrows():
            col_orig = row["original_column_name"]
            val_desc = row.get(val_desc_col, "")
            if isinstance(val_desc, str) and val_desc.strip():
                key = f"{table_orig}.{col_orig}"
                column_value_desc[key] = val_desc.strip()

    if column_value_desc:
        bulk.append(
            UpdateMany(
                {"db_id": db_id},
                {"$set": {"column_value_desc": column_value_desc}},
                upsert=False,
            )
        )

# ­────────────────────────── Execute ­───────────────────────────
if bulk:
    res = col.bulk_write(bulk, ordered=False)
    print(f"[{args.dataset}_{args.split}] added value descriptions to "
          f"{res.modified_count} documents.")
else:
    print("Nothing to update – check CSV files or collection contents.")

client.close()
