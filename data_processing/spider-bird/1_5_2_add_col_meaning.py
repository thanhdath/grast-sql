#!/usr/bin/env python3
"""
Update `column_meaning` in mats.<dataset>_<split>_samples.

Priority per column
-------------------
1. Meaning built from the database_description CSVs.
2. If that meaning is the empty string -> fall back to the pretty name
   from <split>_tables.json (Spider/Bird schema).

Robust CSV reading
------------------
• Tries 'utf-8', 'utf-8-sig', 'cp1252', 'latin1' in that order.
• If all fail, uses chardet / charset-normalizer (if installed) to guess.

Usage
-----
python update_column_meaning_from_csv.py train                    # bird dataset
python update_column_meaning_from_csv.py dev --dataset spider     # spider dataset
python update_column_meaning_from_csv.py dev --base-dir ./data/bird
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


# ───────────────────────────── CLI ──────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update column_meaning from CSVs (with encoding & JSON fallback).")
    p.add_argument("split", choices=["train", "dev"])
    p.add_argument("--dataset", choices=["bird", "spider"], default="bird",
                   help="Which dataset to process (bird | spider, default: %(default)s)")
    p.add_argument("--base-dir", type=Path, 
                   help="Root folder that contains the dataset files "
                        "(default: ../data/bird-062024 for bird, ../data/sft_data_collections/spider for spider)")
    p.add_argument("--json-path", type=Path, default=None,
                   help="Explicit <split>_tables.json path")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"))
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


def get_tables_json_path(base_dir: Path, dataset: str, split: str, json_path: Path = None) -> Path:
    """Get the tables JSON path based on dataset structure."""
    if json_path:
        return json_path
    
    if dataset == "bird":
        # Bird dataset: ../data/bird-062024/train/train_tables.json
        return base_dir / split / f"{split}_tables.json"
    elif dataset == "spider":
        # Spider dataset: ../data/sft_data_collections/spider/tables.json
        return base_dir / "tables.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


args = parse_args()
load_dotenv("../.env")
MONGO_URI = args.mongo_uri

# Set default base directory if not provided
if args.base_dir is None:
    args.base_dir = get_default_base_dir(args.dataset)

# ───────────── Verify paths ─────────────
db_root = get_db_root(args.base_dir, args.dataset, args.split)
if not db_root.is_dir():
    sys.exit(f"❌  Database directory not found: {db_root}")

tables_json = get_tables_json_path(args.base_dir, args.dataset, args.split, args.json_path)
if not tables_json.is_file():
    sys.exit(f"❌  Cannot find tables JSON: {tables_json}")

# ───────────── MongoDB ─────────────
client = MongoClient(MONGO_URI)
# Use dataset-specific collection names
collection_name = f"{args.dataset}_{args.split}_samples"
samples = client["mats"][collection_name]

# ─────────── Load JSON fallback once ───────────
with tables_json.open(encoding="utf-8") as fp:
    schemas: List[Dict] = json.load(fp)

json_fallback: Dict[str, Dict[str, str]] = {}
for entry in schemas:
    db_id = entry["db_id"]
    col_map: Dict[str, str] = {}
    for (tbl_idx, col_orig), (_, col_norm) in zip(
        entry["column_names_original"], entry["column_names"]
    ):
        if tbl_idx == -1:           # '*' pseudo-column
            continue
        tbl_orig = entry["table_names_original"][tbl_idx]
        col_map[f"{tbl_orig}.{col_orig}"] = col_norm
    json_fallback[db_id] = col_map

# ───────── Helper: build meaning string ─────────
def build_meaning(col_name, col_desc) -> str:
    name = str(col_name).strip() if pd.notna(col_name) else ""
    desc = str(col_desc).strip() if pd.notna(col_desc) else ""
    if not desc:
        return name
    if not name:
        return desc
    return desc if name == desc else f"{name}. {desc}"

# ───────── Robust CSV reader ─────────
_ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "cp1252", "latin1")

def read_csv_any(path: Path) -> pd.DataFrame:
    """Try several encodings; fall back to chardet/charset-normalizer."""
    for enc in _ENCODING_CANDIDATES:
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, engine="python")
        except UnicodeDecodeError:
            continue

    # Optional automatic detection
    try:
        try:
            # Preferred: charset-normalizer (faster, no C ext required)
            import charset_normalizer as cn
            with open(path, "rb") as fh:
                raw = fh.read(32768)
            guess = cn.from_bytes(raw).best().encoding
        except ImportError:
            # Fallback to chardet
            import chardet
            with open(path, "rb") as fh:
                raw = fh.read(32768)
            guess = chardet.detect(raw)["encoding"]

        if guess:
            return pd.read_csv(path, dtype=str, encoding=guess, engine="python")
    except Exception:
        pass

    raise UnicodeDecodeError("csv", b"", 0, 1, "unable to detect encoding")

# ───────── Main loop ─────────
processed, updated = 0, 0

for db_dir in db_root.iterdir():
    if not db_dir.is_dir():
        continue
    db_id = db_dir.name
    desc_dir = db_dir / "database_description"
    
    csv_map: Dict[str, str] = {}
    if desc_dir.is_dir():
        for csv_file in desc_dir.glob("*.csv"):
            table_orig = csv_file.stem
            try:
                df = read_csv_any(csv_file)
            except Exception as exc:
                print(f"⚠️  {csv_file} skipped ({exc})")
                continue

            req_cols = {"original_column_name", "column_name", "column_description"}
            if not req_cols.issubset(df.columns):
                print(f"⚠️  {csv_file} missing {req_cols - set(df.columns)} – skipped.")
                continue

            for _, row in df.iterrows():
                key = f"{table_orig}.{row['original_column_name']}"
                csv_map[key] = build_meaning(row['column_name'], row['column_description'])

    # -------- Combine per-column meanings ----------
    json_map = json_fallback.get(db_id, {})
    final_map: Dict[str, str] = {}

    for key in csv_map.keys() | json_map.keys():
        csv_val = csv_map.get(key, "").strip()
        if csv_val:
            final_map[key] = csv_val
        else:
            json_val = json_map.get(key, "").strip()
            if json_val:
                final_map[key] = json_val

    processed += 1
    if not final_map:
        continue

    existing = samples.find_one({"db_id": db_id}, {"column_meaning": 1})
    existing_meaning = existing.get("column_meaning") if existing and existing.get("column_meaning") else {}
    merged = existing_meaning | final_map

    res = samples.update_many(
        {"db_id": db_id},
        {"$set": {"column_meaning": merged}},
        upsert=False
    )
    if res.modified_count:
        updated += res.modified_count

print(f"[{args.dataset}_{args.split}] scanned {processed} DBs – updated {updated}.")
client.close()
