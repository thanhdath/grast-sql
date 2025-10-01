#!/usr/bin/env python3
"""
spider_2_13_write_prompts_for_existing_meanings.py
──────────────────────────────────────────────────
Scan Spider2.0-lite samples in MongoDB and write prompt files for columns
that already have entries under `generated_column_meaning`.

This script NEVER calls OpenAI. It only generates prompt text files mirroring
how prompts are written by spider_2_13 for missing meanings.

Usage
-----
python spider_2_13_write_prompts_for_existing_meanings.py
python spider_2_13_write_prompts_for_existing_meanings.py --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set
import re

from dotenv import load_dotenv
from pymongo import MongoClient
import tiktoken


# ───────────────────────────────────────────────────────────────
# 0.  CLI & ENV
# ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write prompts for columns under generated_column_meaning")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not write prompt files; only compute token counts")
    return p.parse_args()


args = parse_args()
load_dotenv("/home/datht/mats/.env")
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017")
COLLECTION_NAME = "spider2_lite_samples"
DEFAULT_PROMPT_DIR = Path(__file__).parent / "prompts"


# ───────────────────────────────────────────────────────────────
# 1.  Helpers (aligned with spider_2_13)
# ───────────────────────────────────────────────────────────────


def column_name_of(fullname: str) -> str:
    if "." not in fullname:
        return fullname
    return fullname.split(".", 1)[1]


def sanitize_filename(name: str, max_length: int = 150) -> str:
    """
    Sanitize a string to be safe for use as a filename across platforms.
    - Keep alphanumerics, dot, underscore, and hyphen
    - Replace all other characters with underscore
    - Collapse consecutive underscores
    - Trim leading/trailing dots/underscores/spaces
    - Enforce a reasonable max length
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("._ ")
    if not safe:
        safe = "untitled"
    if len(safe) > max_length:
        safe = safe[:max_length]
    return safe


def ensure_unique_path(base_dir: Path, stem: str, suffix: str = ".txt") -> Path:
    return base_dir / f"{stem}{suffix}"


def generate_column_prompt(db_id: str, table_fullname: str, column_name: str,
                          column_type: str, column_values: List[str], db_type: str,
                          external_knowledge: str = "") -> str:
    sample_value_str = ""
    for i, value in enumerate(column_values):
        sample_value_str += f"Sample value {i+1}: {value}\n\n"
    if not sample_value_str:
        sample_value_str = 'No sample values available'

    prompt = f"""You are a database schema expert. Your task is to write a clear, concise description for a database column.

Database Context:
- Database ID: {db_id}
- Database Type: {db_type}
- Table: {table_fullname}
- Column Name: {column_name}
- Data Type: {column_type}

Sample Values:
{sample_value_str}

"""

    if external_knowledge:
        prompt += f"Additional Context:\n{external_knowledge}\n\n"

    prompt += """Based on the column name, data type, and sample values above, provide a clear description of what this column represents or contains. The description should be:
- Concise in 1 sentence
- Clear and understandable to database users
- Specific to the data content shown

Write 1 sentence for the meaning of the given column. Don't need to mention the column name again."""

    return prompt


# ───────────────────────────────────────────────────────────────
# 2.  Core logic
# ───────────────────────────────────────────────────────────────


def main() -> None:
    client = MongoClient(MONGO_URI)
    coll = client["mats"][COLLECTION_NAME]

    prompt_dir: Path = DEFAULT_PROMPT_DIR
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # Prepare tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")

    # Projection includes fields needed to build prompts
    projection = {
        "schema": 1,
        "column_info": 1,
        "sample_rows": 1,
        "evidence": 1,
        "db_id": 1,
        "db_type": 1,
        "generated_column_meaning": 1,
    }

    cursor = coll.find({"generated_column_meaning": {"$exists": True}}, projection)

    total_docs = 0
    total_prompts = 0
    total_tokens = 0
    written_keys: Set[str] = set()  # avoid duplicate db_id::table.col per run

    for doc in cursor:
        total_docs += 1
        db_id = doc.get("db_id")
        db_type = doc.get("db_type")
        column_info: Dict[str, Any] = doc.get("column_info", {}) or {}
        evidence = doc.get("evidence", "")
        gcm: Dict[str, Any] = doc.get("generated_column_meaning", {}) or {}

        # iterate existing generated meanings keys (table.column)
        keys: List[str] = []
        if isinstance(gcm, dict):
            for k in gcm.keys():
                if isinstance(k, str) and k:
                    keys.append(k)
        keys = sorted(keys)

        if not keys:
            continue

        db_prompt_dir = prompt_dir / str(db_id)
        if not args.dry_run:
            db_prompt_dir.mkdir(parents=True, exist_ok=True)

        for fullname in keys:
            if "." not in fullname:
                continue

            unique_key = f"{db_id}::{fullname}"
            if unique_key in written_keys:
                continue

            table_name, column_name = fullname.split(".", 1)

            # Column type
            col_type = "UNKNOWN"
            if fullname in column_info:
                col_type = column_info[fullname].get("type", "UNKNOWN")
            if col_type == "UNKNOWN":
                # keep behavior aligned with spider_2_13 (skip unknown types)
                continue

            # Sample values (first up to 5 similar values)
            column_values: List[str] = []
            if fullname in column_info:
                similar_values = column_info[fullname].get('similar_values', [])
                if isinstance(similar_values, list):
                    for value in similar_values[:5]:
                        if value is not None:
                            value_str = str(value).strip()
                            if value_str:
                                column_values.append(value_str)

            prompt = generate_column_prompt(
                db_id=db_id,
                table_fullname=table_name,
                column_name=fullname,
                column_type=col_type,
                column_values=column_values,
                db_type=db_type,
                external_knowledge=evidence,
            )

            # Write prompt file (unless dry-run)
            if not args.dry_run:
                safe_stem = sanitize_filename(fullname)
                prompt_path = ensure_unique_path(db_prompt_dir, safe_stem, ".txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(prompt)

            # Count tokens
            tokens = len(encoding.encode(prompt))
            total_tokens += tokens
            total_prompts += 1
            written_keys.add(unique_key)

    print(f"DOCS_SCANNED\t{total_docs}")
    print(f"PROMPTS_COUNT\t{total_prompts}")
    print(f"TOTAL_TOKENS\t{total_tokens}")
    if args.dry_run:
        print("Mode: dry-run (no prompt files written)")
    else:
        print(f"Prompts directory: {prompt_dir}")

    client.close()


if __name__ == "__main__":
    main() 