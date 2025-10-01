#!/usr/bin/env python3
"""
spider_2_13_list_columns_without_meaning.py
────────────────────────────────────────────
Scan Spider2.0-lite samples in MongoDB and list schema columns that do not
have a meaning assigned in `column_meaning`.

Exclusions:
- Primary key columns (from `primary_keys`)
- Columns involved in foreign keys (`from` and `to` in `foreign_keys`)
- Columns whose name ends with "_id" (heuristic foreign key indicator)

Outputs a JSON report to an output directory within this folder.

Usage
-----
python spider_2_13_list_columns_without_meaning.py
python spider_2_13_list_columns_without_meaning.py --limit 20
python spider_2_13_list_columns_without_meaning.py --out-dir meaning_gaps
python spider_2_13_list_columns_without_meaning.py --target-ids 5 42 101
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
from tqdm import tqdm
import tiktoken
from openai import OpenAI


# ───────────────────────────────────────────────────────────────
# 0.  CLI & ENV
# ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List schema columns without meanings (excluding PKs/FKs and *_id)")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides)")
    p.add_argument("--collection-name", default="spider2_lite_samples",
                   help="MongoDB collection name (default: %(default)s)")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "meaning_gaps",
                   help="Directory to write JSON reports (will be created if missing)")
    p.add_argument("--prompt-dir", type=Path, default=Path(__file__).parent / "prompts",
                   help="Directory to write prompts (will be created if missing)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of documents to process (for testing)")
    p.add_argument("--target-ids", nargs="*", type=int, default=None,
                   help="Process specific document _id values (overrides --limit if provided)")
    p.add_argument("--filter", type=Path, default=None,
                   help="Path to low_recall.json file to filter columns (only process columns that appear in missing_cols)")
    p.add_argument("--overwrite-processed", action="store_true",
                   help="Regenerate and overwrite generated_column_meaning for already processed columns (default skips)")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not call OpenAI or update MongoDB; only generate prompts and compute token counts")
    return p.parse_args()


args = parse_args()
load_dotenv("/home/datht/mats/.env")
MONGO_URI = args.mongo_uri


# ───────────────────────────────────────────────────────────────
# 1.  Helpers
# ───────────────────────────────────────────────────────────────


def build_pk_fullname_set(primary_keys: Dict[str, List[str]]) -> Set[str]:
    fullname_set: Set[str] = set()
    for table, cols in (primary_keys or {}).items():
        for col in cols or []:
            fullname_set.add(f"{table}.{col}")
    return fullname_set


def build_fk_fullname_set(foreign_keys: Dict[str, List[Dict[str, Any]]]) -> Set[str]:
    fullname_set: Set[str] = set()
    for table, fk_list in (foreign_keys or {}).items():
        if not fk_list:
            continue
        for fk in fk_list:
            from_col = fk.get("from")
            to_col = fk.get("to")
            if from_col:
                fullname_set.add(f"{table}.{from_col}")
            if to_col:
                fullname_set.add(str(to_col))  # already "ref_table.ref_col"
    return fullname_set


def column_name_of(fullname: str) -> str:
    # Expecting "table.column"; return column part
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
    # Replace any character not in the allowed set with underscore
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe)
    # Strip leading/trailing unsafe characters
    safe = safe.strip("._ ")
    if not safe:
        safe = "untitled"
    # Truncate to max_length (reserve 4 for ".txt")
    if len(safe) > max_length:
        safe = safe[:max_length]
    return safe


def ensure_unique_path(base_dir: Path, stem: str, suffix: str = ".txt") -> Path:
    """Return the base path; overwrite if it exists (no numeric suffix)."""
    return base_dir / f"{stem}{suffix}"


def load_filter_columns(filter_path: Path) -> Set[str]:
    """
    Load column names from low_recall.json file.
    Parse column_desc field to extract table.column format.
    """
    if not filter_path or not filter_path.exists():
        return set()
    
    filter_columns = set()
    
    try:
        with open(filter_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            missing_cols = item.get('missing_cols', [])
            for col_info in missing_cols:
                column_desc = col_info.get('column_desc', '')
                if column_desc:
                    # Split by ';' and take first part, then strip
                    table_column_part = column_desc.split(';')[0].strip()
                    # Split by ' . ' to get table and column
                    if ' . ' in table_column_part:
                        table, column = table_column_part.split(' . ', 1)
                        table = table.strip()
                        column = column.strip()
                        if table and column:
                            filter_columns.add(f'{table}.{column}')
    
    except Exception as e:
        print(f'Warning: Could not load filter file {filter_path}: {e}')
        return set()
    
    print(f'Loaded {len(filter_columns)} columns from filter file: {filter_path}')
    return filter_columns


def generate_column_prompt(db_id: str, table_fullname: str, column_name: str, 
                          column_type: str, column_values: List[str], db_type: str,
                          external_knowledge: str = "") -> str:
    """
    Generate a prompt for writing column description.
    """
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


def save_prompts_to_files(report_items: List[Dict[str, Any]], 
                         prompt_dir: Path, 
                         coll,
                         dry_run: bool = False) -> int:
    """
    Save prompts for each missing column to individual files and return total token count.
    Also call OpenAI to generate a one-sentence meaning and store it in MongoDB under
    "generated_column_meaning.<table.column>" without modifying existing "column_meaning".
    When dry_run is True, skip calling OpenAI and skip MongoDB updates; only compute token counts (and write prompt files).
    """
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    
    total_tokens = 0
    
    # Avoid duplicate prompts within the same run (same db_id + column)
    written_keys: Set[str] = set()
    
    for item in report_items:
        doc_id = item["_id"]
        db_id = item["db_id"]
        db_type = item["db_type"]
        missing_columns = item["missing_columns"]
        
        # Get full document to extract column details
        doc = coll.find_one({"_id": doc_id})
        if not doc:
            continue
            
        schema = doc.get("schema", [])
        column_info = doc.get("column_info", {})
        sample_rows = doc.get("sample_rows", [])
        evidence = doc.get("evidence", "")
        generated_column_meaning: Dict[str, Any] = doc.get("generated_column_meaning", {}) or {}
        
        for column_fullname in missing_columns:
            if "." not in column_fullname:
                continue
                
            # Skip if we've already written this db_id + column_fullname in this run
            written_key = f"{db_id}::{column_fullname}"
            if written_key in written_keys:
                continue
            
            # By default, skip if DB already has a generated meaning, unless overwrite requested
            if isinstance(generated_column_meaning, dict) and not args.overwrite_processed:
                existing_val = generated_column_meaning.get(column_fullname)
                if isinstance(existing_val, str) and existing_val.strip():
                    # Print existing mapping when skipping
                    try:
                        print(f"{db_type}\t{db_id}\t{column_fullname}\t{existing_val}")
                    except Exception:
                        print(f"{db_type}\t{db_id}\t{column_fullname}\t<existing meaning printed>")
                    continue
            
            table_name, column_name = column_fullname.split(".", 1)
            
            # Get column type from column_info
            col_type = "UNKNOWN"
            if column_fullname in column_info:
                col_type = column_info[column_fullname].get("type", "UNKNOWN")
            
            # Skip columns with unknown types instead of raising exception
            if col_type == "UNKNOWN":
                print(f"Warning: Skipping column {column_fullname} in database {db_id} - column type is UNKNOWN")
                continue
            
            # Get sample values for this column from column_info similar_values
            column_values = []
            if column_fullname in column_info:
                similar_values = column_info[column_fullname].get('similar_values', [])
                if isinstance(similar_values, list):
                    for value in similar_values[:5]:  # Use first 5 similar values
                        if value is not None:
                            value_str = str(value).strip()
                            if value_str:
                                    column_values.append(value_str)
            
            # Generate prompt
            prompt = generate_column_prompt(
                db_id=db_id,
                table_fullname=table_name,
                column_name=column_fullname,
                column_type=col_type,
                column_values=column_values,
                db_type=db_type,
                external_knowledge=evidence
            )
            
            # Call OpenAI to get meaning
            meaning_text = None
            if not dry_run:
                meaning_text = generate_meaning_with_openai(prompt)
            if not dry_run and meaning_text:
                try:
                    # Update local cache with flat key like "TABLE.COLUMN"
                    if not isinstance(generated_column_meaning, dict):
                        generated_column_meaning = {}
                    generated_column_meaning[column_fullname] = meaning_text
                    # Write back the whole subdocument to preserve flat keys
                    coll.update_many(
                        {"db_id": db_id, "db_type": db_type},
                        {"$set": {"generated_column_meaning": generated_column_meaning}}
                    )
                except Exception as e:
                    print(f"Mongo update failed for {db_id}:{column_fullname} - {e}")
                # Print generated mapping
                try:
                    print(f"{db_type}\t{db_id}\t{column_fullname}\t{meaning_text}")
                except Exception:
                    print(f"{db_type}\t{db_id}\t{column_fullname}\t<generated meaning printed>")
            elif not dry_run and not meaning_text:
                print(f"{db_type}\t{db_id}\t{column_fullname}\t<no meaning generated>")
            
            # Save prompt to file: dir/db_id/table.column.txt (overwrites)
            if not dry_run:
                db_prompt_dir = prompt_dir / db_id
                db_prompt_dir.mkdir(parents=True, exist_ok=True)
                safe_stem = sanitize_filename(column_fullname)
                prompt_file = ensure_unique_path(db_prompt_dir, safe_stem, ".txt")
                with open(prompt_file, "w", encoding="utf-8") as f:
                    f.write(prompt)
            
            # Count tokens
            tokens = len(encoding.encode(prompt))
            total_tokens += tokens
            
            # Mark as written to avoid duplicates in this run
            written_keys.add(written_key)
            
    return total_tokens


# Initialize OpenAI client (expects OPENAI_API_KEY in environment)
_openai_client: OpenAI | None = None

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        # Ensure environment variables (including OPENAI_API_KEY) are loaded
        load_dotenv("/home/datht/mats/.env")
        try:
            _openai_client = OpenAI()
        except TypeError as e:
            # Will fallback in generate function
            _openai_client = None
    return _openai_client


def generate_meaning_with_openai(prompt: str, model: str = "gpt-4.1-mini") -> str:
    """Call OpenAI to get a concise one-sentence column meaning from the prompt.
    Falls back to legacy openai.ChatCompletion when needed.
    """
    # Try new client first
    try:
        client = get_openai_client()
        if client is not None:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text
    except TypeError as e:
        # Handle errors like unexpected keyword argument 'proxies' by falling back
        pass
    except Exception as e:
        print(f"OpenAI error: {e}")
        return ""

    # Fallback to legacy SDK
    try:
        import openai as openai_legacy  # type: ignore
        load_dotenv("/home/datht/mats/.env")
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            openai_legacy.api_key = api_key
        legacy_resp = openai_legacy.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        text = (legacy_resp["choices"][0]["message"]["content"] or "").strip()
        return text
    except Exception as e:
        print(f"OpenAI error (legacy): {e}")
        return ""


# ───────────────────────────────────────────────────────────────
# 2.  Core logic
# ───────────────────────────────────────────────────────────────


def find_columns_without_meaning(doc: Dict[str, Any], filter_columns: Set[str] | None = None) -> List[str]:
    schema: List[str] = doc.get("schema", []) or []
    column_meaning: Dict[str, str] = doc.get("column_meaning", {}) or {}
    primary_keys: Dict[str, List[str]] = doc.get("primary_keys", {}) or {}
    foreign_keys: Dict[str, List[Dict[str, Any]]] = doc.get("foreign_keys", {}) or {}

    schema_set: Set[str] = set(schema)

    # Missing meaning candidates: schema columns not in column_meaning, or meaning is empty/blank
    missing_candidates: List[str] = []
    for col in schema:
        meaning_val = column_meaning.get(col)
        if meaning_val is None:
            missing_candidates.append(col)
        else:
            if isinstance(meaning_val, str):
                if not meaning_val.strip():
                    missing_candidates.append(col)
            else:
                # Non-string or unexpected value -> treat as missing
                missing_candidates.append(col)

    # Build exclusion sets
    pk_full: Set[str] = build_pk_fullname_set(primary_keys)
    fk_full: Set[str] = build_fk_fullname_set(foreign_keys)

    # Apply exclusions
    filtered: List[str] = []
    for fullname in missing_candidates:
        col_name = column_name_of(fullname).lower()
        if fullname in pk_full:
            continue
        if fullname in fk_full:
            continue
        if col_name.endswith("_id"):
            continue
        # Exclude generic identifier columns
        if col_name == "uuid" or col_name == "id":
            continue
        # Apply filter if provided
        if filter_columns is not None and fullname not in filter_columns:
            continue
        # Keep only if still in schema
        if fullname in schema_set:
            filtered.append(fullname)

    return sorted(filtered)


# ───────────────────────────────────────────────────────────────
# 3.  Main
# ───────────────────────────────────────────────────────────────


def main() -> None:
    # Load filter columns if provided
    filter_columns = None
    if args.filter:
        filter_columns = load_filter_columns(args.filter)
        if not filter_columns:
            print("Warning: No filter columns loaded. Processing all columns.")
    
    client = MongoClient(MONGO_URI)
    coll = client["mats"][args.collection_name]

    # Determine documents to process
    query: Dict[str, Any] = {}
    projection = {"schema": 1, "column_meaning": 1, "primary_keys": 1, "foreign_keys": 1, 
                  "db_id": 1, "db_type": 1, "column_info": 1, "sample_rows": 1, "evidence": 1}

    if args.target_ids:
        query["_id"] = {"$in": args.target_ids}

    docs = list(coll.find(query, projection))

    if args.target_ids is None and args.limit is not None:
        docs = docs[: args.limit]

    # Prepare output directory and file name
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "columns_without_meaning.json"

    report_items: List[Dict[str, Any]] = []
    total_missing = 0
    all_missing_columns: Set[str] = set()  # For deduplication

    for doc in tqdm(docs, desc="Scanning documents"):
        missing_cols = find_columns_without_meaning(doc, filter_columns)
        # Only include databases that have missing columns
        if missing_cols:
            total_missing += len(missing_cols)
            all_missing_columns.update(missing_cols)  # Add to set for deduplication
            report_items.append({
                "_id": doc.get("_id"),
                "db_id": doc.get("db_id"),
                "db_type": doc.get("db_type"),
                "missing_count": len(missing_cols),
                "missing_columns": missing_cols,
            })

    report: Dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "collection": args.collection_name,
        "document_count": len(docs),
        "databases_with_missing_columns": len(report_items),
        "total_missing_columns": total_missing,
        "unique_missing_columns": len(all_missing_columns),
        "items": report_items,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote report to: {out_path}")
    print(f"Total unique columns without meaning (after dedup): {len(all_missing_columns)}")

    # Generate prompts for missing columns
    if report_items:
        print(f"\nGenerating prompts for {len(report_items)} databases with missing columns...")
        total_tokens = save_prompts_to_files(report_items, args.prompt_dir, coll, args.dry_run)
        if not args.dry_run:
            print(f"Generated prompts in: {args.prompt_dir}")
        else:
            print("Dry-run: skipped OpenAI calls, MongoDB updates, and writing prompt files.")
        print(f"Total tokens for all prompts: {total_tokens:,}")
    else:
        print("\nNo missing columns found - no prompts generated.")

    client.close()


if __name__ == "__main__":
    main() 