#!/usr/bin/env python3
"""
spider_2_8_add_has_null_values.py
─────────────────────────────────
Update only the `has_null` field in `column_info` for each column in Spider 2.0.

This script:
1. Loads documents from MongoDB.
2. For each column in the schema:
   - Checks if the column has any NULL values in similar_values
   - For SQLite databases, also checks the actual database for NULL values
   - Updates only the `has_null` field in `column_info`.
3. Preserves all existing fields (e.g., type, similar_values).

Usage
-----
python spider_2_8_add_has_null_values.py                    # process all samples
python spider_2_8_add_has_null_values.py --limit 10         # test with 10 samples
python spider_2_8_add_has_null_values.py --force-update     # force update all samples
"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm


# ───────────────────────────────────────────────────────────────
# 0.  CLI & ENV
# ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add has_null values to Spider2.0-lite samples.")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides)")
    p.add_argument("--collection-name", default="spider2_lite_samples",
                   help="MongoDB collection name (default: %(default)s)")
    p.add_argument("--base-dir", type=Path, default=Path("/home/datht/Spider2/spider2-lite"),
                   help="Root directory containing Spider2.0-lite dataset (default: %(default)s)")
    p.add_argument("--processes", type=int, default=8,
                   help="Worker processes (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of documents to process (for testing)")
    p.add_argument("--skip-processed", action="store_true", default=True,
                   help="Skip samples that already have has_null in column_info (default: True)")
    p.add_argument("--force-update", action="store_true",
                   help="Force update all samples, including those that already have has_null")
    p.add_argument("--table-mapping-dir", type=Path, default=Path("/home/datht/Spider2/data_processing/table_mapping"),
                   help="Directory containing table mapping JSON files (default: %(default)s)")
    return p.parse_args()


args = parse_args()
load_dotenv("/home/datht/mats/.env")
MONGO_URI = args.mongo_uri


# ───────────────────────────────────────────────────────────────
# 1.  Database mapping helpers
# ───────────────────────────────────────────────────────────────
def reverse_map_db_name(mapped_db_name: str) -> str:
    """Reverse map database names from mapped names back to original names for SQLite files."""
    # Reverse mappings for SQLite database files
    reverse_mappings = {
        "SQLITE_SAKILA": "sqlite-sakila",
        "DB_IMDB": "Db-IMDB",
    }
    
    return reverse_mappings.get(mapped_db_name, mapped_db_name)


def get_sqlite_db_path(db_id: str, base_dir: Path) -> Optional[Path]:
    """
    Get the path to the SQLite database file for a given database ID.
    
    Args:
        db_id: Database ID
        base_dir: Base directory containing the dataset
        
    Returns:
        Path to SQLite database file or None if not found
    """
    # Try the mapped name first (for SQLite databases)
    original_db_name = reverse_map_db_name(db_id)
    db_path = base_dir / "resource" / "databases" / "spider2-localdb" / f"{original_db_name}.sqlite"
    
    if db_path.exists():
        return db_path
    
    # Try the original db_id name
    db_path = base_dir / "resource" / "databases" / "spider2-localdb" / f"{db_id}.sqlite"
    
    if db_path.exists():
        return db_path
    
    return None


def check_null_in_sqlite(db_path: Path, table_name: str, column_name: str) -> Optional[bool]:
    """
    Check if a column has NULL values in SQLite database.
    
    Args:
        db_path: Path to SQLite database
        table_name: Table name
        column_name: Column name
        
    Returns:
        True if column has NULL values, False if not, None if error
    """
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check for NULL values
        cursor.execute(f'SELECT 1 FROM "{table_name}" WHERE "{column_name}" IS NULL LIMIT 1')
        result = cursor.fetchone()
        
        conn.close()
        return bool(result)
        
    except Exception as e:
        print(f"Error checking NULL in SQLite {db_path} for {table_name}.{column_name}: {e}")
        return None


def check_null_in_similar_values(similar_values: List[str]) -> bool:
    """
    Check if similar_values contains None or null-like values.
    
    Args:
        similar_values: List of similar values
        
    Returns:
        True if None/null-like values are found
    """
    if not similar_values:
        return False
    
    null_like_values = {'none', 'null', 'nil', 'nan', 'undefined', ''}
    
    for value in similar_values:
        if value is None:
            return True
        if isinstance(value, str):
            if value.lower() in null_like_values:
                return True
            if value.strip() == '':
                return True
    
    return False


def load_table_mappings_from_json(mapping_file: Path) -> Dict[str, Any]:
    """
    Load table name mappings from a JSON file.
    
    Args:
        mapping_file: Path to the JSON mapping file
        
    Returns:
        Dictionary containing the full mapping data including mappings and wildcard_patterns
    """
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Warning: Could not load mapping file {mapping_file}: {e}")
        return {}


def load_table_mappings_from_table_mapping_folder(db_id: str, table_mapping_dir: Path) -> Dict[str, Any]:
    """
    Load table name mappings from the table_mapping folder.
    
    Args:
        db_id: Database ID to look for
        table_mapping_dir: Path to the table_mapping directory
        
    Returns:
        Dictionary containing the full mapping data including mappings and wildcard_patterns
    """
    mapping_file = table_mapping_dir / f"{db_id}.json"
    return load_table_mappings_from_json(mapping_file)


# ───────────────────────────────────────────────────────────────
# 2.  Worker function
# ───────────────────────────────────────────────────────────────
def process_doc(task):
    idx, doc, base_dir = task
    client = MongoClient(MONGO_URI)
    coll = client["mats"][args.collection_name]

    db_id = doc["db_id"]
    db_type = doc["db_type"]
    schema = doc.get("schema", [])
    existing_col_info = doc.get("column_info", {})

    try:
        # Get SQLite database path if available
        sqlite_db_path = None
        if db_type == "sqlite":
            sqlite_db_path = get_sqlite_db_path(db_id, base_dir)
        
        # Start with existing column info
        col_info = dict(existing_col_info)
        
        for col in schema:
            try:
                table_name, column_name = col.split(".", 1)
            except ValueError:
                print(f"[{idx}] Invalid column name: {col}")
                continue
            
            # Get existing column data
            col_data = dict(existing_col_info.get(col, {}))
            
            # Check for NULL values
            has_null = None
            
            # First check similar_values for null-like values
            similar_values = col_data.get("similar_values", [])
            if similar_values:
                has_null = check_null_in_similar_values(similar_values)
                if has_null:
                    print(f"[{idx}] {col}: Found null-like values in similar_values")
            
            # For SQLite databases, also check the actual database
            if has_null is None and sqlite_db_path:
                db_has_null = check_null_in_sqlite(sqlite_db_path, table_name, column_name)
                if db_has_null is not None:
                    has_null = db_has_null
                    if has_null:
                        print(f"[{idx}] {col}: Found NULL values in SQLite database")
            
            # If we still don't know, default to False (no NULL values found)
            if has_null is None:
                has_null = False
            
            # Update the column data
            col_data['has_null'] = has_null
            col_info[col] = col_data
        
        return UpdateOne(
            {"_id": idx}, 
            {"$set": {"column_info": col_info}}
        )
        
    except Exception as exc:
        print(f"[{idx}] ERROR: {exc}")
        client.close()
        return None


# ───────────────────────────────────────────────────────────────
# 3.  Main function
# ───────────────────────────────────────────────────────────────
def main() -> None:
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    coll = client["mats"][args.collection_name]
    
    # Get documents to process
    if args.force_update:
        # Force update all samples
        docs_to_process = list(coll.find({}))
        print(f"FORCE UPDATE: Processing all {len(docs_to_process)} samples")
    elif args.skip_processed:
        # Skip samples that already have has_null in column_info
        docs_to_process = list(coll.find({
            "$or": [
                {"column_info": {"$exists": False}},
                {"column_info": {}},
                # Check if any column doesn't have has_null field
                {"column_info": {"$not": {"$elemMatch": {"has_null": {"$exists": True}}}}}
            ]
        }))
        print(f"Found {len(docs_to_process)} samples without has_null in column_info")
    else:
        # Process all samples
        docs_to_process = list(coll.find({}))
        print(f"Processing all {len(docs_to_process)} samples")
    
    if args.limit:
        docs_to_process = docs_to_process[:args.limit]
        print(f"Limited to {len(docs_to_process)} samples for testing")
    
    if not docs_to_process:
        print("No documents to process")
        client.close()
        return
    
    # Prepare tasks for multiprocessing
    tasks = [(doc["_id"], doc, args.base_dir) for doc in docs_to_process]
    
    # Process documents in parallel
    print(f"Processing {len(tasks)} documents with {args.processes} processes...")
    
    with Pool(processes=args.processes) as pool:
        updates = list(tqdm(
            pool.imap(process_doc, tasks, chunksize=4),
            total=len(tasks),
            desc="Processing documents"
        ))
    
    # Filter out None results and execute bulk write
    updates = [u for u in updates if u is not None]
    
    if updates:
        res = coll.bulk_write(updates, ordered=False)
        print(f"[{args.collection_name}] Updated {res.modified_count} documents with has_null values")
    else:
        print("No valid updates generated")
    
    # Print summary statistics
    total_with_column_info = coll.count_documents({"column_info": {"$exists": True}})
    total_docs = coll.count_documents({})
    
    print(f"\n=== Summary ===")
    print(f"Total documents in collection: {total_docs}")
    print(f"Documents with column_info: {total_with_column_info}")
    print(f"Coverage: {(total_with_column_info / total_docs * 100):.1f}%")
    
    # Show sample of has_null values for verification
    sample_doc = coll.find_one({"column_info": {"$exists": True}}, {"column_info": 1, "db_id": 1})
    if sample_doc:
        print(f"\nSample has_null values from {sample_doc.get('db_id', 'unknown')}:")
        column_info = sample_doc.get("column_info", {})
        has_null_count = 0
        total_columns = 0
        
        for col, info in list(column_info.items())[:5]:  # Show first 5 columns
            total_columns += 1
            has_null = info.get('has_null', 'NOT_SET')
            if has_null:
                has_null_count += 1
            print(f"  {col}: has_null={has_null}")
        
        print(f"Sample: {has_null_count}/{total_columns} columns have NULL values")
    
    client.close()


if __name__ == "__main__":
    main() 