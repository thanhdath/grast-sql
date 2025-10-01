#!/usr/bin/env python3
"""
spider_2_6_add_column_meaning.py
────────────────────────────────────────────────
Enrich each document in spider2_lite_samples with:
    column_meaning = {
        "<table>.<column>": "description from JSON file",
        ...
    }

The script:
1. Loads the docs from MongoDB (skip if column_meaning already exists)
2. Reads column descriptions from JSON schema files
3. Creates column_meaning mapping from table.column to description
4. Bulk-updates the documents.

Usage
-----
python spider_2_6_add_column_meaning.py                    # process all samples
python spider_2_6_add_column_meaning.py --limit 10         # test with 10 samples
python spider_2_6_add_column_meaning.py --force-update     # force update all samples
"""
from __future__ import annotations
import argparse
import json
import os
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
    p = argparse.ArgumentParser(description="Add column meaning to Spider2.0-lite samples.")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides)")
    p.add_argument("--collection-name", default="spider2_lite_samples",
                   help="MongoDB collection name (default: %(default)s)")
    p.add_argument("--base-dir", type=Path, default=Path("/home/datht/Spider2/spider2-lite"),
                   help="Root directory containing Spider2.0-lite dataset (default: %(default)s)")
    p.add_argument("--table-mapping-dir", type=Path, default=Path("/home/datht/Spider2/data_processing/table_mapping"),
                   help="Directory containing table mapping JSON files (default: %(default)s)")
    p.add_argument("--processes", type=int, default=8,
                   help="Worker processes (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of documents to process (for testing)")
    p.add_argument("--skip-processed", action="store_true", default=True,
                   help="Skip samples that already have column_meaning (default: True)")
    p.add_argument("--force-update", action="store_true",
                   help="Force update all samples, including those that already have column_meaning")
    return p.parse_args()


args = parse_args()
load_dotenv("/home/datht/mats/.env")                       # makes sure .env is honoured
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


def get_schema_files_for_db(db_id: str, db_type: str, base_dir: Path) -> List[Path]:
    """
    Get all JSON schema files for a given database.
    
    Args:
        db_id: Database ID
        db_type: Database type (sqlite, bigquery, snowflake)
        base_dir: Base directory containing the dataset
        
    Returns:
        List of JSON schema file paths
    """
    schema_files = []
    
    if db_type == "sqlite":
        # For SQLite, use the db_id directly (the directories use the mapped names)
        db_path = base_dir / "resource" / "databases" / "sqlite" / db_id
        
        if db_path.exists():
            # Find all JSON files in the directory
            schema_files = list(db_path.glob("*.json"))
    
    elif db_type in ["bigquery", "snowflake"]:
        # For BigQuery and Snowflake, read from JSON schema files
        db_path = base_dir / "resource" / "databases" / db_type / db_id
        
        if db_path.exists():
            # Look for JSON files in subdirectories
            schema_files = list(db_path.rglob("*.json"))
    
    return schema_files


def load_column_meaning_from_json_files(schema_files: List[Path]) -> Dict[str, str]:
    """
    Load column meaning from JSON schema files.
    
    Args:
        schema_files: List of JSON schema file paths
        
    Returns:
        Dictionary mapping "table.column" to column meaning/description
    """
    column_meaning = {}
    
    for json_file in schema_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                table_name = data.get('table_name', '')
                column_names = data.get('column_names', [])
                descriptions = data.get('description', [])
                
                if not table_name or not column_names:
                    continue
                
                # Create column meaning for each column
                for i, col_name in enumerate(column_names):
                    description = descriptions[i] if i < len(descriptions) else None
                    
                    # Only assign meaning if description exists and is not empty
                    if description and isinstance(description, str) and description.strip():
                        meaning = description.strip()
                        column_key = f"{table_name}.{col_name}"
                        column_meaning[column_key] = meaning
                    
        except Exception as e:
            print(f"Warning: Could not read JSON schema file {json_file}: {e}")
            continue
    
    return column_meaning


def match_wildcard_pattern(pattern: str, table_name: str) -> bool:
    """
    Check if a table name matches a wildcard pattern.
    
    Args:
        pattern: Wildcard pattern like "events_*"
        table_name: Actual table name like "events_20210125"
        
    Returns:
        True if table_name matches the pattern
    """
    import re
    
    # Convert wildcard pattern to regex
    regex_pattern = pattern.replace("*", ".*")
    return re.match(regex_pattern, table_name) is not None


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


def get_column_meaning_for_schema_with_mappings(schema: List[str], all_column_meaning: Dict[str, str], table_mappings: Dict[str, Any]) -> Dict[str, str]:
    """
    Match schema columns (which may contain wildcards) with actual column meaning using table mappings.
    
    Args:
        schema: List of schema columns like ["events_*.event_timestamp", ...]
        all_column_meaning: Dictionary of all column meaning from JSON files
        table_mappings: Dictionary containing the full mapping data including mappings and wildcard_patterns
        
    Returns:
        Dictionary mapping schema columns to column meaning
    """
    matched_column_meaning = {}
    
    # Load the wildcard_patterns from the table mapping file
    wildcard_patterns = table_mappings.get("wildcard_patterns", {})
    
    for schema_col in schema:
        if '.' not in schema_col:
            continue
            
        table_pattern, column_name = schema_col.split('.', 1)
        
        # Check if this table pattern is a wildcard pattern that we have mappings for
        if table_pattern in wildcard_patterns:
            # This is a wildcard pattern - find the specific tables that match it
            specific_tables = wildcard_patterns[table_pattern]
            
            # Look for column meaning in any of the specific tables
            for specific_table in specific_tables:
                column_key = f"{specific_table}.{column_name}"
                if column_key in all_column_meaning:
                    matched_column_meaning[schema_col] = all_column_meaning[column_key]
                    break  # Use the first match for each schema column
        
        # If no match found through wildcard mappings, search through all_column_meaning for direct matches
        if schema_col not in matched_column_meaning:
            for col_key, meaning in all_column_meaning.items():
                if '.' not in col_key:
                    continue
                    
                actual_table, actual_column = col_key.split('.', 1)
                
                # Check if table matches wildcard pattern and column matches
                if (match_wildcard_pattern(table_pattern.lower(), actual_table.lower()) and 
                    actual_column.lower() == column_name.lower()):
                    matched_column_meaning[schema_col] = meaning
                    break  # Use the first match for each schema column
    
    return matched_column_meaning


def get_column_meaning_for_schema(schema: List[str], all_column_meaning: Dict[str, str]) -> Dict[str, str]:
    """
    Match schema columns (which may contain wildcards) with actual column meaning.
    
    Args:
        schema: List of schema columns like ["events_*.event_timestamp", ...]
        all_column_meaning: Dictionary of all column meaning from JSON files
        
    Returns:
        Dictionary mapping schema columns to column meaning
    """
    matched_column_meaning = {}
    
    for schema_col in schema:
        if '.' not in schema_col:
            continue
            
        table_pattern, column_name = schema_col.split('.', 1)
        
        # Find all columns that match this pattern
        for col_key, meaning in all_column_meaning.items():
            if '.' not in col_key:
                continue
                
            actual_table, actual_column = col_key.split('.', 1)
            
            # Check if table matches pattern and column matches (case-insensitive)
            if (match_wildcard_pattern(table_pattern.lower(), actual_table.lower()) and 
                actual_column.lower() == column_name.lower()):
                matched_column_meaning[schema_col] = meaning
                break  # Use the first match for each schema column
    
    return matched_column_meaning


# ───────────────────────────────────────────────────────────────
# 2.  Worker function
# ───────────────────────────────────────────────────────────────
def process_doc(task):
    idx, doc, base_dir, table_mapping_dir = task
    client = MongoClient(MONGO_URI)
    coll = client["mats"][args.collection_name]

    db_id = doc["db_id"]
    db_type = doc["db_type"]
    schema = doc.get("schema", [])

    try:
        # 1) Load schema files for this database
        schema_files = get_schema_files_for_db(db_id, db_type, base_dir)
        if not schema_files:
            print(f"[{idx}] WARNING: No schema files found for {db_id} ({db_type})")
            client.close()
            return None

        # 2) Load all column meanings from JSON files
        all_column_meaning = load_column_meaning_from_json_files(schema_files)

        # 3) Always attempt to load table mapping data (for reverse mapping)
        mapping_data: Dict[str, Any] = load_table_mappings_from_table_mapping_folder(db_id, table_mapping_dir)
        wildcard_patterns: Dict[str, List[str]] = mapping_data.get("wildcard_patterns", {}) if isinstance(mapping_data, dict) else {}

        # 4) Match schema to meanings
        # Unified strategy without fallback: 1) Direct exact lookups, 2) Wildcard mapping if available
        # 1) Direct exact lookups
        direct_column_meaning = {}
        for schema_col in schema:
            if schema_col in all_column_meaning:
                direct_column_meaning[schema_col] = all_column_meaning[schema_col]

        # 2) Wildcard mapping for remaining columns (if mappings exist)
        remaining_after_direct = [c for c in schema if c not in direct_column_meaning]
        wildcard_column_meaning = {}
        if wildcard_patterns and remaining_after_direct:
            print(f"[{idx}] {db_id}: Using table mappings for wildcard resolution")
            wildcard_column_meaning = get_column_meaning_for_schema_with_mappings(remaining_after_direct, all_column_meaning, mapping_data)

        # Combine results (direct takes precedence)
        column_meaning = {**wildcard_column_meaning, **direct_column_meaning}
        print(f"[{idx}] {db_id}: Direct={len(direct_column_meaning)}, Wildcard={len(wildcard_column_meaning)}; Total matched={len(column_meaning)} / {len(schema)}")

        print(f"[{idx}] {db_id}: Found {len(column_meaning)} columns with meaning out of {len(schema)} schema columns")

        return UpdateOne(
            {"_id": idx},
            {"$set": {"column_meaning": column_meaning}}
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
        # Skip samples that already have non-empty column_meaning
        docs_to_process = list(coll.find({
            "$or": [
                {"column_meaning": {"$exists": False}},
                {"column_meaning": {}}
            ]
        }))
        print(f"Found {len(docs_to_process)} samples without column_meaning or with empty column_meaning")
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
    tasks = [(doc["_id"], doc, args.base_dir, args.table_mapping_dir) for doc in docs_to_process]
    
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
        print(f"[{args.collection_name}] Updated {res.modified_count} documents with column_meaning")
    else:
        print("No valid updates generated")
    
    # Print summary statistics
    total_with_column_meaning = coll.count_documents({"column_meaning": {"$exists": True}})
    total_docs = coll.count_documents({})
    
    print(f"\n=== Summary ===")
    print(f"Total documents in collection: {total_docs}")
    print(f"Documents with column_meaning: {total_with_column_meaning}")
    print(f"Coverage: {(total_with_column_meaning / total_docs * 100):.1f}%")
    
    # Show sample of column_meaning for verification
    sample_doc = coll.find_one({"column_meaning": {"$exists": True}}, {"column_meaning": 1, "db_id": 1})
    if sample_doc:
        print(f"\nSample column_meaning from {sample_doc.get('db_id', 'unknown')}:")
        column_meaning = sample_doc.get("column_meaning", {})
        for col, meaning in list(column_meaning.items())[:3]:  # Show first 3 columns
            print(f"  {col}: {meaning}")
    
    # Validate that all column_meaning keys exist in schema
    print(f"\n=== Validation ===")
    validation_errors = []
    total_validated = 0
    
    for doc in coll.find({"column_meaning": {"$exists": True}}):
        doc_id = doc.get("_id")
        schema = doc.get("schema", [])
        column_meaning = doc.get("column_meaning", {})
        
        if not column_meaning:
            continue
            
        total_validated += 1
        invalid_keys = []
        
        for col_key in column_meaning.keys():
            if col_key not in schema:
                invalid_keys.append(col_key)
        
        if invalid_keys:
            validation_errors.append({
                "doc_id": doc_id,
                "db_id": doc.get("db_id"),
                "invalid_keys": invalid_keys[:5]  # Show first 5 invalid keys
            })
    
    print(f"Validated {total_validated} documents with column_meaning")
    
    if validation_errors:
        print(f"❌ Found {len(validation_errors)} documents with validation errors:")
        for error in validation_errors[:10]:  # Show first 10 errors
            print(f"  Document {error['doc_id']} ({error['db_id']}): {len(error['invalid_keys'])} invalid keys")
            print(f"    Sample invalid keys: {error['invalid_keys']}")
    else:
        print(f"✅ All {total_validated} documents passed validation!")
    
    client.close()


if __name__ == "__main__":
    main() 