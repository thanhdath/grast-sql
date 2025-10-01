#!/usr/bin/env python3
"""
spider_2_5_add_column_type_and_similar_values.py
────────────────────────────────────────────────
Enrich each document in spider2_lite_samples with:
    column_info = {
        "<table>.<column>": {
            "type": "STRING" | "INTEGER" | "TEXT" | ...,
            "similar_values": [...]
        }, ...
    }

The script:
1. Loads the docs from MongoDB (skip if column_info already exists)
2. Reads column types and sample values from JSON schema files
3. Extracts similar values from sample_rows in JSON files
4. Bulk-updates the documents.

Usage
-----
python spider_2_5_add_column_type_and_similar_values.py                    # process all samples
python spider_2_5_add_column_type_and_similar_values.py --limit 10         # test with 10 samples
python spider_2_5_add_column_type_and_similar_values.py --mongo-uri mongodb://user:pass@host:27017
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
    p = argparse.ArgumentParser(description="Add column type and similar values to Spider2.0-lite samples.")
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
    p.add_argument("--target-ids", nargs="*", type=int, default=None,
                   help="Process specific document _id values (overrides --limit if provided)")
    p.add_argument("--skip-processed", action="store_true", default=True,
                   help="Skip samples that already have column_info (default: True)")
    p.add_argument("--force-update", action="store_true",
                   help="Force update all samples, including those that already have column_info")
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


def load_column_info_from_json_files(schema_files: List[Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load column information from JSON schema files.
    
    Args:
        schema_files: List of JSON schema file paths
        
    Returns:
        Dictionary mapping "table.column" to column info
    """
    column_info = {}
    
    for json_file in schema_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                table_name = data.get('table_name', '')
                column_names = data.get('column_names', [])
                column_types = data.get('column_types', [])
                sample_rows = data.get('sample_rows', [])
                
                if not table_name or not column_names:
                    continue
                
                # Create column info for each column
                for i, col_name in enumerate(column_names):
                    col_type = column_types[i] if i < len(column_types) else "unknown"
                    
                    # Extract similar values from sample rows
                    similar_values = []
                    for row in sample_rows:
                        if isinstance(row, dict) and col_name in row:
                            value = row[col_name]
                            if value is not None and str(value).strip() != "":
                                similar_values.append(str(value))
                    
                    # Limit to 5 similar values to avoid too much data
                    similar_values = list(set(similar_values))[:5]
                    
                    column_key = f"{table_name}.{col_name}"
                    column_info[column_key] = {
                        "type": col_type,
                        "similar_values": similar_values
                    }
                    
        except Exception as e:
            print(f"Warning: Could not read JSON schema file {json_file}: {e}")
            continue
    
    return column_info


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
            return data  # Return the full data, not just mappings
    except Exception as e:
        print(f"Warning: Could not load mapping file {mapping_file}: {e}")
        return {}


def load_table_mappings_from_table_mapping_folder(db_id: str, table_mapping_dir: Path) -> Dict[str, str]:
    """
    Load table name mappings from the table_mapping folder.
    
    Args:
        db_id: Database ID to look for
        table_mapping_dir: Path to the table_mapping directory
        
    Returns:
        Dictionary mapping table names to wildcard patterns
    """
    mapping_file = table_mapping_dir / f"{db_id}.json"
    return load_table_mappings_from_json(mapping_file)


def get_column_info_for_schema_with_mappings(schema: List[str], all_column_info: Dict[str, Dict[str, Any]], table_mappings: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Match schema columns (which may contain wildcards) with actual column info using table mappings.
    
    Args:
        schema: List of schema columns like ["events_*.event_timestamp", ...]
        all_column_info: Dictionary of all column info from JSON files
        table_mappings: Dictionary containing the full mapping data including mappings and wildcard_patterns
        
    Returns:
        Dictionary mapping schema columns to column info
    """
    matched_column_info = {}
    
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
            
            # Look for column info in any of the specific tables
            for specific_table in specific_tables:
                column_key = f"{specific_table}.{column_name}"
                if column_key in all_column_info:
                    matched_column_info[schema_col] = all_column_info[column_key]
                    break  # Use the first match for each schema column
        
        # If no match found through wildcard mappings, search through all_column_info for direct matches
        if schema_col not in matched_column_info:
            for col_key, col_info in all_column_info.items():
                if '.' not in col_key:
                    continue
                    
                actual_table, actual_column = col_key.split('.', 1)
                
                # Check if table matches wildcard pattern and column matches
                if (match_wildcard_pattern(table_pattern.lower(), actual_table.lower()) and 
                    actual_column.lower() == column_name.lower()):
                    matched_column_info[schema_col] = col_info
                    break  # Use the first match for each schema column
    
    return matched_column_info


def get_column_info_for_schema(schema: List[str], all_column_info: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Match schema columns (which may contain wildcards) with actual column info.
    
    Args:
        schema: List of schema columns like ["events_*.event_timestamp", ...]
        all_column_info: Dictionary of all column info from JSON files
        
    Returns:
        Dictionary mapping schema columns to column info
    """
    matched_column_info = {}
    
    for schema_col in schema:
        if '.' not in schema_col:
            continue
            
        table_pattern, column_name = schema_col.split('.', 1)
        
        # Find all columns that match this pattern
        for col_key, col_info in all_column_info.items():
            if '.' not in col_key:
                continue
                
            actual_table, actual_column = col_key.split('.', 1)
            
            # Check if table matches pattern and column matches (case-insensitive)
            if (match_wildcard_pattern(table_pattern.lower(), actual_table.lower()) and 
                actual_column.lower() == column_name.lower()):
                matched_column_info[schema_col] = col_info
                break  # Use the first match for each schema column
    
    return matched_column_info


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
        # Get schema files for this database
        schema_files = get_schema_files_for_db(db_id, db_type, base_dir)
        
        if not schema_files:
            print(f"[{idx}] WARNING: No schema files found for {db_id} ({db_type})")
            client.close()
            return None
        
        # Load column information from JSON files
        all_column_info = load_column_info_from_json_files(schema_files)
        
        # Load table mappings from the table_mapping folder
        table_mappings = load_table_mappings_from_table_mapping_folder(db_id, table_mapping_dir)
        
        # Unified strategy across DB types:
        # 1) Direct exact lookups
        # 2) Wildcard mapping (if available)
        direct_column_info = {}
        for schema_col in schema:
            if schema_col in all_column_info:
                direct_column_info[schema_col] = all_column_info[schema_col]
        
        remaining_after_direct = [c for c in schema if c not in direct_column_info]
        
        wildcard_column_info = {}
        if remaining_after_direct:
            print(f"[{idx}] {db_id}: Using table mappings for wildcard resolution")
            wildcard_column_info = get_column_info_for_schema_with_mappings(remaining_after_direct, all_column_info, table_mappings)
        
        # Combine results (direct takes precedence)
        column_info = {**wildcard_column_info, **direct_column_info}
        
        print(f"[{idx}] {db_id}: Direct={len(direct_column_info)}, Wildcard={len(wildcard_column_info)}; Total matched={len(column_info)} / {len(schema)}")
        
        return UpdateOne(
            {"_id": idx}, 
            {"$set": {"column_info": column_info}}
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
        # Skip samples that already have non-empty column_info
        docs_to_process = list(coll.find({
            "$or": [
                {"column_info": {"$exists": False}},
                {"column_info": {}}
            ]
        }))
        print(f"Found {len(docs_to_process)} samples without column_info or with empty column_info")
    else:
        # Process all samples
        docs_to_process = list(coll.find({}))
        print(f"Processing all {len(docs_to_process)} samples")
    
    if args.limit:
        docs_to_process = docs_to_process[:args.limit]
        print(f"Limited to {len(docs_to_process)} samples for testing")
    
    if args.target_ids:
        # Filter docs_to_process to only include documents with target_ids
        docs_to_process = [doc for doc in docs_to_process if doc.get("_id") in args.target_ids]
        print(f"Processing only {len(docs_to_process)} samples with target IDs: {args.target_ids}")
    
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
        print(f"[{args.collection_name}] Updated {res.modified_count} documents with column_info")
    else:
        print("No valid updates generated")
    
    # Print summary statistics
    total_with_column_info = coll.count_documents({"column_info": {"$exists": True}})
    total_docs = coll.count_documents({})
    
    print(f"\n=== Summary ===")
    print(f"Total documents in collection: {total_docs}")
    print(f"Documents with column_info: {total_with_column_info}")
    print(f"Coverage: {(total_with_column_info / total_docs * 100):.1f}%")
    
    # Show sample of column_info for verification
    sample_doc = coll.find_one({"column_info": {"$exists": True}}, {"column_info": 1, "db_id": 1})
    if sample_doc:
        print(f"\nSample column_info from {sample_doc.get('db_id', 'unknown')}:")
        column_info = sample_doc.get("column_info", {})
        for col, info in list(column_info.items())[:3]:  # Show first 3 columns
            print(f"  {col}: type={info.get('type', 'unknown')}, values={info.get('similar_values', [])[:3]}")
    
    # Validate that all column_info keys exist in schema
    print(f"\n=== Validation ===")
    validation_errors = []
    total_validated = 0
    
    for doc in coll.find({"column_info": {"$exists": True}}):
        doc_id = doc.get("_id")
        schema = doc.get("schema", [])
        column_info = doc.get("column_info", {})
        
        if not column_info:
            continue
            
        total_validated += 1
        invalid_keys = []
        
        for col_key in column_info.keys():
            if col_key not in schema:
                invalid_keys.append(col_key)
        
        if invalid_keys:
            validation_errors.append({
                "doc_id": doc_id,
                "db_id": doc.get("db_id"),
                "invalid_keys": invalid_keys[:5]  # Show first 5 invalid keys
            })
    
    print(f"Validated {total_validated} documents with column_info")
    
    if validation_errors:
        print(f"❌ Found {len(validation_errors)} documents with validation errors:")
        for error in validation_errors[:10]:  # Show first 10 errors
            print(f"  Document {error['doc_id']} ({error['db_id']}): {len(error['invalid_keys'])} invalid keys")
            print(f"    Sample invalid keys: {error['invalid_keys']}")
    else:
        print(f"✅ All {total_validated} documents passed validation!")
    
    # Additional completeness check: ensure every schema column has column_info
    print(f"\n=== Completeness Check (schema → column_info) ===")
    completeness_issues = []
    total_checked = 0
    for doc in coll.find({"column_info": {"$exists": True}}):
        doc_id = doc.get("_id")
        db_id = doc.get("db_id")
        schema = doc.get("schema", [])
        column_info = doc.get("column_info", {})
        if not schema:
            continue
        total_checked += 1
        missing_cols = [col for col in schema if col not in column_info]
        if missing_cols:
            completeness_issues.append({
                "doc_id": doc_id,
                "db_id": db_id,
                "num_missing": len(missing_cols),
                "sample_missing": missing_cols[:5]
            })
    print(f"Checked {total_checked} documents with schema")
    if completeness_issues:
        print(f"❌ {len(completeness_issues)} documents have schema columns missing in column_info")
        for issue in completeness_issues[:10]:
            print(f"  Document {issue['doc_id']} ({issue['db_id']}): {issue['num_missing']} missing")
            print(f"    Sample missing: {issue['sample_missing']}")
    else:
        print("✅ All documents have column_info for every schema column!")
    
    client.close()


if __name__ == "__main__":
    main() 