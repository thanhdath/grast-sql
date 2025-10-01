#!/usr/bin/env python3
"""
spider_2_9_add_primary_key_info.py
──────────────────────────────────
Annotate MongoDB documents with primary-key information for Spider 2.0.

For every document in spider2_lite_samples we:
    • For SQLite databases: open the .sqlite file and run PRAGMA table_info(table)
    • For BigQuery/Snowflake: try to extract from DDL.csv files
    • Collect columns where pk > 0 (composite-key order preserved)
    • Write {table: ["col1", "col2", …]} to primary_keys field

Usage
-----
python spider_2_9_add_primary_key_info.py                    # process all samples
python spider_2_9_add_primary_key_info.py --limit 10         # test with 10 samples
python spider_2_9_add_primary_key_info.py --force-update     # force update all samples
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm


# ───────────────────────────────────────────────────────────────
# 0.  CLI & ENV
# ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add primary key information to Spider2.0-lite samples.")
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
                   help="Skip samples that already have primary_keys (default: True)")
    p.add_argument("--force-update", action="store_true",
                   help="Force update all samples, including those that already have primary_keys")
    p.add_argument("--table-mapping-dir", type=Path,
                   help="Directory containing JSON mapping files for table names (e.g., for wildcard patterns)")
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


def get_ddl_csv_path(db_id: str, db_type: str, base_dir: Path) -> Optional[Path]:
    """
    Get the path to the DDL.csv file for a given database.
    
    Args:
        db_id: Database ID
        db_type: Database type (bigquery, snowflake)
        base_dir: Base directory containing the dataset
        
    Returns:
        Path to DDL.csv file or None if not found
    """
    db_dir = base_dir / "resource" / "databases" / db_type / db_id
    
    if db_dir.exists():
        # Look for DDL.csv recursively
        ddl_files = list(db_dir.rglob("DDL.csv"))
        if ddl_files:
            return ddl_files[0]  # Return the first one found
    
    return None


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


def _match_wildcard_table(pattern: str, table_name: str) -> bool:
    import re
    regex_pattern = '^' + re.escape(pattern).replace('\\*', '.*') + '$'
    return re.match(regex_pattern, table_name) is not None


def _split_fullname(fullname: str) -> Optional[tuple[str, str]]:
    if '.' not in fullname:
        return None
    parts = fullname.split('.')
    table = parts[0]
    column = '.'.join(parts[1:])
    return table, column


def _map_concrete_table_to_schema_table(concrete_table: str, wildcard_patterns: Dict[str, List[str]]) -> Optional[str]:
    for wildcard_table, specific_list in (wildcard_patterns or {}).items():
        if concrete_table in (specific_list or []):
            return wildcard_table
    return None


def _map_to_schema_fullname(schema: List[str], concrete_table: str, column_name: str, wildcard_patterns: Dict[str, List[str]]) -> Optional[str]:
    candidate = f"{concrete_table}.{column_name}"
    if candidate in set(schema):
        return candidate
    mapped_table = _map_concrete_table_to_schema_table(concrete_table, wildcard_patterns)
    if mapped_table:
        candidate = f"{mapped_table}.{column_name}"
        if candidate in set(schema):
            return candidate
    target_col_lower = column_name.lower()
    for schema_entry in schema:
        split_schema = _split_fullname(schema_entry)
        if not split_schema:
            continue
        schema_table, schema_col = split_schema
        if schema_col.lower() != target_col_lower:
            continue
        if _match_wildcard_table(schema_table, concrete_table):
            return schema_entry
    return None


def normalize_primary_keys(pk_info: Dict[str, List[str]], schema: List[str], wildcard_patterns: Dict[str, List[str]]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    schema_set = set(schema)
    for concrete_table, pk_cols in pk_info.items():
        mapped_table = _map_concrete_table_to_schema_table(concrete_table, wildcard_patterns) or concrete_table
        out_cols: List[str] = []
        for col in pk_cols:
            full_concrete = f"{concrete_table}.{col}"
            if full_concrete in schema_set:
                out_cols.append(col)
                continue
            # Try mapped table name
            full_mapped = f"{mapped_table}.{col}"
            if full_mapped in schema_set:
                out_cols.append(col)
                continue
            # Fallback: scan schema
            candidate = _map_to_schema_fullname(schema, concrete_table, col, wildcard_patterns)
            if candidate:
                # ensure table match for grouping
                _, resolved_col = candidate.split('.', 1)
                out_cols.append(resolved_col)
        if out_cols:
            # group under the schema table name if available
            group_table = mapped_table if any(f"{mapped_table}.{c}" in schema_set for c in out_cols) else concrete_table
            normalized[group_table] = out_cols
    return normalized


# ───────────────────────────────────────────────────────────────
# 2.  Primary key extraction functions
# ───────────────────────────────────────────────────────────────
def get_primary_keys_from_sqlite(db_path: Path, tables: List[str]) -> Dict[str, List[str]]:
    """
    Extract primary keys from SQLite database using PRAGMA table_info.
    
    Args:
        db_path: Path to SQLite database
        tables: List of table names to check
        
    Returns:
        Dictionary mapping table names to lists of primary key columns
    """
    pk_info: Dict[str, List[str]] = {}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        for table in tables:
            try:
                cursor.execute(f"PRAGMA table_info('{table}')")
                rows = cursor.fetchall()  # cid, name, type, notnull, dflt, pk
                
                # Get columns where pk > 0, ordered by pk value
                pk_columns = [row[1] for row in sorted(rows, key=lambda r: r[5]) if row[5] > 0]
                
                if pk_columns:
                    pk_info[table] = pk_columns
                    
            except Exception as e:
                print(f"Error fetching PKs for table {table}: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error opening SQLite database {db_path}: {e}")
    
    return pk_info


def extract_primary_keys_from_ddl_csv(ddl_path: Path) -> Dict[str, List[str]]:
    """
    Extract primary keys from DDL.csv file.
    
    Args:
        ddl_path: Path to DDL.csv file
        
    Returns:
        Dictionary mapping table names to lists of primary key columns
    """
    pk_info: Dict[str, List[str]] = {}
    
    try:
        # Try a couple of common encodings only, avoid optional dependencies
        ddl_content = None
        for encoding in ['utf-8', 'latin-1']:
            try:
                ddl_content = ddl_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if ddl_content is None:
            print(f"Could not read DDL.csv {ddl_path} with utf-8 or latin-1 encodings")
            return pk_info
        
        # Extract primary keys from DDL content
        pk_info = extract_pk_from_ddl(ddl_content)
        
    except Exception as e:
        print(f"Error reading DDL.csv {ddl_path}: {e}")
    
    return pk_info


def extract_pk_from_ddl(ddl: str) -> Dict[str, List[str]]:
    """
    Extract primary key constraints from DDL string.
    
    Args:
        ddl: DDL string containing CREATE TABLE statements
        
    Returns:
        Dictionary mapping table names to lists of primary key columns
    """
    pk_info: Dict[str, List[str]] = {}
    
    # Look for CREATE TABLE statements
    create_table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:`?(\w+)`?\.)?`?(\w+)`?\s*\((.*?)\)'
    
    matches = re.finditer(create_table_pattern, ddl, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        schema_name = match.group(1)
        table_name = match.group(2)
        table_body = match.group(3)
        
        # Look for PRIMARY KEY constraints
        pk_columns = []
        
        # Pattern 1: PRIMARY KEY (col1, col2, ...)
        pk_pattern1 = r'PRIMARY\s+KEY\s*\(\s*([^)]+)\s*\)'
        pk_match1 = re.search(pk_pattern1, table_body, re.IGNORECASE)
        
        if pk_match1:
            pk_cols_str = pk_match1.group(1)
            # Split by comma and clean up
            pk_columns = [col.strip().strip('`"') for col in pk_cols_str.split(',')]
        
        # Pattern 2: column_name data_type PRIMARY KEY
        pk_pattern2 = r'(\w+)\s+\w+(?:\s*\(\s*\d+\s*\))?\s+PRIMARY\s+KEY'
        pk_matches2 = re.finditer(pk_pattern2, table_body, re.IGNORECASE)
        
        for pk_match2 in pk_matches2:
            col_name = pk_match2.group(1)
            if col_name not in pk_columns:
                pk_columns.append(col_name)
        
        # Pattern 3: column_name data_type PRIMARY KEY AUTO_INCREMENT
        pk_pattern3 = r'(\w+)\s+\w+(?:\s*\(\s*\d+\s*\))?\s+PRIMARY\s+KEY\s+AUTO_INCREMENT'
        pk_matches3 = re.finditer(pk_pattern3, table_body, re.IGNORECASE)
        
        for pk_match3 in pk_matches3:
            col_name = pk_match3.group(1)
            if col_name not in pk_columns:
                pk_columns.append(col_name)
        
        if pk_columns:
            # Use full table name if schema is provided
            full_table_name = f"{schema_name}.{table_name}" if schema_name else table_name
            pk_info[full_table_name] = pk_columns
    
    return pk_info


# ───────────────────────────────────────────────────────────────
# 3.  Worker function
# ───────────────────────────────────────────────────────────────
def process_doc(task):
    idx, doc, base_dir = task
    client = MongoClient(MONGO_URI)
    coll = client["mats"][args.collection_name]

    db_id = doc["db_id"]
    db_type = doc["db_type"]
    schema = doc.get("schema", [])

    try:
        # Extract table names from schema
        tables = sorted(set(c.split(".", 1)[0] for c in schema if "." in c))
        
        pk_info = {}
        
        if db_type == "sqlite":
            # Extract from SQLite database
            db_path = get_sqlite_db_path(db_id, base_dir)
            if db_path:
                print(f"[{idx}] {db_id}: Extracting primary keys from SQLite database")
                pk_info = get_primary_keys_from_sqlite(db_path, tables)
            else:
                print(f"[{idx}] {db_id}: SQLite database file not found")
        
        elif db_type in ["bigquery", "snowflake"]:
            # Try to extract from DDL.csv
            ddl_path = get_ddl_csv_path(db_id, db_type, base_dir)
            if ddl_path:
                print(f"[{idx}] {db_id}: Extracting primary keys from DDL.csv")
                pk_info = extract_primary_keys_from_ddl_csv(ddl_path)
            else:
                print(f"[{idx}] {db_id}: DDL.csv not found for {db_type}")
        
        if pk_info:
            wildcard_patterns: Dict[str, List[str]] = {}
            # Load wildcard patterns when available
            if hasattr(args, 'table_mapping_dir') and args.table_mapping_dir:
                mapping_data: Dict[str, Any] = load_table_mappings_from_table_mapping_folder(db_id, args.table_mapping_dir)
                if isinstance(mapping_data, dict):
                    wildcard_patterns = mapping_data.get("wildcard_patterns", {}) or {}
            normalized_pk_info = normalize_primary_keys(pk_info, schema, wildcard_patterns)
            print(f"[{idx}] {db_id}: Found {len(normalized_pk_info)} tables with primary keys (normalized)")
        else:
            normalized_pk_info = {}
            print(f"[{idx}] {db_id}: No primary keys found")
        
        return UpdateOne(
            {"_id": idx}, 
            {"$set": {"primary_keys": normalized_pk_info}}
        )
        
    except Exception as exc:
        print(f"[{idx}] ERROR: {exc}")
        client.close()
        return None


# ───────────────────────────────────────────────────────────────
# 4.  Main function
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
        # Skip samples that already have non-empty primary_keys
        docs_to_process = list(coll.find({
            "$or": [
                {"primary_keys": {"$exists": False}},
                {"primary_keys": {}}
            ]
        }))
        print(f"Found {len(docs_to_process)} samples without primary_keys or with empty primary_keys")
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
        print(f"[{args.collection_name}] Updated {res.modified_count} documents with primary_keys")
    else:
        print("No valid updates generated")
    
    # Print summary statistics
    total_with_primary_keys = coll.count_documents({"primary_keys": {"$exists": True}})
    total_docs = coll.count_documents({})
    
    print(f"\n=== Summary ===")
    print(f"Total documents in collection: {total_docs}")
    print(f"Documents with primary_keys: {total_with_primary_keys}")
    print(f"Coverage: {(total_with_primary_keys / total_docs * 100):.1f}%")
    
    # Show sample of primary_keys for verification
    sample_doc = coll.find_one({"primary_keys": {"$exists": True, "$ne": {}}}, {"primary_keys": 1, "db_id": 1})
    if sample_doc:
        print(f"\nSample primary_keys from {sample_doc.get('db_id', 'unknown')}:")
        primary_keys = sample_doc.get("primary_keys", {})
        for table, pk_cols in list(primary_keys.items())[:3]:  # Show first 3 tables
            print(f"  {table}: {pk_cols}")
    
    # Validate that all primary key table.column names exist in schema
    print(f"\n=== Validation ===")
    validation_errors = []
    total_validated = 0
    for doc in coll.find({"primary_keys": {"$exists": True}}):
        doc_id = doc.get("_id")
        db_id = doc.get("db_id")
        schema = doc.get("schema", [])
        pk_map = doc.get("primary_keys", {})
        if not pk_map:
            continue
        total_validated += 1
        schema_set = set(schema)
        invalid_keys: List[str] = []
        for table, pk_cols in pk_map.items():
            for col in pk_cols:
                full = f"{table}.{col}"
                if full not in schema_set:
                    invalid_keys.append(full)
        if invalid_keys:
            validation_errors.append({
                "doc_id": doc_id,
                "db_id": db_id,
                "invalid_keys": invalid_keys[:5]
            })
    print(f"Validated {total_validated} documents with primary_keys")
    if validation_errors:
        print(f"❌ Found {len(validation_errors)} documents with validation errors:")
        for error in validation_errors[:10]:
            print(f"  Document {error['doc_id']} ({error['db_id']}): {len(error['invalid_keys'])} invalid keys")
            print(f"    Sample invalid keys: {error['invalid_keys']}")
    else:
        print(f"✅ All {total_validated} documents passed validation!")
    
    client.close()


if __name__ == "__main__":
    main() 