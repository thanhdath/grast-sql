#!/usr/bin/env python3
"""
spider_2_7_add_foreign_keys.py
────────────────────────────────────────────────
Enrich each document in spider2_lite_samples with:
    foreign_keys = {
        "<table>": [
            {
                "from": "column_name",
                "to": "ref_table.ref_column",
                "ref_table": "ref_table",
                "on_update": "CASCADE|NO ACTION|...",
                "on_delete": "CASCADE|NO ACTION|...",
                "match": "SIMPLE|..."
            },
            ...
        ],
        ...
    }

The script:
1. Loads the docs from MongoDB
2. For SQLite: Extracts foreign keys from actual SQLite database files
3. For BigQuery/Snowflake: Attempts to extract from DDL.csv files (if available)
4. Normalizes foreign key table.column names to match schema
5. Bulk-updates the documents.

Usage
-----
python spider_2_7_add_foreign_keys.py                    # process all samples
python spider_2_7_add_foreign_keys.py --limit 10         # test with 10 samples
python spider_2_7_add_foreign_keys.py --force-update     # force update all samples
"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
import csv
import re
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
    p = argparse.ArgumentParser(description="Add foreign keys to Spider2.0-lite samples.")
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
                   help="Skip samples that already have foreign_keys (default: True)")
    p.add_argument("--force-update", action="store_true",
                   help="Force update all samples, including those that already have foreign_keys")
    p.add_argument("--table-mapping-dir", type=Path, default=None,
                   help="Directory containing JSON files for table-to-wildcard mappings per database.")
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


def get_sqlite_db_path(db_id: str, base_dir: Path) -> Optional[Path]:
    """Get the SQLite database file path."""
    original_db_name = reverse_map_db_name(db_id)
    db_path = base_dir / "resource" / "databases" / "spider2-localdb" / f"{original_db_name}.sqlite"
    
    if db_path.exists():
        return db_path
    return None


def get_ddl_csv_path(db_id: str, db_type: str, base_dir: Path) -> Optional[Path]:
    """Get the DDL.csv file path for a database."""
    if db_type == "sqlite":
        db_path = base_dir / "resource" / "databases" / "sqlite" / db_id / "DDL.csv"
    elif db_type in ["bigquery", "snowflake"]:
        # For BigQuery/Snowflake, DDL.csv might be in subdirectories
        db_dir = base_dir / "resource" / "databases" / db_type / db_id
        if db_dir.exists():
            # Look for DDL.csv in subdirectories
            ddl_files = list(db_dir.rglob("DDL.csv"))
            if ddl_files:
                return ddl_files[0]  # Return the first one found
    else:
        return None
    
    return None


# ───────────────────────────────────────────────────────────────
# 2.  Foreign key extraction functions
# ───────────────────────────────────────────────────────────────
def extract_foreign_keys_from_sqlite(db_path: Path, tables: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract foreign keys from SQLite database.
    
    Args:
        db_path: Path to SQLite database file
        tables: List of table names to check
        
    Returns:
        Dictionary mapping table names to list of foreign key info
    """
    fk_info = {}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        for table in tables:
            try:
                cursor.execute(f"PRAGMA foreign_key_list('{table}')")
                rows = cursor.fetchall()
                fk_list = []
                
                for row in rows:
                    # row format: (id, seq, table, from, to, on_update, on_delete, match)
                    to_col = row[4]  # referenced column
                    if to_col is None or to_col.strip().lower() == "none" or to_col.strip() == "":
                        continue
                    
                    fk_list.append({
                        "from": row[3],  # column in current table
                        "to": f"{row[2]}.{to_col}",  # referenced table.column
                        "ref_table": row[2],  # referenced table
                        "on_update": row[5] or "NO ACTION",
                        "on_delete": row[6] or "NO ACTION",
                        "match": row[7] or "SIMPLE"
                    })
                
                if fk_list:
                    fk_info[table] = fk_list
                    
            except Exception as e:
                print(f"Error fetching foreign keys for table {table}: {e}")
                continue
        
        conn.close()
        
    except Exception as e:
        print(f"Error opening SQLite database {db_path}: {e}")
    
    return fk_info


def extract_foreign_keys_from_ddl_csv(ddl_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract foreign keys from DDL.csv file.
    This is a fallback for BigQuery/Snowflake databases.
    
    Args:
        ddl_path: Path to DDL.csv file
        
    Returns:
        Dictionary mapping table names to list of foreign key info
    """
    fk_info = {}
    
    try:
        with open(ddl_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                table_name = row.get('table_name', '')
                ddl = row.get('ddl', '')
                
                if not table_name or not ddl:
                    continue
                
                # Try to extract foreign key constraints from DDL
                fk_constraints = extract_fk_from_ddl(ddl)
                if fk_constraints:
                    fk_info[table_name] = fk_constraints
                    
    except Exception as e:
        print(f"Error reading DDL.csv {ddl_path}: {e}")
    
    return fk_info


def extract_fk_from_ddl(ddl: str) -> List[Dict[str, Any]]:
    """
    Extract foreign key constraints from DDL string.
    
    Args:
        ddl: DDL string containing CREATE TABLE statement
        
    Returns:
        List of foreign key constraints
    """
    fk_constraints = []
    
    # Look for FOREIGN KEY patterns in DDL
    # Pattern: FOREIGN KEY (column) REFERENCES table (ref_column) [ON DELETE/UPDATE ...]
    fk_pattern = r'FOREIGN\s+KEY\s*\(\s*([^)]+)\s*\)\s+REFERENCES\s+([^\s(]+)\s*\(\s*([^)]+)\s*\)(?:\s+ON\s+(DELETE|UPDATE)\s+([^\s,)]+))?(?:\s+ON\s+(DELETE|UPDATE)\s+([^\s,)]+))?'
    
    matches = re.finditer(fk_pattern, ddl, re.IGNORECASE)
    
    for match in matches:
        from_col = match.group(1).strip()
        ref_table = match.group(2).strip()
        ref_column = match.group(3).strip()
        
        # Parse ON DELETE/UPDATE clauses
        on_delete = "NO ACTION"
        on_update = "NO ACTION"
        
        if match.group(4) and match.group(5):
            if match.group(4).upper() == "DELETE":
                on_delete = match.group(5).upper()
            elif match.group(4).upper() == "UPDATE":
                on_update = match.group(5).upper()
        
        if match.group(6) and match.group(7):
            if match.group(6).upper() == "DELETE":
                on_delete = match.group(7).upper()
            elif match.group(6).upper() == "UPDATE":
                on_update = match.group(7).upper()
        
        fk_constraints.append({
            "from": from_col,
            "to": f"{ref_table}.{ref_column}",
            "ref_table": ref_table,
            "on_update": on_update,
            "on_delete": on_delete,
            "match": "SIMPLE"
        })
    
    return fk_constraints


# ───────────────────────────────────────────────────────────────
# 3.  Normalization functions
# ───────────────────────────────────────────────────────────────
def _split_fullname(fullname: str) -> Optional[tuple[str, str]]:
    if '.' not in fullname:
        return None
    parts = fullname.split('.')
    table = parts[0]
    column = '.'.join(parts[1:])
    return table, column


def _match_wildcard_table(pattern: str, table_name: str) -> bool:
    """
    Return True if table_name matches the wildcard table pattern (e.g., events_*).
    """
    import re
    regex_pattern = '^' + re.escape(pattern).replace('\*', '.*') + '$'
    return re.match(regex_pattern, table_name) is not None


def _schema_contains_column(schema: List[str], fullname: str) -> bool:
    """
    Check whether a fully qualified column (table.column) exists in schema by direct membership.
    Assumes names have already been mapped to schema form.
    """
    return fullname in set(schema)


def _map_concrete_table_to_schema_table(concrete_table: str, wildcard_patterns: Dict[str, List[str]]) -> Optional[str]:
    """
    Using wildcard_patterns {wildcard: [specific_tables...]}, find the wildcard that owns the concrete_table.
    Returns the wildcard table name when found.
    """
    for wildcard_table, specific_list in (wildcard_patterns or {}).items():
        if concrete_table in (specific_list or []):
            return wildcard_table
    return None


def _map_to_schema_fullname(schema: List[str], concrete_table: str, column_name: str, wildcard_patterns: Dict[str, List[str]]) -> Optional[str]:
    """
    Given a concrete table and column, find the corresponding schema fullname
    (which may use a wildcard table pattern) and return it. Returns None if not found.
    Prefer mapping via wildcard_patterns; fallback to wildcard regex matching against schema entries.
    """
    # First try direct exact match
    candidate = f"{concrete_table}.{column_name}"
    if candidate in set(schema):
        return candidate
    # Try mapping via wildcard_patterns
    mapped_table = _map_concrete_table_to_schema_table(concrete_table, wildcard_patterns)
    if mapped_table:
        candidate = f"{mapped_table}.{column_name}"
        if candidate in set(schema):
            return candidate
    # Fallback: scan schema and match column name + wildcard pattern compatibility
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


def normalize_foreign_keys(fk_info: Dict[str, List[Dict[str, Any]]], schema: List[str], wildcard_patterns: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Normalize foreign keys to align with schema naming (including wildcard table patterns).
    This maps both source and target FK endpoints to their schema entries and groups FKs by the
    schema table name rather than the concrete source table name.
    """
    normalized_fk_info: Dict[str, List[Dict[str, Any]]] = {}
    
    for concrete_table, fk_list in fk_info.items():
        for fk in fk_list:
            from_col = fk['from']
            to_full = fk['to']  # format: ref_table.ref_column
            if '.' not in to_full:
                continue
            ref_table, ref_col = to_full.split('.', 1)
            
            # Map both ends to schema fullnames
            mapped_from_full = _map_to_schema_fullname(schema, concrete_table, from_col, wildcard_patterns)
            mapped_to_full = _map_to_schema_fullname(schema, ref_table, ref_col, wildcard_patterns)
            
            if not mapped_from_full or not mapped_to_full:
                print(f"Warning: Could not map FK {concrete_table}.{from_col} -> {to_full} to schema entries")
                continue
            
            schema_table, schema_from_col = mapped_from_full.split('.', 1)
            schema_ref_table, schema_ref_col = mapped_to_full.split('.', 1)
            
            mapped_fk = {
                "from": schema_from_col,
                "to": f"{schema_ref_table}.{schema_ref_col}",
                "ref_table": schema_ref_table,
                "on_update": fk.get("on_update", "NO ACTION"),
                "on_delete": fk.get("on_delete", "NO ACTION"),
                "match": fk.get("match", "SIMPLE")
            }
            
            normalized_fk_info.setdefault(schema_table, []).append(mapped_fk)
    
    return normalized_fk_info


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
# 4.  Worker function
# ───────────────────────────────────────────────────────────────
def process_doc(task):
    idx, doc, base_dir = task
    client = MongoClient(MONGO_URI)
    coll = client["mats"][args.collection_name]

    db_id = doc["db_id"]
    db_type = doc["db_type"]
    schema = doc.get("schema", [])

    try:
        fk_info = {}
        
        if db_type == "sqlite":
            # Extract from SQLite database
            db_path = get_sqlite_db_path(db_id, base_dir)
            if db_path:
                print(f"[{idx}] {db_id}: Extracting foreign keys from SQLite database")
                tables = sorted(set(c.split('.', 1)[0] for c in schema if '.' in c))
                fk_info = extract_foreign_keys_from_sqlite(db_path, tables)
            else:
                print(f"[{idx}] {db_id}: SQLite database file not found")
        
        elif db_type in ["bigquery", "snowflake"]:
            # Try to extract from DDL.csv
            ddl_path = get_ddl_csv_path(db_id, db_type, base_dir)
            if ddl_path:
                print(f"[{idx}] {db_id}: Extracting foreign keys from DDL.csv")
                fk_info = extract_foreign_keys_from_ddl_csv(ddl_path)
            else:
                print(f"[{idx}] {db_id}: DDL.csv not found for {db_type}")
        
        # Normalize foreign keys to match schema
        if fk_info:
            # Load wildcard patterns for this database if available
            wildcard_patterns: Dict[str, List[str]] = {}
            if args.table_mapping_dir:
                mapping_data: Dict[str, Any] = load_table_mappings_from_table_mapping_folder(db_id, args.table_mapping_dir)
                if isinstance(mapping_data, dict):
                    wildcard_patterns = mapping_data.get("wildcard_patterns", {}) or {}
            
            normalized_fk_info = normalize_foreign_keys(fk_info, schema, wildcard_patterns)
            print(f"[{idx}] {db_id}: Found {len(normalized_fk_info)} tables with foreign keys (normalized)")
        else:
            normalized_fk_info = {}
            print(f"[{idx}] {db_id}: No foreign keys found")
        
        return UpdateOne(
            {"_id": idx}, 
            {"$set": {"foreign_keys": normalized_fk_info}}
        )
        
    except Exception as exc:
        print(f"[{idx}] ERROR: {exc}")
        client.close()
        return None


# ───────────────────────────────────────────────────────────────
# 5.  Main function
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
        # Skip samples that already have non-empty foreign_keys
        docs_to_process = list(coll.find({
            "$or": [
                {"foreign_keys": {"$exists": False}},
                {"foreign_keys": {}}
            ]
        }))
        print(f"Found {len(docs_to_process)} samples without foreign_keys or with empty foreign_keys")
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
        print(f"[{args.collection_name}] Updated {res.modified_count} documents with foreign_keys")
    else:
        print("No valid updates generated")
    
    # Print summary statistics
    total_with_foreign_keys = coll.count_documents({"foreign_keys": {"$exists": True}})
    total_docs = coll.count_documents({})
    
    print(f"\n=== Summary ===")
    print(f"Total documents in collection: {total_docs}")
    print(f"Documents with foreign_keys: {total_with_foreign_keys}")
    print(f"Coverage: {(total_with_foreign_keys / total_docs * 100):.1f}%")
    
    # Show sample of foreign_keys for verification
    sample_doc = coll.find_one({"foreign_keys": {"$exists": True, "$ne": {}}}, {"foreign_keys": 1, "db_id": 1})
    if sample_doc:
        print(f"\nSample foreign_keys from {sample_doc.get('db_id', 'unknown')}:")
        foreign_keys = sample_doc.get("foreign_keys", {})
        for table, fk_list in list(foreign_keys.items())[:2]:  # Show first 2 tables
            print(f"  {table}: {len(fk_list)} foreign keys")
            for fk in fk_list[:2]:  # Show first 2 foreign keys
                print(f"    {fk['from']} -> {fk['to']}")
    
    # Validate that all foreign key table.column names exist in schema
    print(f"\n=== Validation ===")
    validation_errors = []
    total_validated = 0
    
    for doc in coll.find({"foreign_keys": {"$exists": True}}):
        doc_id = doc.get("_id")
        schema = doc.get("schema", [])
        foreign_keys = doc.get("foreign_keys", {})
        
        if not foreign_keys:
            continue
            
        total_validated += 1
        invalid_keys = []
        
        for table, fk_list in foreign_keys.items():
            for fk in fk_list:
                from_col = f"{table}.{fk['from']}"
                to_col = fk['to']
                
                if from_col not in schema:
                    invalid_keys.append(from_col)
                if to_col not in schema:
                    invalid_keys.append(to_col)
        
        if invalid_keys:
            validation_errors.append({
                "doc_id": doc_id,
                "db_id": doc.get("db_id"),
                "invalid_keys": list(set(invalid_keys))[:5]  # Show first 5 invalid keys
            })
    
    print(f"Validated {total_validated} documents with foreign_keys")
    
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