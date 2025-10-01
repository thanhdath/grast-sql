#!/usr/bin/env python3
"""
spider_2_3_fix_missing_cases_in_column_usage.py
────────────────────────────────────────────────
Repair documents in the MongoDB collection spider2_lite_samples whose
`used_columns` list is empty, missing, or contains items that do not exist
in the database schema (case-insensitive).

Typical usage
-------------
python spider_2_3_fix_missing_cases_in_column_usage.py --limit 10  # test with 10 samples
python spider_2_3_fix_missing_cases_in_column_usage.py              # process all samples
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
from pymongo import MongoClient


# ───────────────────────────────────────────────────────────────
# 0.  CLI & ENV
# ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repair used_columns mismatches for Spider2.0-lite.")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides).")
    p.add_argument("--collection-name", default="spider2_lite_samples",
                   help="MongoDB collection name (default: %(default)s)")
    p.add_argument("--processes", type=int, default=16,
                   help="Worker processes (default: %(default)s).")
    p.add_argument("--base-dir", type=Path, default=Path("/home/datht/Spider2/spider2-lite"),
                   help="Root directory containing Spider2.0-lite dataset (default: %(default)s)")
    p.add_argument("--table-mapping-dir", type=Path, default=Path("/home/datht/Spider2/data_processing/table_mapping"),
                   help="Directory containing table mapping JSON files (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of documents to process (for testing)")
    return p.parse_args()


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


args = parse_args()
load_dotenv("/home/datht/mats/.env")
MONGO_URI = args.mongo_uri

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ───────────────────────────────────────────────────────────────
# 1.  UTILITIES
# ───────────────────────────────────────────────────────────────
def reverse_map_db_name(mapped_db_name: str) -> str:
    """Reverse map database names from mapped names back to original names for SQLite files."""
    # Reverse mappings for SQLite database files
    reverse_mappings = {
        "SQLITE_SAKILA": "sqlite-sakila",
        "DB_IMDB": "Db-IMDB",
    }
    
    return reverse_mappings.get(mapped_db_name, mapped_db_name)


def get_schema_for_db_type(db_id: str, db_type: str, base_dir: Path) -> List[str]:
    """Get schema for different database types in Spider2.0-lite."""
    schema = []
    
    if db_type == "sqlite":
        # For SQLite, use the SQLite database directly
        # Need to reverse map the db_id to get the original database name
        original_db_name = reverse_map_db_name(db_id)
        db_path = base_dir / "resource" / "databases" / "spider2-localdb" / f"{original_db_name}.sqlite"
        
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                table_names = [result[0].lower() for result in cursor.fetchall()]
                
                # For each table, get column names
                for table_name in table_names:
                    if table_name == "sqlite_sequence":  # Skip SQLite system table
                        continue
                    
                    cursor.execute(f"SELECT name FROM PRAGMA_TABLE_INFO('{table_name}')")
                    column_names = [result[0].lower() for result in cursor.fetchall()]
                    
                    for column_name in column_names:
                        schema.append(f"{table_name}.{column_name}")
                
                conn.close()
            except Exception as e:
                print(f"Error reading SQLite database {db_path}: {e}")
    
    elif db_type in ["bigquery", "snowflake"]:
        # For BigQuery and Snowflake, read from JSON schema files
        db_path = base_dir / "resource" / "databases" / db_type / db_id
        print(f"DEBUG: Looking for schema files in: {db_path}")
        
        # Look for JSON files in subdirectories
        json_files = list(db_path.rglob("*.json"))
        print(f"DEBUG: Found {len(json_files)} JSON files")
        
        if json_files:
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        schema_data = json.load(f)
                        
                        table_name = schema_data.get('table_name', '')
                        column_names = schema_data.get('column_names', [])
                        
                        print(f"DEBUG: Processing file {json_file.name}, table: {table_name}, columns: {len(column_names)}")
                        
                        if table_name and column_names:
                            # Add each column directly without any pattern matching
                            for column_name in column_names:
                                schema.append(f"{table_name}.{column_name}")
                                
                except Exception as e:
                    print(f"Error reading JSON schema file {json_file}: {e}")
                    continue
    
    return sorted(schema)


def load_table_mappings_from_json(mapping_file: Path) -> Dict[str, str]:
    """
    Load table name mappings to wildcard patterns from a JSON file.
    
    Args:
        mapping_file: Path to the JSON mapping file
        
    Returns:
        Dictionary mapping table names to wildcard patterns
    """
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('mappings', {})
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


def extract_used_columns_with_mappings(sql: str, schema: List[str], table_mappings: Dict[str, str], model: str = "gpt-5") -> List[str]:
    """
    Extract used columns from SQL using table mappings to help AI understand table groupings.
    
    Args:
        sql: SQL query (unchanged)
        schema: Database schema
        table_mappings: Table name mappings to wildcard patterns (used only for schema transformation)
        model: OpenAI model to use
        
    Returns:
        List of used columns
    """
    # Transform schema to use wildcard patterns for better AI understanding
    mapped_schema = []
    for column in schema:
        table_name, col_name = column.split('.', 1)
        if table_name in table_mappings:
            # Replace table name with wildcard pattern in schema
            wildcard_pattern = table_mappings[table_name]
            mapped_schema.append(f"{wildcard_pattern}.{col_name}")
        else:
            mapped_schema.append(column)
    
    # Find relevant schema items based on table names mentioned in SQL
    sql_lower = sql.lower()
    relevant_schema_items = []
    
    # First, add all schema items that contain table names mentioned in SQL
    for item in mapped_schema:
        table_name = item.split('.')[0]
        if table_name.lower() in sql_lower or table_name.replace('_', '').lower() in sql_lower:
            relevant_schema_items.append(item)
    
    # If no relevant items found, fall back to first 100
    if not relevant_schema_items:
        relevant_schema_items = mapped_schema[:100]
    
    # Remove duplicates and limit to reasonable size
    relevant_schema_items = list(set(relevant_schema_items))[:200]
    
    # Use original SQL and relevant schema for extraction
    prompt = f"""You are a SQL expert. Given this SQL query and database schema, extract the table and column names that are actually used in the query.

SQL Query:
{sql}

Database Schema (relevant tables):
{chr(10).join(relevant_schema_items)}{'...' if len(mapped_schema) > len(relevant_schema_items) else ''}

CRITICAL RULES:
1. Return ONLY a JSON object with a "columns" array
2. Each column MUST be in format "table.column" 
3. Only include columns that are actually referenced in the SQL
4. If a table name contains wildcard (*), keep it as is - do NOT expand it
5. Handle table aliases correctly
6. Include columns from SELECT, WHERE, JOIN, GROUP BY, ORDER BY, HAVING clauses
7. When you see a wildcard pattern like "events_*" in the schema, treat it as a single table name
8. Map specific table names in the SQL to their wildcard patterns when they exist in the schema
9. IMPORTANT: After you extract columns, you normalize them against the schema above, it must be exist in the schema

Return format:
{{"columns": ["table1.column1", "table2.column2", ...]}}

Note: After extraction, all columns will be validated against the schema above. Only columns that exist in the schema will be kept in the final result so you must be careful with the columns you return, make sure you normalize them against the schema above."""

    print(f"DEBUG: SQL Query: {sql}")
    print(f"DEBUG: Schema length: {len(mapped_schema)}")
    print(f"DEBUG: First 5 schema items: {mapped_schema[:5]}")
    print(f"DEBUG: Table mappings: {table_mappings}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=12048,
        )
        
        content = response.choices[0].message.content.strip()
        print(f"DEBUG: Raw OpenAI response: {content}")
        
        # Handle markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        print(f"DEBUG: Processed content: {content}")
        
        if not content:
            print("Warning: Empty response from model")
            return []
        
        result = json.loads(content)
        columns = result.get("columns", [])
        print(f"DEBUG: Extracted columns: {columns}")
        return columns
        
    except Exception as e:
        print(f"Error extracting columns: {e}")
        print(f"DEBUG: Full error details: {type(e).__name__}: {str(e)}")
        return []


def needs_fix(doc: dict) -> bool:
    """
    Return True if the document should be repaired.

    • `used_columns` field is missing
    • `used_columns` is empty or None
    """
    if "used_columns" not in doc:
        return True

    # Empty list or None → needs repair
    if not doc["used_columns"]:
        return True

    return False


def count_empty_used_columns(coll) -> tuple[int, int, int]:
    """
    Count documents with different used_columns states.
    
    Returns:
        Tuple of (total_docs, docs_with_used_columns, docs_with_empty_used_columns)
    """
    total_docs = coll.count_documents({})
    docs_with_used_columns = coll.count_documents({"used_columns": {"$exists": True, "$ne": []}})
    docs_with_empty_used_columns = coll.count_documents({"$or": [
        {"used_columns": {"$exists": False}},
        {"used_columns": []},
        {"used_columns": None}
    ]})
    
    return total_docs, docs_with_used_columns, docs_with_empty_used_columns


# ───────────────────────────────────────────────────────────────
# 2.  WORKER
# ───────────────────────────────────────────────────────────────
def repair_document(task):
    idx, doc, base_dir, table_mapping_dir = task
    client = MongoClient(MONGO_URI)
    coll = client["mats"][args.collection_name]

    try:
        db_id = doc["db_id"]
        db_type = doc["db_type"]
        sql_query = doc["SQL"]

        schema = get_schema_for_db_type(db_id, db_type, base_dir)
        if not schema:
            print(f"[{idx}] WARNING: No schema found for {db_id} ({db_type})")
            client.close()
            return
            
        # Load table mappings from the table_mapping folder
        table_mappings = load_table_mappings_from_table_mapping_folder(db_id, table_mapping_dir)

        # Transform schema using table mappings for MongoDB storage
        transformed_schema = []
        for column in schema:
            table_name, col_name = column.split('.', 1)
            if table_name in table_mappings:
                # Replace table name with wildcard pattern
                wildcard_pattern = table_mappings[table_name]
                transformed_schema.append(f"{wildcard_pattern}.{col_name}")
            else:
                transformed_schema.append(column)
        
        # Deduplicate the transformed schema to remove duplicate wildcard columns
        transformed_schema = list(set(transformed_schema))

        # Extract used columns and normalize case
        used_cols = extract_used_columns_with_mappings(sql_query, schema, table_mappings, model="gpt-5")
        print(f"[{idx}] Extracted columns: {used_cols}")
        
        # Normalize case to match schema
        used_cols = normalize_columns_case(used_cols, transformed_schema)
        print(f"[{idx}] Used columns after filtering: {used_cols}")
        
    except Exception as exc:
        print(f"[{idx}] ERROR: {exc}")
        client.close()
        return

    coll.update_one(
        {"_id": idx},
        {
            "$set": {
                "schema": transformed_schema,
                "used_columns": used_cols,
                "updated_at": dt.datetime.utcnow(),
            },
            "$unset": {
                # clear any previous audit flags
                "audit_not_in_schema": "",
                "audit_case_mismatch": "",
            },
        },
    )
    print(f"[{idx}] fixed → {len(used_cols)} columns")
    client.close()


# ───────────────────────────────────────────────────────────────
# 3.  MAIN
# ───────────────────────────────────────────────────────────────
def main() -> None:
    root_dir = args.base_dir
    collection_name = args.collection_name

    client = MongoClient(MONGO_URI)
    coll = client["mats"][collection_name]

    # First, count the documents with different states
    total_docs, docs_with_used_columns, docs_with_empty_used_columns = count_empty_used_columns(coll)
    
    print(f"Dataset: Spider2.0-lite")
    print(f"Collection: {collection_name}")
    print(f"Total documents: {total_docs}")
    print(f"Documents with used_columns: {docs_with_used_columns}")
    print(f"Documents with empty/missing used_columns: {docs_with_empty_used_columns}")
    print(f"Percentage needing repair: {(docs_with_empty_used_columns/total_docs*100):.1f}%")
    print()

    to_fix = [
        (doc["_id"], doc, root_dir, args.table_mapping_dir)
        for doc in coll.find(
            {}, {"db_id": 1, "db_type": 1, "SQL": 1, "schema": 1, "used_columns": 1}
        )
        if needs_fix(doc)
    ]
    
    if args.limit:
        to_fix = to_fix[:args.limit]
        print(f"Limited to {len(to_fix)} samples for testing")
    
    client.close()

    print(f"{len(to_fix)} documents will be repaired")

    if not to_fix:
        print("No documents need repair!")
        return

    with Pool(processes=args.processes) as pool:
        pool.map(repair_document, to_fix, chunksize=4)


if __name__ == "__main__":
    main() 