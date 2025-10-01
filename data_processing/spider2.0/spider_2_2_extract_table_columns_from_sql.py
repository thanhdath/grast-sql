#!/usr/bin/env python3
"""
Annotate each SQL query with the list of columns it touches for Spider2.0-lite dataset,
then verify that the extracted columns actually appear in the SQL query.
First compute total tokens and estimated price for GPT-4.1-mini calls,
then ask user to confirm before proceeding.
"""
from __future__ import annotations

import argparse, datetime as dt, json, os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import openai
from dotenv import load_dotenv
from pymongo import MongoClient, errors, UpdateOne

# Import sql_metadata for proper SQL tokenization
from sql_metadata import Parser


# ───────────────────────────── CLI & ENV ──────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate Spider2.0-lite SQL with used columns.")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI",
                                     "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides).")
    p.add_argument("--collection-name", default="spider2_lite_samples",
                   help="MongoDB collection name (default: %(default)s)")
    p.add_argument("--processes", type=int, default=16,
                   help="Worker processes (default: %(default)s).")
    p.add_argument("--price-per-1k-tokens", type=float, default=0.0044,
                   help="o4-mini price per 1000 tokens (USD).")
    p.add_argument("--base-dir", type=Path, default=Path("/home/datht/Spider2/spider2-lite"),
                   help="Root directory containing Spider2.0-lite dataset (default: %(default)s)")
    p.add_argument("--table-mapping-dir", type=Path, default=Path("/home/datht/Spider2/data_processing/table_mapping"),
                   help="Directory containing table mapping JSON files (default: %(default)s)")
    p.add_argument("--dry-run", action="store_true",
                   help="Dry run mode: only write schema statistics, skip OpenAI API calls")
    p.add_argument("--recompute-only", action="store_true",
                   help="Recompute mode: only re-verify existing used_columns, skip OpenAI API calls")
    p.add_argument("--skip-processed", action="store_true", default=True,
                   help="Skip samples that already have used_columns (default: True)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of documents to process (for testing)")
    return p.parse_args()


args = parse_args()
load_dotenv("/home/datht/mats/.env")
MONGO_URI = args.mongo_uri


# ───────────────────────────── HELPERS ────────────────────────────────
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
            import sqlite3
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
        
        # Look for JSON files in subdirectories
        json_files = list(db_path.rglob("*.json"))
        
        if json_files:
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        schema_data = json.load(f)
                        
                        table_name = schema_data.get('table_name', '')
                        column_names = schema_data.get('column_names', [])
                        
                        if table_name and column_names:
                            # Add each column directly without any pattern matching
                            for column_name in column_names:
                                schema.append(f"{table_name}.{column_name}")
                                
                except Exception as e:
                    print(f"Error reading JSON schema file {json_file}: {e}")
                    continue
    
    return sorted(schema)


def count_tokens(text: str) -> int:
    # Basic approximation: 1 token ~ 4 characters (this is heuristic)
    # You can replace this with tiktoken or official tokenizer for accuracy
    return max(1, len(text) // 4)


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





def extract_used_columns_with_mappings(sql: str, schema: List[str], table_mappings: Dict[str, str], model: str = "o4-mini") -> List[str]:
    """
    Extract used columns from SQL using table mappings to help AI understand table groupings.
    Uses chunking to handle large schemas that exceed token limits.
    
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
    
    # Chunk the schema into smaller pieces (max 3000 columns per chunk)
    chunk_size = 3000
    schema_chunks = [mapped_schema[i:i + chunk_size] for i in range(0, len(mapped_schema), chunk_size)]
    
    all_extracted_columns = []
    
    for i, chunk in enumerate(schema_chunks):
        print(f"Processing schema chunk {i+1}/{len(schema_chunks)} ({len(chunk)} columns)")
        
        # Create prompt for this chunk
        prompt = f"""You are a SQL expert. Given this SQL query and a portion of the database schema, extract the table and column names that are actually used in the query.

SQL Query:
{sql}

Database Schema (Chunk {i+1}/{len(schema_chunks)} - {len(chunk)} columns):
{chr(10).join(chunk)}

CRITICAL RULES:
1. Return ONLY a JSON object with a "columns" array
2. Each column MUST be in format "table.column" 
3. Only include columns that are actually referenced in the SQL
4. If a table name contains wildcard (*), keep it as is - do NOT expand it
5. Handle table aliases correctly
6. Include columns from SELECT, WHERE, JOIN, GROUP BY, ORDER BY, HAVING clauses
7. When you see a wildcard pattern like "events_*" in the schema, treat it as a single table name
8. Map specific table names in the SQL to their wildcard patterns when they exist in the schema
9. IMPORTANT: Only return columns that exist in the schema chunk above

Return format:
{{"columns": ["table1.column1", "table2.column2", ...]}}

Note: This is chunk {i+1} of {len(schema_chunks)}. Only return columns that are actually used in the SQL query and exist in this schema chunk."""

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=12048,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            if not content:
                print(f"Warning: Empty response from model for chunk {i+1}")
                continue
            
            result = json.loads(content)
            chunk_columns = result.get("columns", [])
            all_extracted_columns.extend(chunk_columns)
            print(f"Chunk {i+1} extracted {len(chunk_columns)} columns")
            
        except Exception as e:
            print(f"Error extracting columns from chunk {i+1}: {e}")
            continue
    
    # Remove duplicates from all chunks
    unique_columns = list(set(all_extracted_columns))
    print(f"Total extracted columns after combining chunks: {len(unique_columns)}")
    
    return unique_columns


def normalize_columns_case(used_cols: List[str], schema: List[str]) -> List[str]:
    """
    Normalize the case of used_columns to match the schema case.
    Do NOT exclude any columns - only normalize case.
    
    Args:
        used_cols: List of columns extracted by AI
        schema: List of actual schema columns
        
    Returns:
        List of columns with normalized case to match schema
    """
    normalized_cols = []
    schema_lower_to_original = {col.lower(): col for col in schema}
    
    for col in used_cols:
        col_lower = col.lower()
        
        # Check for exact match first (case-insensitive) and use schema case
        if col_lower in schema_lower_to_original:
            normalized_cols.append(schema_lower_to_original[col_lower])
        else:
            # Keep original if not found in schema
            normalized_cols.append(col)
    
    return normalized_cols


# ───────────────────────────── VERIFICATION ────────────────────────────────
def tokenize_sql(sql: str) -> List[str]:
    """
    Tokenize SQL using sql_metadata.Parser.
    Returns a list of lowercase tokens.
    """
    try:
        # Use sql_metadata.Parser
        sql_tokens = [token.value for token in Parser(sql.lower()).tokens]
        return sql_tokens
    except Exception as e:
        # Fallback to simple tokenization if parsing fails
        sql_lower = sql.lower()
        tokens = []
        for comma_part in sql_lower.split(','):
            for token in comma_part.split():
                if '.' in token:
                    parts = token.split('.')
                    tokens.extend(parts)
                else:
                    tokens.append(token)
        return tokens


def extract_table_aliases(sql: str) -> Dict[str, str]:
    """
    Extract table aliases from SQL query.
    Returns a mapping from alias to full table name.
    """
    try:
        parser = Parser(sql.lower())
        tables_aliases = parser.tables_aliases
        return tables_aliases
    except Exception as e:
        return {}


def verify_columns_in_sql(used_columns: List[str], sql: str) -> Tuple[List[str], List[str]]:
    """
    Verify which columns from used_columns actually appear in the SQL query.
    Simple verification: just check if column name appears anywhere in SQL.
    
    Returns:
        Tuple of (valid_columns, invalid_columns)
    """
    if not used_columns:
        return [], []
    
    sql_lower = sql.lower()
    valid_columns = []
    invalid_columns = []
    
    for col in used_columns:
        col_lower = col.lower()
        
        # Extract just the column name (last part after the last dot)
        if '.' in col_lower:
            # For table.column format, get just the column part
            column_name = col_lower.split('.')[-1]
        else:
            column_name = col_lower
        
        # Simple check: if column name appears anywhere in SQL, consider it valid
        if column_name in sql_lower:
            valid_columns.append(col)
        else:
            invalid_columns.append(col)
    
    return valid_columns, invalid_columns


# ───────────────────────────── WORKER ─────────────────────────────────
def process_sample(task):
    idx, sample, base_dir, table_mapping_dir, dry_run, recompute_only = task
    client = MongoClient(MONGO_URI)
    coll = client["mats"][args.collection_name]

    model = 'o4-mini'

    db_id = sample["db_id"]
    db_type = sample["db_type"]
    sql_query = sample["SQL"]

    try:
        # Check if sample already has used_columns (for recompute mode)
        if recompute_only and "used_columns" not in sample:
            print(f"[{idx}] SKIP: No used_columns found, skipping recomputation")
            client.close()
            return
            
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
        
        # Calculate schema statistics for the FINAL schema (after transformation and deduplication)
        unique_tables_final = set()
        for col in transformed_schema:
            if '.' in col:
                table_name = col.split('.')[0]
                unique_tables_final.add(table_name)
        
        # Write schema statistics to file (using the final schema length)
        with open("schema_statistics.txt", 'a') as f:
            f.write(f"{idx}\t{db_id}\t{db_type}\t{len(transformed_schema)}\t{len(unique_tables_final)}\n")
        
        print(f"[{idx}] Schema stats: {len(unique_tables_final)} tables, {len(transformed_schema)} columns for {db_id} (after deduplication)")
        if table_mappings:
            print(f"[{idx}] Table mappings loaded: {len(table_mappings)} mappings")
        
        # Initialize variables for status message
        used_cols = []
        invalid_columns = []
        
        if recompute_only:
            # Recompute mode: use existing used_columns and just re-verify
            used_cols = sample.get("used_columns", [])
            print(f"[{idx}] RECOMPUTE: Re-verifying {len(used_cols)} existing columns")
            
            # Verify that extracted columns actually appear in SQL
            valid_columns, invalid_columns = verify_columns_in_sql(used_cols, sql_query)
            
            # Update with new verification results
            update_data = {
                "column_usage_checking": {
                    "has_invalid_columns": len(invalid_columns) > 0,
                    "invalid_columns": invalid_columns,
                    "valid_columns": valid_columns,
                    "total_used_columns": len(used_cols),
                    "invalid_count": len(invalid_columns),
                    "verification_date": dt.datetime.utcnow()
                }
            }
        elif dry_run:
            print(f"[{idx}] DRY RUN: Skipping OpenAI API call, updating schema only")
            # For dry run, only update schema without column extraction
            update_data = {
                "schema": transformed_schema,  # Save transformed schema with wildcard patterns
                "sql_parsed_at": dt.datetime.utcnow(),
            }
        else:
            # Extract used columns and verify them
            used_cols = extract_used_columns_with_mappings(sql_query, schema, table_mappings, model)
            
            # Normalize case to match schema (AI should have already done validation)
            # This only adjusts case, doesn't exclude any columns
            used_cols = normalize_columns_case(used_cols, schema)
            
            # Verify that extracted columns actually appear in SQL
            valid_columns, invalid_columns = verify_columns_in_sql(used_cols, sql_query)
            
            # Update with verification results
            update_data = {
                "used_columns": used_cols,
                "schema": transformed_schema,  # Save transformed schema with wildcard patterns
                "sql_parsed_at": dt.datetime.utcnow(),
                "column_usage_checking": {
                    "has_invalid_columns": len(invalid_columns) > 0,
                    "invalid_columns": invalid_columns,
                    "valid_columns": valid_columns,
                    "total_used_columns": len(used_cols),
                    "invalid_count": len(invalid_columns),
                    "verification_date": dt.datetime.utcnow()
                }
            }
        
    except Exception as exc:
        print(f"[{idx}] ERROR: {exc}")
        client.close()
        return

    try:
        coll.update_one(
            {"_id": idx},
            {"$set": update_data},
            upsert=False,
        )
        
        # Print status with verification info
        if recompute_only:
            status_msg = f"[{idx}] recomputed verification ({len(used_cols)} cols)"
            if invalid_columns:
                status_msg += f" - WARNING: {len(invalid_columns)} invalid columns: {invalid_columns[:3]}{'...' if len(invalid_columns) > 3 else ''}"
        elif dry_run:
            status_msg = f"[{idx}] updated (schema only, dry run)"
        else:
            status_msg = f"[{idx}] updated ({len(used_cols)} cols, model={model})"
            if invalid_columns:
                status_msg += f" - WARNING: {len(invalid_columns)} invalid columns: {invalid_columns[:3]}{'...' if len(invalid_columns) > 3 else ''}"
        print(status_msg)
        
    except errors.PyMongoError as exc:
        print(f"[{idx}] Mongo ERROR: {exc}")
    finally:
        client.close()


# ───────────────────────────── MAIN ──────────────────────────────────
def main() -> None:
    root_dir = args.base_dir
    collection_name = args.collection_name

    # Connect to MongoDB and get all samples
    client = MongoClient(MONGO_URI)
    coll = client["mats"][collection_name]
    
    # Get samples based on mode
    if args.recompute_only:
        # For recompute mode, get only samples that already have used_columns
        samples = list(coll.find({"SQL": {"$ne": None}, "used_columns": {"$exists": True}}))
        print(f"RECOMPUTE MODE: Found {len(samples)} samples with existing used_columns")
    else:
        # Get all samples that have SQL (not None)
        all_samples = list(coll.find({"SQL": {"$ne": None}}))
        
        if args.skip_processed:
            # Skip samples that already have used_columns
            samples = [s for s in all_samples if not s.get("used_columns") or len(s.get("used_columns", [])) == 0]
            skipped_count = len(all_samples) - len(samples)
            print(f"NORMAL MODE: Found {len(all_samples)} total samples with SQL")
            print(f"Skipping {skipped_count} already processed samples")
            print(f"Processing {len(samples)} samples that need column extraction")
        else:
            # Process all samples (including already processed ones)
            samples = all_samples
            print(f"NORMAL MODE: Processing all {len(samples)} samples with SQL (including already processed)")
    
    if args.limit:
        samples = samples[:args.limit]
        print(f"Limited to {len(samples)} samples for testing")
    client.close()
    
    # Create output file for schema statistics
    output_file = Path("schema_statistics.txt")
    with open(output_file, 'w') as f:
        f.write("Sample_ID\tDatabase_ID\tDatabase_Type\tSchema_Length\tUnique_Tables\n")
    
    if not samples:
        print(f"No samples found in collection {collection_name}")
        return

    # Calculate cost only for non-recompute modes
    if not args.recompute_only:
        total_tokens = 0
        for sample in samples:
            # count tokens in SQL text as proxy for cost
            sql_text = sample.get("SQL", "")
            total_tokens += count_tokens(sql_text)

        # Calculate estimated price for GPT calls
        estimated_cost = (total_tokens / 1000) * args.price_per_1k_tokens

    print(f"Dataset: Spider2.0-lite")
    print(f"Collection: {collection_name}")
    print(f"Total samples: {len(samples)}")
    
    if args.recompute_only:
        print("RECOMPUTE MODE: Re-verifying existing used_columns, skipping OpenAI API calls")
    elif args.dry_run:
        print("DRY RUN MODE: Skipping OpenAI API calls, updating schema in MongoDB")
    else:
        print(f"Estimated total tokens: {total_tokens}")
        print(f"Estimated cost for o4-mini calls: ${estimated_cost:.4f} USD")

    # Auto-proceed without asking for confirmation
    print("Auto-proceeding with processing...")

    tasks = [(sample["_id"], sample, root_dir, args.table_mapping_dir, args.dry_run, args.recompute_only) for sample in samples]
    print(f"Processing {len(tasks)} samples...")

    with Pool(processes=args.processes) as pool:
        pool.map(process_sample, tasks, chunksize=4)
    
    # Print verification summary
    print(f"\n=== Processing Complete ===")
    
    client = MongoClient(MONGO_URI)
    coll = client["mats"][collection_name]
    
    if args.dry_run:
        # For dry run, check schema updates
        total_processed = coll.count_documents({"schema": {"$exists": True}})
        print(f"DRY RUN SUMMARY:")
        print(f"Total documents with schema updated: {total_processed}")
    else:
        # For normal run, check verification results
        print(f"Checking verification results...")
        total_processed = coll.count_documents({"used_columns": {"$exists": True}})
        docs_with_invalid = coll.count_documents({"column_usage_checking.has_invalid_columns": True})
        
        if total_processed > 0:
            print(f"Total documents processed: {total_processed}")
            print(f"Documents with invalid columns: {docs_with_invalid}")
            print(f"Accuracy rate: {((total_processed - docs_with_invalid) / total_processed * 100):.2f}%")
            
            if docs_with_invalid > 0:
                print(f"\nSample invalid columns found:")
                invalid_samples = coll.find({"column_usage_checking.has_invalid_columns": True}, 
                                          {"_id": 1, "column_usage_checking.invalid_columns": 1}).limit(5)
                for sample in invalid_samples:
                    print(f"  [_id={sample['_id']}] Invalid: {sample['column_usage_checking']['invalid_columns'][:3]}{'...' if len(sample['column_usage_checking']['invalid_columns']) > 3 else ''}")
    
    client.close()
    
    # Print summary of schema statistics
    print(f"\n=== Schema Statistics Summary ===")
    print(f"Schema statistics written to: schema_statistics.txt")
    
    # Read and display summary statistics
    try:
        with open("schema_statistics.txt", 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:  # Skip header
                schema_lengths = []
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        schema_lengths.append(int(parts[3]))
                
                if schema_lengths:
                    print(f"Total samples processed: {len(schema_lengths)}")
                    print(f"Average schema length: {sum(schema_lengths) / len(schema_lengths):.1f}")
                    print(f"Min schema length: {min(schema_lengths)}")
                    print(f"Max schema length: {max(schema_lengths)}")
    except Exception as e:
        print(f"Error reading schema statistics: {e}")


if __name__ == "__main__":
    main() 