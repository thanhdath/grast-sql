#!/usr/bin/env python3
"""
Verify that extracted column usage actually appears in the ground truth SQL queries.

This script checks if the lowercase column names from `used_columns` exist in the 
lowercase SQL query. It identifies cases where the LLM may have hallucinated 
columns that don't actually appear in the SQL.

Usage
-----
python 1_5_7_check_error_in_extracting_column_usage.py train
python 1_5_7_check_error_in_extracting_column_usage.py dev --dataset spider
python 1_5_7_check_error_in_extracting_column_usage.py train --mongo-uri mongodb://user:pass@host:27017
"""
from __future__ import annotations
import argparse
import json
import os
from typing import List, Set, Tuple, Dict
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

# Import sql_metadata for proper SQL tokenization
from sql_metadata import Parser


# ───────────────────────────────────────────────────────────────
# 0.  CLI & ENV
# ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify column usage against SQL queries.")
    p.add_argument("split", choices=["train", "dev"],
                   help="Which split collection to verify (train | dev)")
    p.add_argument("--dataset", choices=["bird", "spider"], default="bird",
                   help="Which dataset to process (bird | spider, default: %(default)s)")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides)")
    p.add_argument("--verbose", action="store_true",
                   help="Show detailed output for each issue")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of documents to process (for testing)")
    p.add_argument("--debug-tokens", action="store_true",
                   help="Print SQL tokens for first few documents")
    p.add_argument("--output-dir", type=str, default="logs",
                   help="Directory to save error logs (default: logs)")
    p.add_argument("--update-mongodb", action="store_true",
                   help="Update MongoDB with column_usage_checking field for invalid samples")
    return p.parse_args()


def get_sql_field(dataset: str) -> str:
    """Get the correct SQL field name for the dataset."""
    if dataset == "bird":
        return "SQL"
    elif dataset == "spider":
        return "query"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


args = parse_args()
load_dotenv("../.env")
MONGO_URI = args.mongo_uri


# ───────────────────────────────────────────────────────────────
# 1.  SQL tokenization using sql_metadata (same as prepare_sft_datasets.py)
# ───────────────────────────────────────────────────────────────
def tokenize_sql(sql: str) -> List[str]:
    """
    Tokenize SQL using sql_metadata.Parser (same approach as prepare_sft_datasets.py).
    Returns a list of lowercase tokens.
    """
    try:
        # Use sql_metadata.Parser like in prepare_sft_datasets.py
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
    Uses the same approach as prepare_sft_datasets.py.
    
    Returns:
        Tuple of (valid_columns, invalid_columns)
    """
    if not used_columns:
        return [], []
    
    sql_tokens = tokenize_sql(sql)
    aliases = extract_table_aliases(sql)
    
    valid_columns = []
    invalid_columns = []
    
    for col in used_columns:
        col_lower = col.lower()
        
        # Check if the column appears in SQL tokens (same logic as prepare_sft_datasets.py)
        if col_lower in sql_tokens:
            valid_columns.append(col)
        else:
            # Check for table.column format
            if '.' in col_lower:
                table, column = col_lower.split('.', 1)
                
                # Check if just the column part appears
                if column in sql_tokens:
                    valid_columns.append(col)
                else:
                    # Check if table is aliased and the alias.column appears
                    found = False
                    for alias, full_table in aliases.items():
                        if full_table == table:
                            alias_col = f"{alias}.{column}"
                            if alias_col in sql_tokens:
                                valid_columns.append(col)
                                found = True
                                break
                    
                    if not found:
                        invalid_columns.append(col)
            else:
                invalid_columns.append(col)
    
    return valid_columns, invalid_columns


# ───────────────────────────────────────────────────────────────
# 2.  Main verification logic
# ───────────────────────────────────────────────────────────────
def main() -> None:
    client = MongoClient(MONGO_URI)
    # Use correct collection names based on actual database structure
    if args.dataset == "bird":
        if args.split == "train":
            collection_name = "train_samples"
        else:  # dev
            collection_name = "dev_samples"
    else:  # spider
        if args.split == "train":
            collection_name = "spider_train_samples"
        else:  # dev
            collection_name = "spider_dev_samples"
    
    coll = client["mats"][collection_name]
    
    sql_field = get_sql_field(args.dataset)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Statistics
    total_docs = 0
    docs_with_issues = 0
    total_invalid_columns = 0
    error_samples = []
    debug_count = 0
    bulk_ops = []
    
    # Build query with limit if specified
    query = {}
    if args.limit:
        print(f"Processing limited to {args.limit} documents...")
    
    # Process documents
    cursor = coll.find(query, {"used_columns": 1, sql_field: 1, "_id": 1})
    if args.limit:
        cursor = cursor.limit(args.limit)
    
    for doc in tqdm(cursor, desc="Verifying columns"):
        total_docs += 1
        
        used_columns = doc.get("used_columns", [])
        sql_query = doc.get(sql_field, "")
        
        if not used_columns or not sql_query:
            continue
        
        # Debug: Print tokens for first few documents
        if args.debug_tokens and debug_count < 5:
            tokens = tokenize_sql(sql_query)
            print(f"\n[_id={doc['_id']}] SQL: {sql_query[:100]}...")
            print(f"Tokens: {tokens}")
            print(f"Used columns: {used_columns}")
            debug_count += 1
        
        valid_columns, invalid_columns = verify_columns_in_sql(used_columns, sql_query)
        
        if invalid_columns:
            docs_with_issues += 1
            total_invalid_columns += len(invalid_columns)
            
            # Log error sample to JSON
            error_sample = {
                "_id": doc["_id"],
                "sql": sql_query,
                "used_columns": used_columns,
                "valid_columns": valid_columns,
                "invalid_columns": invalid_columns,
                "sql_tokens": tokenize_sql(sql_query)
            }
            error_samples.append(error_sample)
            
            # Update MongoDB with column_usage_checking field
            if args.update_mongodb:
                bulk_ops.append(
                    UpdateOne(
                        {"_id": doc["_id"]},
                        {
                            "$set": {
                                "column_usage_checking": {
                                    "has_invalid_columns": True,
                                    "invalid_columns": invalid_columns,
                                    "valid_columns": valid_columns,
                                    "total_used_columns": len(used_columns),
                                    "invalid_count": len(invalid_columns),
                                    "verification_date": datetime.datetime.utcnow()
                                }
                            }
                        }
                    )
                )
            
            if args.verbose:
                print(f"\n[_id={doc['_id']}] Found {len(invalid_columns)} invalid columns:")
                print(f"  SQL: {sql_query[:200]}...")
                print(f"  Invalid columns: {invalid_columns}")
                print(f"  Valid columns: {valid_columns}")
                print(f"  Original used_columns: {used_columns}")
                print(f"  SQL tokens: {tokenize_sql(sql_query)}")
        else:
            # Mark documents with valid columns
            if args.update_mongodb:
                bulk_ops.append(
                    UpdateOne(
                        {"_id": doc["_id"]},
                        {
                            "$set": {
                                "column_usage_checking": {
                                    "has_invalid_columns": False,
                                    "invalid_columns": [],
                                    "valid_columns": valid_columns,
                                    "total_used_columns": len(used_columns),
                                    "invalid_count": 0,
                                    "verification_date": datetime.datetime.utcnow()
                                }
                            }
                        }
                    )
                )
    
    # Execute bulk updates if requested
    if args.update_mongodb and bulk_ops:
        print(f"\nUpdating MongoDB with column_usage_checking field...")
        result = coll.bulk_write(bulk_ops, ordered=False)
        print(f"Updated {result.modified_count} documents")
    
    # Save error samples to JSON file
    timestamp = args.split + "_" + args.dataset + "_" + str(total_docs) + "_docs"
    error_log_file = os.path.join(args.output_dir, f"column_verification_errors_{timestamp}.json")
    
    with open(error_log_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "dataset": args.dataset,
                "split": args.split,
                "collection": collection_name,
                "total_documents_processed": total_docs,
                "documents_with_issues": docs_with_issues,
                "total_invalid_columns": total_invalid_columns,
                "issue_rate_percent": (docs_with_issues/total_docs*100) if total_docs > 0 else 0,
                "mongodb_updated": args.update_mongodb
            },
            "error_samples": error_samples
        }, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n=== Verification Summary ===")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Collection: {collection_name}")
    print(f"Total documents processed: {total_docs}")
    print(f"Documents with issues: {docs_with_issues}")
    print(f"Total invalid columns found: {total_invalid_columns}")
    if total_docs > 0:
        print(f"Issue rate: {docs_with_issues/total_docs*100:.2f}%")
    
    print(f"\nError samples saved to: {error_log_file}")
    print(f"Total error samples logged: {len(error_samples)}")
    
    if args.update_mongodb:
        print(f"MongoDB updated with column_usage_checking field")
        print(f"To filter out invalid samples, use: {{'column_usage_checking.has_invalid_columns': false}}")
    
    client.close()


if __name__ == "__main__":
    import datetime
    main()
