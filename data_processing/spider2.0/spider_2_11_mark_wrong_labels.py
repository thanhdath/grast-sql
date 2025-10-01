#!/usr/bin/env python3
"""
spider_2_11_mark_wrong_labels.py
─────────────────────────────────
Script to mark samples with used_columns that don't exist in schema as wrong_label = True.

This script identifies samples where used_columns contain items not present in the schema field
and marks them with wrong_label = True for further review.

Usage
-----
python spider_2_11_mark_wrong_labels.py                    # mark all samples with wrong labels
python spider_2_11_mark_wrong_labels.py --limit 10         # test with 10 samples
python spider_2_11_mark_wrong_labels.py --target-ids 5 171 172  # process specific document IDs
python spider_2_11_mark_wrong_labels.py --dry-run          # show what would be marked without updating
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from dotenv import load_dotenv
from pymongo import MongoClient


def analyze_missing_in_schema(used_columns: list, schema: list) -> dict:
    """
    Analyze which used_columns are not found in the schema.
    
    Args:
        used_columns: List of columns from used_columns field
        schema: List of columns from schema field
        
    Returns:
        Dictionary with missing columns analysis
    """
    schema_set = set(schema)
    missing_columns = []
    
    for col in used_columns:
        if col not in schema_set:
            missing_columns.append(col)
    
    return {
        'total_used_columns': len(used_columns),
        'columns_in_schema': len([col for col in used_columns if col in schema_set]),
        'missing_columns': missing_columns,
        'missing_count': len(missing_columns)
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mark samples with used_columns not in schema as wrong_label = True.")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides).")
    p.add_argument("--collection-name", default="spider2_lite_samples",
                   help="MongoDB collection name (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of documents to process (for testing)")
    p.add_argument("--target-ids", nargs='+', type=int,
                   help="Specific document IDs to process (e.g., --target-ids 5 171 172)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be marked without updating MongoDB")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv("/home/datht/mats/.env")
    
    if args.dry_run:
        print("=== DRY RUN MODE - No changes will be made to MongoDB ===")
    else:
        print("=== Marking samples with wrong labels ===")
    
    client = MongoClient(args.mongo_uri)
    coll = client["mats"][args.collection_name]
    
    # Get all documents with used_columns
    docs_with_used_columns = list(coll.find({
        "used_columns": {"$exists": True, "$ne": []}
    }))
    
    # Filter by target IDs if specified
    if args.target_ids:
        target_ids_set = set(args.target_ids)
        docs_with_used_columns = [doc for doc in docs_with_used_columns if doc["_id"] in target_ids_set]
        print(f"Filtering to {len(docs_with_used_columns)} target documents: {sorted(args.target_ids)}")
    
    if args.limit:
        docs_with_used_columns = docs_with_used_columns[:args.limit]
        print(f"Limited to {len(docs_with_used_columns)} samples for testing")
    
    print(f"Found {len(docs_with_used_columns)} documents with used_columns")
    
    # Process each document
    wrong_label_count = 0
    already_marked_count = 0
    
    for doc in docs_with_used_columns:
        doc_id = doc["_id"]
        db_id = doc["db_id"]
        db_type = doc["db_type"]
        used_columns = doc.get("used_columns", [])
        schema = doc.get("schema", [])
        current_wrong_label = doc.get("wrong_label", False)
        
        if not schema:
            print(f"⚠️  Document {doc_id} ({db_id}, {db_type}) has no schema, skipping")
            continue
        
        # Analyze missing columns
        missing_analysis = analyze_missing_in_schema(used_columns, schema)
        
        if missing_analysis['missing_count'] > 0:
            # This document has wrong labels
            if current_wrong_label:
                print(f"📋 Document {doc_id} ({db_id}, {db_type}) - Already marked as wrong_label")
                already_marked_count += 1
            else:
                print(f"❌ Document {doc_id} ({db_id}, {db_type}) - Missing {missing_analysis['missing_count']} columns:")
                for missing_col in missing_analysis['missing_columns']:
                    print(f"      {missing_col}")
                
                if not args.dry_run:
                    # Mark as wrong_label = True
                    coll.update_one(
                        {"_id": doc_id},
                        {"$set": {"wrong_label": True}}
                    )
                    print(f"    ✅ Marked as wrong_label = True")
                else:
                    print(f"    🔍 Would mark as wrong_label = True (dry run)")
                
                wrong_label_count += 1
        else:
            # All columns exist in schema, remove wrong_label if it was set
            if current_wrong_label:
                print(f"✅ Document {doc_id} ({db_id}, {db_type}) - All columns valid, removing wrong_label")
                if not args.dry_run:
                    coll.update_one(
                        {"_id": doc_id},
                        {"$unset": {"wrong_label": ""}}
                    )
                    print(f"    ✅ Removed wrong_label flag")
                else:
                    print(f"    🔍 Would remove wrong_label flag (dry run)")
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Total documents processed: {len(docs_with_used_columns)}")
    print(f"Documents marked as wrong_label: {wrong_label_count}")
    print(f"Documents already marked: {already_marked_count}")
    
    if args.dry_run:
        print(f"DRY RUN: No changes made to MongoDB")
    else:
        print(f"Updated MongoDB with wrong_label flags")
    
    client.close()


if __name__ == "__main__":
    main() 