#!/usr/bin/env python3
"""
spider_2_12_check_schema_column_info_mismatch.py
────────────────────────────────────────────────
Script to check for samples where used_columns exist in schema but not in column_info.

This helps identify potential mismatches between the schema and column_info fields.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from pymongo import MongoClient


def check_schema_column_info_mismatch():
    """Check for samples where used_columns exist in schema but not in column_info."""
    
    # Load environment variables
    load_dotenv("/home/datht/mats/.env")
    
    # Connect to MongoDB
    client = MongoClient("mongodb://192.168.1.108:27017")
    coll = client["mats"]["spider2_lite_samples"]
    
    print("=== Checking for schema vs column_info mismatches ===")
    print("Looking for samples where used_columns exist in schema but not in column_info")
    print()
    
    # Get all documents with used_columns
    docs_with_used_columns = list(coll.find({
        "used_columns": {"$exists": True, "$ne": []}
    }))
    
    print(f"Found {len(docs_with_used_columns)} documents with used_columns")
    print()
    
    mismatch_count = 0
    total_columns_checked = 0
    
    for doc in docs_with_used_columns:
        doc_id = doc["_id"]
        db_id = doc["db_id"]
        db_type = doc["db_type"]
        used_columns = doc.get("used_columns", [])
        schema = doc.get("schema", [])
        column_info = doc.get("column_info", {})
        
        if not schema or not column_info:
            continue
        
        # Check each used_column
        mismatches = []
        for col in used_columns:
            total_columns_checked += 1
            
            # Check if column exists in schema
            if col in schema:
                # Check if column exists in column_info
                if col not in column_info:
                    mismatches.append(col)
        
        if mismatches:
            mismatch_count += 1
            print(f"--- Document {doc_id} ({db_id}, {db_type}) ---")
            print(f"  Used columns with mismatches ({len(mismatches)}):")
            for col in mismatches:
                print(f"    ❌ {col} (in schema but not in column_info)")
            print()
    
    # Print summary
    print("=== SUMMARY ===")
    print(f"Total documents processed: {len(docs_with_used_columns)}")
    print(f"Documents with mismatches: {mismatch_count}")
    print(f"Total columns checked: {total_columns_checked}")
    
    if mismatch_count > 0:
        print(f"Percentage of documents with mismatches: {(mismatch_count/len(docs_with_used_columns)*100):.1f}%")
    else:
        print("✅ No mismatches found! All used_columns that exist in schema also exist in column_info.")
    
    client.close()


if __name__ == "__main__":
    check_schema_column_info_mismatch() 