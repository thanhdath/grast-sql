#!/usr/bin/env python3
"""
Bulk-import Spider2.0-lite dataset into MongoDB.

This script processes the Spider2.0-lite dataset and constructs samples with BIRD format:
- _id, question_id: Sample identifiers
- question: Natural language question
- SQL: Gold truth SQL query
- evidence: External knowledge content (loaded from files)
- db_id: Database identifier
- db_type: Database type (sqlite, bigquery, snowflake)
- temporal: Temporal indicator (present in some samples)

Examples
--------
python spider_2_1_add_dataset_to_mongo.py                    # loads spider2-lite/spider2-lite.jsonl
python spider_2_1_add_dataset_to_mongo.py --base-dir /path/to/spider2-lite
python spider_2_1_add_dataset_to_mongo.py --mongo-uri mongodb://localhost:27017
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from pymongo import MongoClient, UpdateOne


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import Spider2.0-lite dataset into MongoDB.")
    p.add_argument("--base-dir", type=Path, default=Path("/home/datht/Spider2/spider2-lite"),
                   help="Root directory containing spider2-lite.jsonl and resource/ (default: %(default)s)")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides)")
    p.add_argument("--collection-name", default="spider2_lite_samples",
                   help="MongoDB collection name (default: %(default)s)")
    return p.parse_args()


def map_db_name(db_name: str) -> str:
    """Map database names from JSONL to actual directory names."""
    # Known mappings for case/naming mismatches
    mappings = {
        # SQLite mappings
        "sqlite-sakila": "SQLITE_SAKILA",
        "Db-IMDB": "DB_IMDB",
    }
    
    return mappings.get(db_name, db_name)


def determine_db_type(db_name: str, base_dir: Path) -> str:
    """Determine database type by checking the resource/databases directory structure."""
    # Map the database name to the actual directory name
    mapped_db_name = map_db_name(db_name)
    
    # Check which directory contains the database
    for db_type in ["sqlite", "bigquery", "snowflake"]:
        db_path = base_dir / "resource" / "databases" / db_type / mapped_db_name
        if db_path.exists():
            return db_type
    
    # raise error if db_type is not found
    raise ValueError(f"Database type not found for {db_name} (mapped to {mapped_db_name})")

def load_gold_sql(instance_id: str, base_dir: Path) -> Optional[str]:
    """Load gold SQL for the given instance_id."""
    sql_file = base_dir / "evaluation_suite" / "gold" / "sql" / f"{instance_id}.sql"
    if sql_file.exists():
        return sql_file.read_text(encoding="utf-8").strip()
    return None


def load_external_knowledge(external_knowledge_file: Optional[str], base_dir: Path) -> str:
    """Load external knowledge content from file."""
    if not external_knowledge_file:
        return ""
    
    # Try to find the file in the documents directory
    doc_path = base_dir / "resource" / "documents" / external_knowledge_file
    if doc_path.exists():
        try:
            return doc_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"Warning: Could not read external knowledge file {external_knowledge_file}: {e}")
            return ""
    
    # If not found, return the filename as evidence
    return f"External knowledge reference: {external_knowledge_file}"


def construct_sample(sample_data: Dict[str, Any], base_dir: Path, sample_idx: int) -> Optional[Dict[str, Any]]:
    """Construct a complete sample with BIRD format structure."""
    instance_id = sample_data["instance_id"]
    db_id = sample_data["db"]
    question = sample_data["question"]
    external_knowledge_file = sample_data.get("external_knowledge")
    temporal = sample_data.get("temporal")  # Additional key from Spider2.0-lite
    
    # Map database name and determine database type
    mapped_db_id = map_db_name(db_id)
    db_type = determine_db_type(db_id, base_dir)
    
    # Load gold SQL
    gold_sql = load_gold_sql(instance_id, base_dir)
    
    # Skip samples without SQL
    if gold_sql is None:
        return None
    
    # Load external knowledge content
    evidence = load_external_knowledge(external_knowledge_file, base_dir)
    
    # Build sample with all available keys
    sample = {
        "_id": sample_idx,
        "question_id": sample_idx,  # Same as BIRD format
        "instance_id": instance_id,  # Original instance_id for reference
        "question": question,
        "SQL": gold_sql,  # BIRD format uses "SQL" not "query"
        "evidence": evidence,  # BIRD format uses "evidence" for external knowledge
        "db_id": mapped_db_id,  # This is the mapped db_id that can be found in database directory
        "db_type": db_type
    }
    
    # Add temporal key if present
    if temporal is not None:
        sample["temporal"] = temporal
    
    return sample


def main() -> None:
    args = parse_args()
    
    # Validate base directory
    jsonl_path = args.base_dir / "spider2-lite.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Cannot find {jsonl_path}")
    
    # Connect to MongoDB
    client = MongoClient(args.mongo_uri)
    coll = client["mats"][args.collection_name]
    
    print(f"Processing Spider2.0-lite dataset from {jsonl_path}")
    print(f"Target MongoDB collection: {args.collection_name}")
    
    # Read and process samples
    samples = []
    with jsonl_path.open(encoding="utf-8") as fp:
        for line_num, line in enumerate(fp, 1):
            try:
                sample_data = json.loads(line.strip())
                sample = construct_sample(sample_data, args.base_dir, line_num - 1)
                
                # Skip samples without SQL
                if sample is None:
                    continue
                    
                samples.append(sample)
                
                if line_num % 50 == 0:
                    print(f"Processed {line_num} samples...")
                    
            except Exception as e:
                print(f"Warning: Could not process line {line_num}: {e}")
                continue
    
    print(f"Successfully processed {len(samples)} samples")
    
    # Prepare MongoDB operations
    ops: list[UpdateOne] = [
        UpdateOne(
            {"_id": sample["_id"]},
            {"$setOnInsert": sample},
            upsert=True
        )
        for sample in samples
    ]
    
    # Execute bulk write
    if ops:
        res = coll.bulk_write(ops, ordered=False)
        print(f"[{coll.name}] inserted {res.upserted_count} new docs; "
              f"{len(samples) - res.upserted_count} already existed.")
    
    client.close()
    print("Import completed successfully!")


if __name__ == "__main__":
    main()
