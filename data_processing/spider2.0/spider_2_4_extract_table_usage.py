#!/usr/bin/env python3
"""
spider_2_4_extract_table_usage.py
─────────────────────────────────
Audit table usage in spider2_lite_samples

For every document, this script sets/updates:
    tables_used               : ["film", "actor", …]
    audit_table_case_mismatch : ["country", …]  # only case differs
    audit_table_missing       : ["foo", "bar"]  # no match at all

Usage
-----
python spider_2_4_extract_table_usage.py                    # audits spider2_lite_samples
python spider_2_4_extract_table_usage.py --limit 10         # test with 10 samples
python spider_2_4_extract_table_usage.py --mongo-uri mongodb://user:pass@host:27017
"""
from __future__ import annotations
import argparse
import os
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm


# ───────────────────────────────────────────────────────────────
# 0.  CLI & ENV
# ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit table usage in Spider2.0-lite SQL metadata.")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides)")
    p.add_argument("--collection-name", default="spider2_lite_samples",
                   help="MongoDB collection name (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of documents to process (for testing)")
    return p.parse_args()


args = parse_args()
load_dotenv("/home/datht/mats/.env")                       # makes sure .env is honoured
MONGO_URI = args.mongo_uri


# ───────────────────────────────────────────────────────────────
# 1.  Mongo connection
# ───────────────────────────────────────────────────────────────
client = MongoClient(MONGO_URI)
coll = client["mats"][args.collection_name]


# ───────────────────────────────────────────────────────────────
# 2.  Prepare bulk updates
# ───────────────────────────────────────────────────────────────
bulk_ops           : list[UpdateOne] = []
docs_with_missing  = 0
docs_with_case_mis = 0

# Get documents to process
docs_to_process = list(coll.find({}, {"used_columns": 1, "schema": 1}))
if args.limit:
    docs_to_process = docs_to_process[:args.limit]
    print(f"Limited to {len(docs_to_process)} samples for testing")

for doc in tqdm(docs_to_process, desc="Processing documents"):
    used_cols   = doc.get("used_columns") or []
    schema_cols = doc.get("schema")       or []

    # ---- derive table sets --------------------------------------------------
    tables_used       = sorted({c.split(".", 1)[0] for c in used_cols})
    tables_in_schema  = {c.split(".", 1)[0] for c in schema_cols}
    tables_schema_lc  = {t.lower() for t in tables_in_schema}

    case_mismatch, truly_missing = [], []

    for t in tables_used:
        if t in tables_in_schema:
            continue
        if t.lower() in tables_schema_lc:
            case_mismatch.append(t)
        else:
            truly_missing.append(t)

    if truly_missing:
        docs_with_missing += 1
    if case_mismatch:
        docs_with_case_mis += 1

    bulk_ops.append(
        UpdateOne(
            {"_id": doc["_id"]},
            {"$set": {
                "tables_used": tables_used,
                "audit_table_case_mismatch": case_mismatch,
                "audit_table_missing": truly_missing,
            }}
        )
    )

# ───────────────────────────────────────────────────────────────
# 3.  Execute bulk write
# ───────────────────────────────────────────────────────────────
if bulk_ops:
    res = coll.bulk_write(bulk_ops, ordered=False)
    print(
        f"[{args.collection_name}] updated {res.modified_count} docs – "
        f"{docs_with_case_mis} with case-only mismatches; "
        f"{docs_with_missing} with missing tables."
    )
else:
    print("No documents found – nothing to update.")

# ───────────────────────────────────────────────────────────────
# 4. Query sample with _id 3
# ───────────────────────────────────────────────────────────────
sample_id = 3
sample_doc = coll.find_one({"_id": sample_id}, {'tables_used': 1, 'used_columns': 1, 'audit_table_case_mismatch': 1, 'audit_table_missing': 1})
if sample_doc:
    print(f"\nSample with _id={sample_id}:")
    print(f"Tables used: {sample_doc.get('tables_used', [])}")
    print(f"Used columns: {sample_doc.get('used_columns', [])}")
    print(f"Case mismatches: {sample_doc.get('audit_table_case_mismatch', [])}")
    print(f"Missing tables: {sample_doc.get('audit_table_missing', [])}")
else:
    print(f"\nSample with _id={sample_id} not found.")

client.close() 