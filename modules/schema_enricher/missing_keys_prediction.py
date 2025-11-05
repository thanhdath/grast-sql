#!/usr/bin/env python3
"""
Generate primary key and foreign key candidates using OpenAI from the existing schema list.

- Reads documents from MongoDB (expects `schema` list in each doc)
- Groups schema by table for readability and prompts gpt-4.1-mini
- Expects JSON output with primary keys as ["table.column", ...] and
  foreign keys as [["src_table.col", "trg_table.col"], ...]
- Saves results to `generated_primary_keys` and `generated_foreign_keys`
  in the same formats as `primary_keys` and `foreign_keys` fields:
  - generated_primary_keys: { table: ["col", ...] }
  - generated_foreign_keys: { src_table: [{
        "from": "src_col",
        "to": "trg_table.trg_col",
        "ref_table": "trg_table",
        "on_update": "NO ACTION",
        "on_delete": "NO ACTION",
        "match": "SIMPLE"
    }, ...] }

Usage:
    python spider_2_14_generate_keys_with_openai.py \
        --mongo-uri mongodb://localhost:27017 \
        --collection-name spider2_lite_samples \
        --model gpt-4.1-mini \
        --limit 50

Notes:
- Requires OPENAI_API_KEY in environment. This script will also load /home/datht/mats/.env
- By default, only processes docs missing these generated fields unless --force-update is used
"""
from __future__ import annotations

import argparse
import json
import os
import re
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne, UpdateMany, errors


from openai import OpenAI 
import tiktoken



# ───────────────────────────── CLI ──────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate PK/FK predictions with OpenAI and save to MongoDB.")
    p.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides)")
    p.add_argument("--collection-name", default="spider2_lite_samples",
                   help="MongoDB collection name (default: %(default)s)")
    p.add_argument("--model", default="gpt-4.1-mini",
                   help="OpenAI model to use (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of documents to process (for testing)")
    p.add_argument("--skip-processed", action="store_true", default=False,
                   help="Skip docs that already have generated keys (default: True)")
    p.add_argument("--force-update", action="store_true",
                   help="Force update all documents, ignoring existing generated keys")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not write to MongoDB; only show prompts and parsed results")
    return p.parse_args()




_openai_client: Any = None

def get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    try:
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not available")
        _openai_client = OpenAI()
        return _openai_client
    except Exception:
        _openai_client = None
        return None


def call_openai_for_keys(prompt: str, model: str = "gpt-4.1-mini") -> str:
    """Call OpenAI to get JSON answer. Returns empty string on error."""
    try:
        client = get_openai_client()
        if client is None:
            return ""
        params: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        # GPT-5 models require max_completion_tokens and default temperature (omit param)
        if model.lower().startswith("gpt-5"):
            params["max_completion_tokens"] = 32048
        else:
            params["temperature"] = 0.0
            params["max_tokens"] = 32048
        resp = client.chat.completions.create(**params)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return ""


# ───────────────────────────── Helpers ──────────────────────────

def group_schema_by_table(schema: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for fullname in sorted(schema):
        if "." not in fullname:
            # skip malformed entries
            continue
        table, col = fullname.split(".", 1)
        grouped.setdefault(table, []).append(col)
    # Sort columns within each table
    for table in list(grouped.keys()):
        grouped[table] = sorted(grouped[table])
    return grouped


def is_key_candidate(column_name: str) -> bool:
    name = (column_name or "").lower()
    return name.endswith("_id") or name == "id" or ("uuid" in name)


def filter_grouped_to_key_candidates(grouped: Dict[str, List[str]]) -> Dict[str, List[str]]:
    filtered: Dict[str, List[str]] = {}
    for table, cols in grouped.items():
        cand = [c for c in cols if is_key_candidate(c)]
        if cand:
            filtered[table] = cand
    return filtered


def build_prompt(db_id: str, grouped: Dict[str, List[str]]) -> str:
    parts: List[str] = []
    parts.append(
        """You are given a database schema grouped by table. Task: predict primary keys and foreign keys.
Think carefully first, but ONLY output a final JSON object with these fields:
- primary_keys: an array of strings in the format 'table.column'
- foreign_keys: an array of pairs [ 'src_table.src_col', 'trg_table.trg_col' ]
Note: Primary and foreign key column names often end with 'id'. Prefer id/key columns.
Avoid using date/time/timestamp columns in primary keys.
Foreign keys must reference a different target table (no same-table FKs).
Do not include any explanations or extra text outside the JSON. Make sure the table.column is in the schema."""
    )
    parts.append(f"Database: {db_id}")
    parts.append("\nSchema (candidate key columns by table):")
    for table in sorted(grouped.keys()):
        cols = ", ".join(grouped[table])
        parts.append(f"- {table}: {cols}")
    parts.append("\nReturn JSON only, e.g.: {\"primary_keys\":[\"t.id\"],\"foreign_keys\":[[\"a.x\",\"b.y\"]]}")
    return "\n".join(parts)


def extract_json_block(text: str) -> Optional[str]:
    """Best-effort extraction of the first JSON object from text."""
    if not text:
        return None
    # Find first { ... } block
    start = text.find("{")
    if start == -1:
        return None
    # Simple brace matching
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def parse_openai_answer_to_structs(answer: str,
                                   schema_grouped: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict[str, Any]]]]:
    """
    Convert OpenAI JSON answer into target MongoDB field formats.

    Returns:
        (generated_primary_keys, generated_foreign_keys)
    """
    # Try parse JSON
    raw_json: Optional[str] = None
    try:
        raw_json = answer
        parsed = json.loads(answer)
    except Exception:
        raw_json = extract_json_block(answer)
        if not raw_json:
            raise ValueError("Could not parse JSON from model output")
        parsed = json.loads(raw_json)

    pk_list: List[str] = parsed.get("primary_keys", []) or []
    fk_list: List[Any] = parsed.get("foreign_keys", []) or []

    # Build generated_primary_keys: {table: [col, ...]}
    valid_cols_by_table = schema_grouped
    gen_pks: Dict[str, List[str]] = {}
    for item in pk_list:
        if not isinstance(item, str) or "." not in item:
            continue
        table, col = item.split(".", 1)
        if table in valid_cols_by_table and col in valid_cols_by_table[table]:
            gen_pks.setdefault(table, [])
            if col not in gen_pks[table]:
                gen_pks[table].append(col)

    # Build generated_foreign_keys: {src_table: [{...}]}
    gen_fks: Dict[str, List[Dict[str, Any]]] = {}

    def add_fk(src_table: str, src_col: str, trg_table: str, trg_col: str) -> None:
        if src_table not in valid_cols_by_table or trg_table not in valid_cols_by_table:
            return
        if src_col not in valid_cols_by_table[src_table] or trg_col not in valid_cols_by_table[trg_table]:
            return
        fk_obj = {
            "from": src_col,
            "to": f"{trg_table}.{trg_col}",
            "ref_table": trg_table,
            "on_update": "NO ACTION",
            "on_delete": "NO ACTION",
            "match": "SIMPLE",
        }
        gen_fks.setdefault(src_table, []).append(fk_obj)

    for item in fk_list:
        # Expect ["src_table.src_col", "trg_table.trg_col"]
        if isinstance(item, list) and len(item) == 2:
            left, right = item
            if isinstance(left, str) and isinstance(right, str) and "." in left and "." in right:
                s_table, s_col = left.split(".", 1)
                t_table, t_col = right.split(".", 1)
                add_fk(s_table, s_col, t_table, t_col)
        elif isinstance(item, dict):
            # Accept object variant: {"from":"t.col","to":"u.col"}
            left = item.get("from")
            right = item.get("to")
            if isinstance(left, str) and isinstance(right, str) and "." in left and "." in right:
                s_table, s_col = left.split(".", 1)
                t_table, t_col = right.split(".", 1)
                add_fk(s_table, s_col, t_table, t_col)
        elif isinstance(item, str) and "->" in item:
            # Accept string variant: "t.col -> u.col"
            parts = [p.strip() for p in item.split("->", 1)]
            if len(parts) == 2 and "." in parts[0] and "." in parts[1]:
                s_table, s_col = parts[0].split(".", 1)
                t_table, t_col = parts[1].split(".", 1)
                add_fk(s_table, s_col, t_table, t_col)

    # Sort lists for stable output
    for t in list(gen_pks.keys()):
        gen_pks[t] = sorted(gen_pks[t])
    for t in list(gen_fks.keys()):
        gen_fks[t] = sorted(gen_fks[t], key=lambda x: (x.get("from", ""), x.get("to", "")))

    return gen_pks, gen_fks


def validate_generated_keys(schema: List[str],
                           gen_pks: Dict[str, List[str]],
                           gen_fks: Dict[str, List[Dict[str, Any]]]
                           ) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict[str, Any]]]]:
    """Filter PKs/FKs to ensure they exist in the provided schema list."""
    schema_set = set(schema)

    # Validate PKs
    valid_pks: Dict[str, List[str]] = {}
    for table, cols in (gen_pks or {}).items():
        for col in cols or []:
            if f"{table}.{col}" in schema_set:
                valid_pks.setdefault(table, []).append(col)

    # Validate FKs
    valid_fks: Dict[str, List[Dict[str, Any]]] = {}
    for src_table, fk_list in (gen_fks or {}).items():
        if not fk_list:
            continue
        filtered: List[Dict[str, Any]] = []
        for fk in fk_list:
            src_col = (fk or {}).get("from", "")
            to_full = (fk or {}).get("to", "")
            if not src_col or not to_full or "." not in to_full:
                continue
            src_full = f"{src_table}.{src_col}"
            if src_full not in schema_set or to_full not in schema_set:
                continue
            trg_table, trg_col = to_full.split('.', 1)
            # Ensure ref_table consistency
            cleaned_fk = {
                "from": src_col,
                "to": f"{trg_table}.{trg_col}",
                "ref_table": trg_table,
                "on_update": (fk or {}).get("on_update", "NO ACTION"),
                "on_delete": (fk or {}).get("on_delete", "NO ACTION"),
                "match": (fk or {}).get("match", "SIMPLE"),
            }
            filtered.append(cleaned_fk)
        if filtered:
            # Sort for stability
            filtered = sorted(filtered, key=lambda x: (x.get("from", ""), x.get("to", "")))
            valid_fks[src_table] = filtered

    # Sort PK lists
    for t in list(valid_pks.keys()):
        valid_pks[t] = sorted(valid_pks[t])

    return valid_pks, valid_fks


def count_tokens_approx(text: str) -> int:
    """Count tokens using tiktoken if available; fallback to ~1 token per 4 chars."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))


# ───────────────────────────── Main ─────────────────────────────

def predict_missing_keys_for_schema(schema_info: Dict[str, Any], model: str, db_id: str) -> None:
    """Predict missing primary keys and foreign keys for a schema_info dict."""
    schema_list: List[str] = sorted(schema_info.get("schema", []))
    if not schema_list:
        schema_info["generated_primary_keys"] = {}
        schema_info["generated_foreign_keys"] = {}
        return
    
    grouped = group_schema_by_table(schema_list)
    grouped_candidates = filter_grouped_to_key_candidates(grouped)
    # If no candidates identified, fall back to full grouped schema to avoid empty prompt
    grouped_for_prompt = grouped_candidates if grouped_candidates else grouped
    prompt = build_prompt(db_id, grouped_for_prompt)
    
    answer = call_openai_for_keys(prompt, model)
    if not answer:
        schema_info["generated_primary_keys"] = {}
        schema_info["generated_foreign_keys"] = {}
        return
    
    try:
        # Parse using the same candidate set used in the prompt
        gen_pks, gen_fks = parse_openai_answer_to_structs(answer, grouped_for_prompt)
    except Exception as e:
        print(f"[{db_id}]: Failed to parse model output: {e}")
        schema_info["generated_primary_keys"] = {}
        schema_info["generated_foreign_keys"] = {}
        return
    
    # Validate generated keys against the schema
    valid_pks, valid_fks = validate_generated_keys(schema_list, gen_pks, gen_fks)
    
    schema_info["generated_primary_keys"] = valid_pks
    schema_info["generated_foreign_keys"] = valid_fks


def main() -> None:
    args = parse_args()
    load_dotenv("/home/datht/mats/.env")
    mongo_uri = args.mongo_uri
    client = MongoClient(mongo_uri)
    coll = client["mats"][args.collection_name]

    # Build query set
    if args.force_update:
        docs = list(coll.find({}, {"schema": 1, "db_id": 1, "db_type": 1}))
        print(f"FORCE UPDATE: Processing all {len(docs)} documents")
    elif args.skip_processed:
        docs = list(coll.find({
            "$or": [
                {"generated_primary_keys": {"$exists": False}},
                {"generated_foreign_keys": {"$exists": False}}
            ]
        }, {"schema": 1, "db_id": 1, "db_type": 1}))
        print(f"Found {len(docs)} documents missing generated keys or empty")
    else:
        docs = list(coll.find({}, {"schema": 1, "db_id": 1, "db_type": 1}))
        print(f"Processing all {len(docs)} documents")

    if args.limit is not None:
        docs = docs[: args.limit]
        print(f"Limited to {len(docs)} documents for this run")

    if not docs:
        print("No documents to process")
        client.close()
        return

    # Group docs by (db_id, db_type) and union schema per group
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d in docs:
        db_id = d.get("db_id", "unknown_db")
        db_type = d.get("db_type", "unknown")
        schema: List[str] = d.get("schema", []) or []
        sample_id = d.get("_id")
        key = (db_id, db_type)
        if key not in groups:
            groups[key] = {"db_id": db_id, "db_type": db_type, "schema": set(schema), "doc_ids": [sample_id]}
        else:
            groups[key]["schema"].update(schema)
            groups[key].setdefault("doc_ids", []).append(sample_id)

    # Compute token counts for unique db_id (deduplicated by db_id) before OpenAI calls
    dbid_to_schema: Dict[str, set] = {}
    for (db_id, _db_type), info in groups.items():
        if db_id not in dbid_to_schema:
            dbid_to_schema[db_id] = set(info["schema"])  # type: ignore[assignment]
        else:
            dbid_to_schema[db_id].update(info["schema"])  # type: ignore[union-attr]

    total_prompt_tokens = 0
    for db_id, schema_set in sorted(dbid_to_schema.items()):
        schema_list = sorted(list(schema_set))
        grouped_schema = group_schema_by_table(schema_list)
        prompt_preview = build_prompt(db_id, grouped_schema)
        total_prompt_tokens += count_tokens_approx(prompt_preview)
    print(f"Token estimate (dedup by db_id): {total_prompt_tokens} tokens across {len(dbid_to_schema)} databases")

    # Ask for confirmation before executing any OpenAI calls
    try:
        confirm = input("Proceed with OpenAI calls? Type 'yes' to continue: ").strip().lower()
    except EOFError:
        confirm = ""
    if confirm != "yes":
        print("Aborted by user.")
        client.close()
        return

    # Write updates to MongoDB immediately per (db_id, db_type) group

    for (db_id, db_type), info in sorted(groups.items()):
        schema_list: List[str] = sorted(list(info["schema"]))
        if not schema_list:
            print(f"[{db_id} | {db_type}]: No schema found, skipping")
            continue

        # Build schema_info dict for predict_missing_keys_for_schema
        schema_info_for_prediction = {
            "schema": schema_list,
            "generated_primary_keys": {},
            "generated_foreign_keys": {},
        }
        
        if args.dry_run:
            grouped = group_schema_by_table(schema_list)
            grouped_candidates = filter_grouped_to_key_candidates(grouped)
            grouped_for_prompt = grouped_candidates if grouped_candidates else grouped
            prompt = build_prompt(db_id, grouped_for_prompt)
            print("\n===== PROMPT =====")
            print(prompt)
            
            answer = call_openai_for_keys(prompt, args.model)
            print("\n===== RAW ANSWER =====")
            print(answer)
            
            try:
                gen_pks, gen_fks = parse_openai_answer_to_structs(answer, grouped_for_prompt)
                print("\n===== PARSED =====")
                print(json.dumps({
                    "generated_primary_keys": gen_pks,
                    "generated_foreign_keys": gen_fks,
                }, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"[{db_id} | {db_type}]: Failed to parse model output: {e}")
            # Skip DB write in dry-run
            continue

        # Use the unified function to predict keys
        predict_missing_keys_for_schema(schema_info_for_prediction, args.model, db_id)
        valid_pks = schema_info_for_prediction["generated_primary_keys"]
        valid_fks = schema_info_for_prediction["generated_foreign_keys"]

        # Update all documents with this (db_id, db_type) immediately
        try:
            res = coll.update_many(
                {"db_id": db_id, "db_type": db_type},
                {"$set": {
                    "generated_primary_keys": valid_pks,
                    "generated_foreign_keys": valid_fks,
                }}
            )
            print(f"[{db_id} | {db_type}]: generated {sum(len(v) for v in valid_pks.values())} PK cols, {sum(len(v) for v in valid_fks.values())} FK entries; updated {res.modified_count} docs")
        except errors.WriteError as we:
            msg = str(we)
            if "larger than" in msg or getattr(we, "code", None) == 17419:
                # Fallback: update each document individually using its _id to avoid exceeding doc size
                doc_ids: List[Any] = info.get("doc_ids", [])  # type: ignore[assignment]
                print(doc_ids)
                modified_total = 0
                inserted_total = 0
                for doc_id in doc_ids:
                    try:
                        r = coll.update_one(
                            {"_id": doc_id},
                            {"$set": {
                                "generated_primary_keys": valid_pks,
                                "generated_foreign_keys": valid_fks,
                            }}
                        )
                        modified_total += getattr(r, "modified_count", 0)
                    except errors.WriteError as we_single:
                        # If even per-doc update fails due to size, store in fallback collection with reference
                        try:
                            fallback_coll = client["mats"]["spider2_lite_samples_2"]
                            fallback_coll.update_one(
                                {"_id": doc_id},
                                {"$set": {
                                    "sample_id": doc_id,
                                    "db_id": db_id,
                                    "db_type": db_type,
                                    "generated_primary_keys": valid_pks,
                                    "generated_foreign_keys": valid_fks,
                                    "model": args.model,
                                    "generated_at": dt.datetime.utcnow(),
                                }},
                                upsert=True,
                            )
                            inserted_total += 1
                        except Exception as e_fallback:
                            print(f"[{db_id} | {db_type} | _id={doc_id}]: Fallback insert error: {e_fallback}")
                            continue
                print(f"[{db_id} | {db_type}]: Large doc fallback: updated {modified_total}/{len(doc_ids)} docs individually; inserted {inserted_total} refs into spider2_lite_samples_2")
            else:
                print(f"[{db_id} | {db_type}]: Mongo WriteError: {we}")
                continue

    # No final bulk write; updates were applied per-group immediately

    client.close()
    print("Done.")
