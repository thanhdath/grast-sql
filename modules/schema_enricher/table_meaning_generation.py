#!/usr/bin/env python3
"""
spider_2_15_generate_table_meaning.py
────────────────────────────────────────────────
Generate a concise one-sentence meaning for each table in a database using OpenAI.

Inputs (read from MongoDB):
- table name
- list of column descriptions: (column name, column meaning, column values)
- external knowledge (stored under `evidence`)
- DDL.csv files (if present) with columns [table_name, description, ddl]; description used as table meaning

Output (written to MongoDB):
- table_meaning: { table_name: "one-sentence meaning" }

Behavior
- First, read DDL.csv to populate table_meaning directly when available
- Then, compute token estimate and ask confirmation
- Finally, call OpenAI only for tables still missing meaning

Usage
-----
python spider_2_15_generate_table_meaning.py \
  --mongo-uri mongodb://localhost:27017 \
  --collection-name spider2_lite_samples \
  --model gpt-4.1-mini \
  --limit 20

Optional filters:
- --only-db-id DB_NAME          # restrict to a single database id
- --only-table TABLE_NAME       # restrict to a single table name
- --dry-run                     # do not write to MongoDB; print prompts/answers
- --force-update                # re-generate even if table_meaning exists

Notes
-----
- Requires OPENAI_API_KEY in environment. This script will also load /home/datht/mats/.env
- Follows OpenAI client usage patterns from other scripts in this repo
"""
from __future__ import annotations

import argparse
import os
import datetime as dt
from typing import Any, Dict, List, Tuple
from pathlib import Path
import csv
import re
from multiprocessing import Pool

from dotenv import load_dotenv
from pymongo import MongoClient, errors
from openai import OpenAI  # type: ignore
import tiktoken  # type: ignore

# ───────────────────────────── CLI ──────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a one-sentence meaning per table using OpenAI and save to MongoDB.")
    p.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides)")
    p.add_argument("--collection-name", default="spider2_lite_samples",
                   help="MongoDB collection name (default: %(default)s)")
    p.add_argument("--model", default="gpt-4.1-mini",
                   help="OpenAI model to use (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of (db_id, db_type) groups to process (for testing)")
    p.add_argument("--skip-processed", action="store_true", default=False,
                   help="Skip docs that already have table_meaning (default: True)")
    p.add_argument("--force-update", action="store_true",
                   help="Force update all documents, ignoring existing table_meaning")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not write to MongoDB; only show prompts and answers")
    p.add_argument("--only-db-id", default=None,
                   help="If provided, only process this db_id")
    p.add_argument("--only-table", default=None,
                   help="If provided, only generate meaning for this table name (within selected db_id if set)")
    p.add_argument("--max-values", type=int, default=3,
                   help="Max number of example values per column in prompt (default: %(default)s)")
    p.add_argument("--max-evidence-chars", type=int, default=1500,
                   help="Max total characters of external knowledge to include (default: %(default)s)")
    p.add_argument("--base-dir", type=Path, default=Path("/home/datht/Spider2/spider2-lite"),
                   help="Root directory containing Spider2.0-lite dataset (default: %(default)s)")
    p.add_argument("--prompts-dir", type=Path, default=Path("./prompts-table-meaning"),
                   help="Directory to write prompts (default: %(default)s)")
    p.add_argument("--processes", type=int, default=8,
                   help="Number of worker processes for prompt generation/token estimation (default: %(default)s)")
    return p.parse_args()



# ───────────────────────────── OpenAI ───────────────────────────

_openai_client: Any = None

def get_openai_client() -> Any:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    _openai_client = OpenAI()
    return _openai_client


def call_openai_for_table_meaning(prompt: str, model: str = "gpt-4.1-mini") -> str:
    """Call OpenAI to get a single-sentence table meaning. Returns empty string on error."""
    try:
        client = get_openai_client()
        params: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        if model.lower().startswith("gpt-5"):
            params["max_completion_tokens"] = 2048
        else:
            params["temperature"] = 0.0
            params["max_tokens"] = 2048
        resp = client.chat.completions.create(**params)
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


# ───────────────────────────── Helpers ──────────────────────────

def union_dict(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow union where right wins on key conflicts."""
    out = dict(left)
    out.update(right or {})
    return out


def build_table_prompt(db_id: str,
                       table_name: str,
                       columns: List[Dict[str, Any]],
                       evidence_snippet: str,
                       max_values: int) -> str:
    parts: List[str] = []
    parts.append(
        """You are given information about a database table. Task: write ONE concise sentence that explains the meaning of the table. If table is abbreviated, explain what it is. Don't need to include again the table name or database name in the answer."""
    )
    parts.append(f"Database: {db_id}")
    parts.append(f"Table: {table_name}")

    if columns:
        parts.append("\nColumns (name: meaning | example values):")
        for col in columns:
            name = col.get("name", "")
            meaning = col.get("meaning", "")
            values = col.get("values", []) or []
            col_type = col.get("type", "")
            if max_values and isinstance(values, list):
                values = values[: max_values]
            # Truncate each value to max 100 words
            values = [truncate_words(str(v), 100) for v in values]
            values_str = ", ".join([str(v) for v in values]) if values else ""
            type_str = f" [{col_type}]" if col_type else ""
            parts.append(f"- {name}{type_str}: {meaning} | {values_str}")

    if evidence_snippet:
        parts.append("\nExternal knowledge:")
        parts.append(evidence_snippet)

    parts.append("\nReturn only the one-sentence meaning.")
    return "\n".join(parts)


def truncate_text(text: str, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def get_token_encoder_for_model(model: str):
    """Return a tiktoken encoder appropriate for the model."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens_with_tiktoken(text: str, model: str) -> int:
    enc = get_token_encoder_for_model(model)
    return len(enc.encode(text))


def get_db_path(base_dir: Path, db_type: str, db_id: str) -> Path:
    return base_dir / "resource" / "databases" / db_type / db_id


def find_ddl_files_for_db(base_dir: Path, db_type: str, db_id: str) -> List[Path]:
    ddl_files: List[Path] = []
    db_path = get_db_path(base_dir, db_type, db_id)
    if db_path.exists():
        ddl_files = list(db_path.rglob("DDL.csv"))
    return ddl_files


def parse_ddl_csv(file_path: Path) -> Dict[str, str]:
    """Return mapping {table_name: description} from a DDL.csv file with headers."""
    mapping: Dict[str, str] = {}
    try:
        with file_path.open(encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            # Normalize headers to lowercase
            fieldnames = [f.lower() for f in (reader.fieldnames or [])]
            # Common aliases
            table_key = None
            desc_key = None
            for name in fieldnames:
                if name in ("table_name", "table"):
                    table_key = name
                if name in ("description", "table_description"):
                    desc_key = name
            if table_key is None or desc_key is None:
                return mapping
            for row in reader:
                t = (row.get(table_key) or "").strip()
                d = (row.get(desc_key) or "").strip()
                if t and d:
                    mapping.setdefault(t, d)
    except Exception:
        return mapping
    return mapping


def normalize_table_name_for_schema(table_name: str) -> str:
    """From a column fullname 'a.b.c' or 'a.b', return the table portion (all but the last segment).
    Example: 'schema.table.column' -> 'schema.table'; 'table.column' -> 'table'"""
    if "." not in table_name:
        return table_name
    # For a fullname of a column, keep everything except the last segment (column)
    return table_name.rsplit(".", 1)[0]


def normalize_table_name_from_ddl(db_id: str, ddl_table_name: str) -> str:
    """Convert DDL table name to the schema-style table key used in Mongo 'schema' (usually 'schema.table' or 'table').
    Strategy:
    - If ddl_table_name starts with '{db_id}.', strip that prefix
    - If there are >= 2 segments, return last two joined (schema.table). Else return as-is
    """
    name = ddl_table_name.strip()
    if name.startswith(f"{db_id}."):
        name = name[len(db_id) + 1 :]
    parts = name.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return name


def to_lower_set(keys: List[str]) -> set:
    return set(k.lower() for k in keys)


def expand_keys_with_aliases_lower(keys: List[str]) -> set:
    s = set()
    for k in keys:
        kl = k.lower()
        s.add(kl)
        # Add last segment alias
        if "." in k:
            s.add(k.rsplit(".", 1)[-1].lower())
        else:
            s.add(kl)
    return s


def sanitize_filename(name: str) -> str:
    name = name.strip().replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name


def write_prompt_file(base_dir: Path, db_id: str, table: str, content: str) -> None:
    db_dir = base_dir / sanitize_filename(db_id)
    db_dir.mkdir(parents=True, exist_ok=True)
    file_path = db_dir / f"{sanitize_filename(table)}.txt"
    try:
        file_path.write_text(content, encoding="utf-8")
    except Exception:
        pass


def worker_estimate_group(task: Tuple[Tuple[str, str], Dict[str, Any], Dict[str, Any]]) -> Tuple[int, int]:
    (db_id, db_type), info_preview, cfg = task
    schema_list_preview: List[str] = sorted(list(info_preview.get("schema", set())))
    if not schema_list_preview:
        return (0, 0)
    evidence_list_preview: List[str] = info_preview.get("evidence_list", []) or []
    evidence_concat_preview = "\n\n".join([e for e in evidence_list_preview if isinstance(e, str) and e.strip()])
    evidence_snippet_preview = evidence_concat_preview  # no truncation
    col_meaning_map_preview: Dict[str, str] = info_preview.get("column_meaning", {}) or {}
    col_info_map_preview: Dict[str, Dict[str, Any]] = info_preview.get("column_info", {}) or {}
    existing_table_meaning_preview: Dict[str, str] = info_preview.get("existing_table_meaning", {}) or {}
    existing_keys_lower = expand_keys_with_aliases_lower(list(existing_table_meaning_preview.keys()))

    # Collect tables
    tables_preview: List[str] = []
    for fullname in schema_list_preview:
        if "." not in fullname:
            continue
        t_full = normalize_table_name_for_schema(fullname)
        if t_full not in tables_preview:
            tables_preview.append(t_full)
    tables_preview = sorted(tables_preview)

    if cfg.get("only_table"):
        only_table = cfg["only_table"]
        tables_preview = [t for t in tables_preview if t == only_table]

    # Skip existing
    tables_preview = [t for t in tables_preview if (t.lower() not in existing_keys_lower and t.rsplit(".", 1)[-1].lower() not in existing_keys_lower)]

    total_tokens = 0
    total_tables = 0
    for table in tables_preview:
        columns_for_table_preview: List[Dict[str, Any]] = []
        prefix = f"{table}."
        for fullname in schema_list_preview:
            if not fullname.startswith(prefix):
                continue
            col = fullname[len(prefix):]
            meaning = col_meaning_map_preview.get(fullname, "")
            ci = col_info_map_preview.get(fullname, {}) or {}
            col_type = ci.get("type", "")
            values = ci.get("similar_values", []) or []
            values = [truncate_words(str(v), 100) for v in values]
            columns_for_table_preview.append({
                "name": col,
                "meaning": meaning,
                "values": values,
                "type": col_type,
            })
        if not columns_for_table_preview:
            continue
        prompt_preview = build_table_prompt(
            db_id=db_id,
            table_name=table,
            columns=columns_for_table_preview,
            evidence_snippet=evidence_snippet_preview,
            max_values=cfg.get("max_values", 3),
        )
        # Write prompt file
        write_prompt_file(Path(cfg["prompts_dir"]), db_id, table, prompt_preview)
        total_tokens += count_tokens_with_tiktoken(prompt_preview, cfg["model"])  # type: ignore[index]
        total_tables += 1
    return (total_tokens, total_tables)


# ───────────────────────────── Main ─────────────────────────────

def generate_table_meanings_for_schema(schema_info: Dict[str, Any], model: str, db_id: str, evidence: str) -> None:
    """Generate table meanings for a schema_info dict. Assumes column meanings are already generated."""
    tables = {}
    for full_col in schema_info["schema"]:
        table, _ = full_col.split(".", 1)
        tables.setdefault(table, []).append(full_col)
    
    # Generate table meanings
    for table, cols in tables.items():
        column_list = []
        for full_col in cols:
            col_name = full_col.split(".", 1)[1]
            col_info = schema_info["column_info"][full_col]
            col_type = col_info["type"]
            samples = col_info.get("similar_values", [])
            col_meaning = schema_info["column_meaning"].get(full_col, f"{col_name} ({col_type})")
            
            column_list.append({
                "name": col_name,
                "meaning": col_meaning,
                "values": samples[:3],
                "type": col_type,
            })
        
        prompt = build_table_prompt(
            db_id=db_id,
            table_name=table,
            columns=column_list,
            evidence_snippet=evidence,
            max_values=3,
        )
        
        meaning = call_openai_for_table_meaning(prompt, model)
        if meaning:
            meaning = meaning.split(".")[0].strip()
            if meaning and not meaning.endswith("."):
                meaning += "."
            schema_info["table_meaning"][table] = meaning


def main() -> None:
    args = parse_args()
    load_dotenv("/home/datht/mats/.env")
    mongo_uri = args.mongo_uri
    client = MongoClient(mongo_uri)
    coll = client["mats"][args.collection_name]

    # Fetch candidate docs
    query: Dict[str, Any] = {}
    projection = {
        "schema": 1,
        "db_id": 1,
        "db_type": 1,
        "column_meaning": 1,
        "column_info": 1,
        "evidence": 1,
        "table_meaning": 1,
    }

    if args.only_db_id:
        query["db_id"] = args.only_db_id

    if args.force_update:
        docs = list(coll.find(query, projection))
        print(f"FORCE UPDATE: Processing all {len(docs)} documents matched")
    elif args.skip_processed and not args.only_table:
        # Skip when table_meaning exists (map not empty)
        docs = list(coll.find({**query, "$or": [{"table_meaning": {"$exists": False}}, {"table_meaning": {}}]}, projection))
        print(f"Found {len(docs)} documents missing table_meaning or empty")
    else:
        docs = list(coll.find(query, projection))
        print(f"Processing {len(docs)} candidate documents")

    if not docs:
        print("No documents to process")
        client.close()
        return

    # Group by (db_id, db_type)
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d in docs:
        db_id = d.get("db_id", "unknown_db")
        db_type = d.get("db_type", "unknown")
        schema: List[str] = d.get("schema", []) or []
        column_meaning: Dict[str, str] = d.get("column_meaning", {}) or {}
        column_info: Dict[str, Dict[str, Any]] = d.get("column_info", {}) or {}
        evidence: str = d.get("evidence", "") or ""
        existing_table_meaning: Dict[str, str] = d.get("table_meaning", {}) or {}
        doc_id = d.get("_id")

        key = (db_id, db_type)
        if key not in groups:
            groups[key] = {
                "db_id": db_id,
                "db_type": db_type,
                "schema": set(schema),
                "column_meaning": dict(column_meaning),
                "column_info": dict(column_info),
                "evidence_list": [evidence] if evidence else [],
                "existing_table_meaning": dict(existing_table_meaning),
                "doc_ids": [doc_id],
            }
        else:
            groups[key]["schema"].update(schema)
            groups[key]["column_meaning"] = union_dict(groups[key]["column_meaning"], column_meaning)
            groups[key]["column_info"] = union_dict(groups[key]["column_info"], column_info)
            groups[key]["existing_table_meaning"] = union_dict(groups[key]["existing_table_meaning"], existing_table_meaning)
            if evidence:
                groups[key].setdefault("evidence_list", []).append(evidence)
            groups[key].setdefault("doc_ids", []).append(doc_id)

    # Optionally limit number of groups for tests
    groups_items = list(sorted(groups.items(), key=lambda kv: kv[0]))
    if args.limit is not None:
        groups_items = groups_items[: args.limit]

    # 1) Read DDL.csv meanings and update MongoDB immediately, collect for skipping
    total_ddl_tables = 0
    for (db_id, db_type), info in groups_items:
        ddl_files = find_ddl_files_for_db(args.base_dir, db_type, db_id)
        if not ddl_files:
            continue
        ddl_meanings: Dict[str, str] = {}
        samples_to_print: List[Tuple[str, str, str]] = []
        for ddl_file in ddl_files:
            mapping = parse_ddl_csv(ddl_file)
            for t_raw, desc in mapping.items():
                t_norm = normalize_table_name_from_ddl(db_id, t_raw)
                if t_norm not in ddl_meanings:
                    ddl_meanings[t_norm] = desc
                    if len(samples_to_print) < 5:
                        samples_to_print.append((str(ddl_file), t_norm, desc))
        if not ddl_meanings:
            continue

        # Print a few samples
        print(f"[{db_id} | {db_type}] Found table meanings in DDL files:")
        for path_str, t_name, desc in samples_to_print:
            print(f"  - {path_str} :: {t_name} => {desc[:120]}{'...' if len(desc) > 120 else ''}")

        # Count and accumulate
        print(f"[{db_id} | {db_type}] DDL tables with meaning: {len(ddl_meanings)}")
        total_ddl_tables += len(ddl_meanings)

        # Update existing_table_meaning cache (in-memory) to reflect DDL meanings (do not write aliases to DB)
        info_existing: Dict[str, str] = info.get("existing_table_meaning", {}) or {}
        info["existing_table_meaning"] = union_dict(info_existing, ddl_meanings)

        # Prepare $set update for table_meaning.<table> paths
        set_fields: Dict[str, Any] = {}
        for t, desc in ddl_meanings.items():
            set_fields[f"table_meaning.{t}"] = desc
        if set_fields and not args.dry_run:
            try:
                res = coll.update_many(
                    {"db_id": db_id, "db_type": db_type},
                    {"$set": set_fields}
                )
                print(f"[{db_id} | {db_type}] DDL update: set {len(set_fields)} table_meaning entries; modified {res.modified_count} docs")
            except errors.WriteError as we:
                print(f"[{db_id} | {db_type}] DDL update write error: {we}")

    print(f"Total DDL tables with meaning: {total_ddl_tables}")

    # 2) Pre-compute token estimate across remaining tables and prompt for confirmation
    # Prepare tasks for multiprocessing
    cfg = {
        "model": args.model,
        "prompts_dir": str(args.prompts_dir),
        "max_values": args.max_values,
        "only_table": args.only_table,
    }
    tasks: List[Tuple[Tuple[str, str], Dict[str, Any], Dict[str, Any]]] = [
        (key, info, cfg) for key, info in groups_items
    ]

    total_prompt_tokens = 0
    total_tables = 0
    if args.processes and args.processes > 1:
        with Pool(processes=args.processes) as pool:
            for tkns, tbls in pool.imap_unordered(worker_estimate_group, tasks, chunksize=1):
                total_prompt_tokens += tkns
                total_tables += tbls
    else:
        for task in tasks:
            tkns, tbls = worker_estimate_group(task)
            total_prompt_tokens += tkns
            total_tables += tbls

    print(f"Token estimate: {total_prompt_tokens} tokens across {total_tables} tables in {len(groups_items)} databases")

    # Ask for confirmation before executing any OpenAI calls
    try:
        confirm = input("Proceed with OpenAI calls? Type 'yes' to continue: ").strip().lower()
    except EOFError:
        confirm = ""
    if confirm != "yes":
        print("Aborted by user.")
        client.close()
        return

    # 3) Process each group for remaining tables only
    for (db_id, db_type), info in groups_items:
        schema_list: List[str] = sorted(list(info.get("schema", set())))
        if not schema_list:
            print(f"[{db_id} | {db_type}]: No schema found, skipping")
            continue

        # Build evidence snippet (do not truncate)
        evidence_list: List[str] = info.get("evidence_list", []) or []
        evidence_concat = "\n\n".join([e for e in evidence_list if isinstance(e, str) and e.strip()])
        evidence_snippet = evidence_concat

        # Aggregate per-table columns
        col_meaning_map: Dict[str, str] = info.get("column_meaning", {}) or {}
        col_info_map: Dict[str, Dict[str, Any]] = info.get("column_info", {}) or {}
        existing_table_meaning_map: Dict[str, str] = info.get("existing_table_meaning", {}) or {}
        # Build skip set (lowercased + aliases)
        existing_keys_lower_exec = expand_keys_with_aliases_lower(list(existing_table_meaning_map.keys()))

        # Collect all tables from schema (support multi-part names)
        tables: List[str] = []
        for fullname in schema_list:
            if "." not in fullname:
                continue
            t_full = normalize_table_name_for_schema(fullname)
            if t_full not in tables:
                tables.append(t_full)
        tables = sorted(tables)

        # Optional restriction to one table (for testing)
        if args.only_table:
            tables = [t for t in tables if t == args.only_table]
            if not tables:
                print(f"[{db_id} | {db_type}]: Table {args.only_table} not found in schema; skipping")
                continue

        # Skip tables with existing meaning (from Mongo or DDL), case-insensitive and alias-aware
        tables = [t for t in tables if (t.lower() not in existing_keys_lower_exec and t.rsplit(".", 1)[-1].lower() not in existing_keys_lower_exec)]

        table_meaning: Dict[str, str] = {}

        for table in tables:
            # Gather columns for this table
            columns_for_table: List[Dict[str, Any]] = []
            prefix = f"{table}."
            for fullname in schema_list:
                if not fullname.startswith(prefix):
                    continue
                col = fullname[len(prefix):]
                meaning = col_meaning_map.get(fullname, "")
                ci = col_info_map.get(fullname, {}) or {}
                col_type = ci.get("type", "")
                values = ci.get("similar_values", []) or []
                # Truncate each value to max 100 words
                values = [truncate_words(str(v), 100) for v in values]
                columns_for_table.append({
                    "name": col,
                    "meaning": meaning,
                    "values": values,
                    "type": col_type,
                })

            # If no descriptors available, still try with just column names
            if not columns_for_table:
                continue

            prompt = build_table_prompt(
                db_id=db_id,
                table_name=table,
                columns=columns_for_table,
                evidence_snippet=evidence_snippet,
                max_values=args.max_values,
            )

            if args.dry_run:
                print("\n===== PROMPT =====")
                print(prompt)

            answer = call_openai_for_table_meaning(prompt, args.model)
            if args.dry_run:
                print("\n===== RAW ANSWER =====")
                print(answer)

            # Keep only first sentence and strip newlines
            final_sentence = (answer or "").split(".")[0].strip()
            if final_sentence:
                if not final_sentence.endswith("."):
                    final_sentence = final_sentence + "."
                table_meaning[table] = final_sentence

        if args.dry_run:
            print("\n===== TABLE MEANING (PARSED) =====")
            print({k: table_meaning[k] for k in sorted(table_meaning.keys())})
            # Skip DB write in dry-run
            continue

        if not table_meaning:
            print(f"[{db_id} | {db_type}]: No table meanings generated; skipping write")
            continue

        # Update all documents in this group with the same db_id and db_type
        try:
            res = coll.update_many(
                {"db_id": db_id, "db_type": db_type},
                {"$set": {
                    "table_meaning": union_dict(info.get("existing_table_meaning", {}), table_meaning),
                    "table_meaning_model": args.model,
                    "table_meaning_generated_at": dt.datetime.utcnow(),
                }}
            )
            print(f"[{db_id} | {db_type}]: generated meanings for {len(table_meaning)} tables; updated {res.modified_count} docs")
        except errors.WriteError as we:
            msg = str(we)
            if "larger than" in msg or getattr(we, "code", None) == 17419:
                # Fallback: update one by one, and if still too large, write to side collection
                doc_ids: List[Any] = info.get("doc_ids", [])  # type: ignore[assignment]
                modified_total = 0
                inserted_total = 0
                for doc_id in doc_ids:
                    try:
                        r = coll.update_one(
                            {"_id": doc_id},
                            {"$set": {
                                "table_meaning": union_dict(info.get("existing_table_meaning", {}), table_meaning),
                                "table_meaning_model": args.model,
                                "table_meaning_generated_at": dt.datetime.utcnow(),
                            }}
                        )
                        modified_total += getattr(r, "modified_count", 0)
                    except errors.WriteError:
                        try:
                            fallback_coll = client["mats"][f"{args.collection_name}_fallback"]
                            fallback_coll.update_one(
                                {"_id": doc_id},
                                {"$set": {
                                    "sample_id": doc_id,
                                    "db_id": db_id,
                                    "db_type": db_type,
                                    "table_meaning": union_dict(info.get("existing_table_meaning", {}), table_meaning),
                                    "table_meaning_model": args.model,
                                    "table_meaning_generated_at": dt.datetime.utcnow(),
                                }},
                                upsert=True,
                            )
                            inserted_total += 1
                        except Exception as e_fallback:
                            print(f"[{db_id} | {db_type} | _id={doc_id}]: Fallback insert error: {e_fallback}")
                            continue
                print(f"[{db_id} | {db_type}]: Large doc fallback: updated {modified_total}/{len(doc_ids)} docs individually; inserted {inserted_total} refs into {args.collection_name}_fallback")
            else:
                print(f"[{db_id} | {db_type}]: Mongo WriteError: {we}")
                continue

    client.close()
    print("Done.")


if __name__ == "__main__":
    main() 