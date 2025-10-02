#!/usr/bin/env python3
"""
1_1_create_data_from_graph.py  –  Create training data from graph pickle
• Reads data/<dataset>_<split>_samples_graph.pkl created by 1_build_graph.py
• Filters out columns that are primary keys or foreign keys
• Falls back to primary keys if no eligible non-PK/FK columns are used
• Creates JSON-Lines format for training
• Skips samples without positive contexts OR negative contexts

Output JSONL (identical record schema):
{
  "query":  "<question>\\nHint: <evidence>",
  "pos":    ["<node_desc>", …],
  "neg":    ["<node_desc>", …],
  …
}
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


# ─────────────────────────── CLI ──────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create training data from graph pickle.")
    p.add_argument("split", choices=["train", "dev"],
                   help="Dataset split to export (train | dev).")
    p.add_argument("--dataset", choices=["bird", "spider"], default="bird",
                   help="Dataset to use: BIRD or Spider (default: bird)")
    p.add_argument("--filter-pkfk", action="store_true", default=False,
                   help="If set, filter out primary key and foreign key columns (default: do not filter)")
    p.add_argument("--multi-positive", action="store_true", default=False,
                   help="If set, create multiple training examples per query, each with one gold column as positive and rest as negatives")
    return p.parse_args()


args = parse_args()
SPLIT = args.split
DATASET = args.dataset
FILTER_PKFK = args.filter_pkfk
MULTI_POSITIVE = args.multi_positive

# Input/Output paths
# GRAPH_PKL = Path(f"../data/{DATASET}_{SPLIT}_samples_graph.pkl")
GRAPH_PKL = Path(f"../data/{DATASET}_{SPLIT}_samples_graph_merged.pkl")
print(GRAPH_PKL)

# Determine output filename based on options
if MULTI_POSITIVE:
    if FILTER_PKFK:
        OUT_FILE = Path(f"data/{DATASET}_{SPLIT}_sts_multi_pos_no_pk_fk.jsonl")
    else:
        OUT_FILE = Path(f"data/{DATASET}_{SPLIT}_sts_multi_pos_all_cols.jsonl")
else:
    if FILTER_PKFK:
        OUT_FILE = Path(f"data/{DATASET}_{SPLIT}_sts_no_pk_fk.jsonl")
    else:
        OUT_FILE = Path(f"data/{DATASET}_{SPLIT}_sts_all_cols.jsonl")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Determine figure filename based on options
if MULTI_POSITIVE:
    if FILTER_PKFK:
        FIG_PATH = FIG_DIR / f"query_passage_token_hist_multi_pos_no_pk_fk_{DATASET}_{SPLIT}.png"
    else:
        FIG_PATH = FIG_DIR / f"query_passage_token_hist_multi_pos_all_cols_{DATASET}_{SPLIT}.png"
else:
    if FILTER_PKFK:
        FIG_PATH = FIG_DIR / f"query_passage_token_hist_no_pk_fk_{DATASET}_{SPLIT}.png"
    else:
        FIG_PATH = FIG_DIR / f"query_passage_token_hist_all_cols_{DATASET}_{SPLIT}.png"

# ───────────────────── Tokenizer ─────────────────────
TOKENIZER = AutoTokenizer.from_pretrained(
    "/home/datht/huggingface/BAAI/bge-reranker-v2-minicpm-layerwise",
    use_fast=True
)

# ───────────── Helper: build node description ──────────
# def make_desc_from_graph(G, col_name: str) -> str:
#     """Create description from graph node attributes."""
#     node_data = G.nodes[col_name]
    
#     table, column = col_name.split(".")
    
#     # make column has values like this Values: "Continuation School", "Opportunity School", make sure it has the quotes
#     similar_values = node_data.get('similar_values', [])
#     similar_values_str = ", ".join(f'"{value}"' for value in similar_values)
    
#     parts = [
#         f"Column: {table}.{column}",
#         f"Column meaning: {node_data.get('meaning', '')}",
#         f"Column type: {node_data.get('type', '')}",
#         f"Column has values: {similar_values_str}", 
#         f"Column has null values: {node_data.get('has_null', False)}",
#     ]
    
#     value_desc = node_data.get("value_desc", "")
#     if value_desc and str(value_desc).strip():
#         parts.append(f"Value description: {value_desc.strip()}")
    
#     return " ; ".join(parts)


def make_desc(node) -> str:
    """Create description string for a node."""
    col = node.get("node_name", "")
    # meaning = concat(table_meaning, column_meaning)
    table_meaning = node.get("table_meaning", "")
    column_meaning = node.get("meaning", "")
    meaning = f"Table meaning: {table_meaning}. Column meaning: {column_meaning}"
    col_type = node.get("type", "")
    # vals = " , ".join(map(str, node.get("similar_values", [])))
    vals = " , ".join([f'"{value}"' for value in node.get("similar_values", [])])
    has_null = node.get("has_null", False)
    val_desc = node.get("value_desc", "")
    table, column = col.split(".")
    parts = [f"{table}.{column}", meaning,
             f"type {col_type}", f"has values {vals}",
             f"has_null = {has_null}"]
    if val_desc.strip():
        parts.append(f"Value description: {val_desc.strip()}")
    return " ; ".join(parts)

def make_desc_from_graph(G, col_name: str) -> str:
    """Create description from graph node attributes."""
    node_data = G.nodes[col_name]
    return make_desc(node_data)

# ───────── Helper: build a JSONL record ─────────
def build_record_from_graph(question: str, G, used_columns: List[str], filter_pkfk: bool) -> dict | None:
    """Build training record from graph, optionally excluding PK/FK columns, with fallback to PK."""
    if filter_pkfk:
        # Get all columns that are NOT primary keys or foreign keys
        eligible_cols = []
        pk_cols = []
        for col_name in G.nodes():
            node_data = G.nodes[col_name]
            is_pk = node_data.get("is_primary_key", False)
            is_fk = node_data.get("is_in_foreign_key", False)
            if not is_pk and not is_fk:
                eligible_cols.append(col_name)
            elif is_pk and not is_fk:  # PK but not FK
                pk_cols.append(col_name)
        # Filter used columns to only include eligible ones
        used_eligible = [col for col in used_columns if col in eligible_cols]
        # If no eligible used columns, fall back to primary keys
        if not used_eligible:
            used_pk = [col for col in used_columns if col in pk_cols]
            if not used_pk:  # Still no positive samples, skip
                return None
            # Use PK columns as positive, only eligible columns as negative
            used_set = set(used_pk)
            neg_cols = [c for c in eligible_cols if c not in used_set]
            # Check if we have both positive and negative contexts
            if not neg_cols:  # No negative contexts available
                return None
            pos_desc = [make_desc_from_graph(G, c) for c in used_pk]
            neg_desc = [make_desc_from_graph(G, c) for c in neg_cols]
        else:
            # Use eligible columns as positive, only eligible columns as negative
            used_set = set(used_eligible)
            neg_cols = [c for c in eligible_cols if c not in used_set]
            # Check if we have both positive and negative contexts
            if not neg_cols:  # No negative contexts available
                return None
            pos_desc = [make_desc_from_graph(G, c) for c in used_eligible]
            neg_desc = [make_desc_from_graph(G, c) for c in neg_cols]
    else:
        # All columns are eligible
        eligible_cols = list(G.nodes())
        used_eligible = [col for col in used_columns if col in eligible_cols]
        if not used_eligible:
            return None
        used_set = set(used_eligible)
        neg_cols = [c for c in eligible_cols if c not in used_set]
        if not neg_cols:
            return None
        pos_desc = [make_desc_from_graph(G, c) for c in used_eligible]
        neg_desc = [make_desc_from_graph(G, c) for c in neg_cols]
    return {
        "query":       question,
        "pos":         pos_desc,
        "neg":         neg_desc,
        "pos_scores":  [1.0] * len(pos_desc),
        "neg_scores":  [0.0] * len(neg_desc),
        "prompt":      "Retrieve columns used for writing SQL for given question.",
        "type":        SPLIT,
    }


def build_multi_positive_records_from_graph(question: str, G, used_columns: List[str], filter_pkfk: bool) -> List[dict]:
    """Build multiple training records from graph, each with one gold column as positive and rest as negatives."""
    records = []
    
    # Get eligible columns based on filter_pkfk
    if filter_pkfk:
        # Get all columns that are NOT primary keys or foreign keys
        eligible_cols = []
        pk_cols = []
        for col_name in G.nodes():
            node_data = G.nodes[col_name]
            is_pk = node_data.get("is_primary_key", False)
            is_fk = node_data.get("is_in_foreign_key", False)
            if not is_pk and not is_fk:
                eligible_cols.append(col_name)
            elif is_pk and not is_fk:  # PK but not FK
                pk_cols.append(col_name)
        
        # Filter used columns to only include eligible ones
        used_eligible = [col for col in used_columns if col in eligible_cols]
        # If no eligible used columns, fall back to primary keys
        if not used_eligible:
            used_eligible = [col for col in used_columns if col in pk_cols]
        
        if not used_eligible:
            return records  # No positive samples, return empty list
        
        # Use eligible columns for both positive and negative
        all_eligible = eligible_cols + pk_cols
    else:
        # All columns are eligible
        all_eligible = list(G.nodes())
        used_eligible = [col for col in used_columns if col in all_eligible]
        if not used_eligible:
            return records  # No positive samples, return empty list
    
    # Create one record for each gold column
    for pos_col in used_eligible:
        # Positive: just this column
        pos_desc = [make_desc_from_graph(G, pos_col)]
        
        # Negative: all other eligible columns (including other gold columns)
        neg_cols = [c for c in all_eligible if c != pos_col]
        if not neg_cols:  # Need at least one negative example
            continue
            
        neg_desc = [make_desc_from_graph(G, c) for c in neg_cols]
        
        record = {
            "query":       question,
            "pos":         pos_desc,
            "neg":         neg_desc,
            "pos_scores":  [1.0],
            "neg_scores":  [0.0] * len(neg_desc),
            "prompt":      "Retrieve columns used for writing SQL for given question.",
            "type":        SPLIT,
        }
        records.append(record)
    
    return records

# ──────────────────────────── main ────────────────────────────
def main() -> None:
    # Load graph data
    print(f"Loading graph data from {GRAPH_PKL}")
    print(f"Dataset: {DATASET.upper()}, Split: {SPLIT}, Filter PK/FK: {FILTER_PKFK}, Multi-positive: {MULTI_POSITIVE}")
    
    with open(GRAPH_PKL, "rb") as f:
        triples = pickle.load(f)
    
    print(f"Loaded {len(triples)} samples")
    
    query_lens: List[int] = []
    passage_lens: List[int] = []
    written = 0
    skipped = 0
    
    with OUT_FILE.open("w", encoding="utf-8") as fh:
        for question, G, used_columns, true_sql in triples:
            if MULTI_POSITIVE:
                # Create multiple records, each with one gold column as positive
                records = build_multi_positive_records_from_graph(question, G, used_columns, FILTER_PKFK)
                
                if not records:
                    skipped += 1
                    continue
                
                # Write all records for this query
                for rec in records:
                    # ---------- token statistics ----------
                    q_tokens = TOKENIZER.encode(rec["query"],
                                                add_special_tokens=False)
                    query_lens.append(len(q_tokens))
                    
                    for passage in rec["pos"] + rec["neg"]:
                        p_tokens = TOKENIZER.encode(passage,
                                                    add_special_tokens=False)
                        passage_lens.append(len(p_tokens))
                    
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
            else:
                # Original single record approach
                rec = build_record_from_graph(question, G, used_columns, FILTER_PKFK)
                
                # Skip if no eligible used columns or no negative contexts
                if rec is None:
                    skipped += 1
                    continue
                
                # ---------- token statistics ----------
                q_tokens = TOKENIZER.encode(rec["query"],
                                            add_special_tokens=False)
                query_lens.append(len(q_tokens))
                
                for passage in rec["pos"] + rec["neg"]:
                    p_tokens = TOKENIZER.encode(passage,
                                                add_special_tokens=False)
                    passage_lens.append(len(p_tokens))
                
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    
    # ─────────── stats & histogram ───────────
    q_arr = np.array(query_lens)
    p_arr = np.array(passage_lens)
    
    print("✅  Wrote {:,} records → {}".format(written, OUT_FILE))
    print("❌  Skipped {:,} records (no eligible used columns or no negative contexts)".format(skipped))
    print("▸ queries :", f"mean={q_arr.mean():.1f}",
          f"min={q_arr.min()}", f"max={q_arr.max()}")
    print("▸ passages:", f"mean={p_arr.mean():.1f}",
          f"min={p_arr.min()}", f"max={p_arr.max()}")
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(q_arr, bins=50)
    plt.title(f"#tokens in queries ({DATASET} {SPLIT})")
    plt.xlabel("tokens")
    plt.ylabel("count")
    
    plt.subplot(1, 2, 2)
    plt.hist(p_arr, bins=50)
    plt.title(f"#tokens in passages ({DATASET} {SPLIT})")
    plt.xlabel("tokens")
    
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    plt.close()
    print(f"📊  Histogram saved → {FIG_PATH}")

if __name__ == "__main__":
    main() 