#!/usr/bin/env python3
"""
Schema initialization: Extract schema from database and build functional dependency graph.

This script:
1. Extracts schema information from SQLite database
2. Generates table and column meanings using proper prompts from schema_enricher
3. Predicts missing primary keys and foreign keys using proper prompts from schema_enricher
4. Builds functional dependency graph using existing build_graph function
5. Saves the graph to a pickle file

Usage:
    python init_schema.py \
        --db-path /path/to/database.sqlite \
        --output graph.pkl \
        --model gpt-4.1-mini
"""

import argparse
import sqlite3
import pickle
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

from modules.schema_enricher.missing_keys_prediction import (
    predict_missing_keys_for_schema,
)
from modules.schema_enricher.column_meaning_generation import (
    generate_column_meanings_for_schema,
)
from modules.schema_enricher.table_meaning_generation import (
    generate_table_meanings_for_schema,
)
from modules.schema_enricher.build_FD_graph import build_graph

load_dotenv()


def extract_schema_from_sqlite(db_path: Path) -> Dict[str, Any]:
    """Extract schema information from SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    schema_info = {
        "schema": [],
        "column_info": {},
        "column_meaning": {},
        "table_meaning": {},
        "primary_keys": {},
        "foreign_keys": {},
        "generated_primary_keys": {},
        "generated_foreign_keys": {},
    }
    
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    for table in tables:
        cursor.execute(f"PRAGMA table_info(`{table}`);")
        columns_info = cursor.fetchall()
        
        pk_cols = []
        for col_info in columns_info:
            col_name = col_info[1]
            col_type = col_info[2]
            is_pk = col_info[5] > 0
            
            full_col_name = f"{table}.{col_name}"
            schema_info["schema"].append(full_col_name)
            
            if is_pk:
                pk_cols.append(col_name)
            
            try:
                cursor.execute(f"SELECT DISTINCT `{col_name}` FROM `{table}` LIMIT 5;")
                sample_values = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]
            except:
                sample_values = []
            
            schema_info["column_info"][full_col_name] = {
                "type": col_type,
                "similar_values": sample_values[:3],
                "has_null": False,
            }
        
        if pk_cols:
            schema_info["primary_keys"][table] = pk_cols
    
    for table in tables:
        try:
            cursor.execute(f"PRAGMA foreign_key_list(`{table}`);")
            fk_list = cursor.fetchall()
            
            if fk_list:
                schema_info["foreign_keys"][table] = []
                for fk_info in fk_list:
                    schema_info["foreign_keys"][table].append({
                        "from": fk_info[3],
                        "to": f"{fk_info[2]}.{fk_info[4]}",
                        "ref_table": fk_info[2],
                        "on_update": "NO ACTION",
                        "on_delete": "NO ACTION",
                        "match": "SIMPLE",
                    })
        except:
            pass
    
    conn.close()
    return schema_info




def main():
    parser = argparse.ArgumentParser(description="Initialize schema graph from SQLite database")
    parser.add_argument("--db-path", type=Path, required=True,
                       help="Path to SQLite database file")
    parser.add_argument("--output", type=Path, default=Path("schema_graph.pkl"),
                       help="Output path for the graph pickle file")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                       help="OpenAI model to use (default: gpt-4.1-mini)")
    parser.add_argument("--db-id", type=str, default="",
                       help="Database ID for prompts (default: extracted from db-path)")
    parser.add_argument("--evidence", type=str, default="",
                       help="External knowledge/evidence to include in prompts")
    args = parser.parse_args()
    
    db_id = args.db_id or args.db_path.stem
    
    print("=" * 80)
    print("GRAST-SQL Schema Initialization")
    print("=" * 80)
    print()
    
    print(f"Step 1: Extracting schema from {args.db_path}")
    schema_info = extract_schema_from_sqlite(args.db_path)
    print(f"  Found {len(set(col.split('.')[0] for col in schema_info['schema']))} tables")
    print(f"  Found {len(schema_info['schema'])} columns")
    print(f"  Found {len(schema_info['primary_keys'])} tables with primary keys")
    print(f"  Found {len(schema_info['foreign_keys'])} tables with foreign keys")
    print()
    
    print(f"Step 2: Predicting missing keys with OpenAI (model: {args.model})")
    predict_missing_keys_for_schema(schema_info, args.model, db_id)
    print(f"  Predicted {sum(len(v) for v in schema_info['generated_primary_keys'].values())} additional primary keys")
    print(f"  Predicted {sum(len(v) for v in schema_info['generated_foreign_keys'].values())} additional foreign keys")
    print()
    
    print(f"Step 3: Generating meanings with OpenAI")
    print("  3a: Generating column meanings...")
    generate_column_meanings_for_schema(schema_info, args.model, db_id, args.evidence)
    print(f"    Generated {len(schema_info['column_meaning'])} column meanings")
    print("  3b: Generating table meanings...")
    generate_table_meanings_for_schema(schema_info, args.model, db_id, args.evidence)
    print(f"    Generated {len(schema_info['table_meaning'])} table meanings")
    print()
    
    print("Step 4: Building functional dependency graph")
    graph = build_graph(schema_info)
    print(f"  Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    print()
    
    print(f"Step 5: Saving graph to {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(graph, f)
    print(f"  ✓ Graph saved successfully")
    print()
    
    print("=" * 80)
    print("Schema initialization completed!")
    print(f"Graph saved to: {args.output.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
