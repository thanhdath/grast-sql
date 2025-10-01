#!/usr/bin/env python3
"""
Export table mappings for all databases in Spider2.0-lite dataset.

This script processes all databases and exports their table name mappings
to wildcard patterns in a table_mapping directory.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from pymongo import MongoClient
from collections import defaultdict

# Special handling for ga360 database


def get_table_schema_from_json(db_id: str, db_type: str, base_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Get table schemas from JSON files for BigQuery/Snowflake databases."""
    if db_type not in ["bigquery", "snowflake"]:
        return {}
    
    db_path = base_dir / "resource" / "databases" / db_type / db_id
    if not db_path.exists():
        return {}
    
    table_schemas = {}
    
    # Read all JSON files recursively in the database directory and subdirectories
    for json_file in db_path.rglob("*.json"):
        # Get table name from filename (without extension)
        table_name = json_file.stem
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
                
                # Handle different JSON schema formats
                if isinstance(schema_data, list):
                    # Direct list of column objects
                    table_schemas[table_name] = schema_data
                elif isinstance(schema_data, dict):
                    # Dictionary with column information
                    if 'columns' in schema_data:
                        table_schemas[table_name] = schema_data['columns']
                    elif 'column_names' in schema_data:
                        # Convert column names to column objects
                        columns = []
                        for col_name in schema_data['column_names']:
                            columns.append({'name': col_name, 'type': 'STRING'})  # Default type
                        table_schemas[table_name] = columns
                    else:
                        # Assume it's a column object itself
                        table_schemas[table_name] = [schema_data]
                else:
                    print(f"Warning: Unexpected schema format in {json_file}")
                    
        except Exception as e:
            print(f"Warning: Could not read {json_file}: {e}")
    
    return table_schemas


def get_table_schema_from_sqlite(db_id: str, base_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Get table schemas from SQLite database."""
    import sqlite3
    
    # Get original database name for SQLite file path
    original_db_name = db_id  # Simplified for now
    db_path = base_dir / "resource" / "databases" / "spider2-localdb" / f"{original_db_name}.sqlite"
    
    if not db_path.exists():
        return {}
    
    table_schemas = {}
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table_name in tables:
            # Get table info
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            columns = []
            for row in cursor.fetchall():
                col_info = {
                    "name": row[1],
                    "type": row[2],
                    "notnull": row[3],
                    "default_value": row[4],
                    "primary_key": row[5]
                }
                columns.append(col_info)
            
            table_schemas[table_name] = columns
        
        conn.close()
    except Exception as e:
        print(f"Warning: Could not read SQLite database {db_path}: {e}")
    
    return table_schemas


def create_schema_signature(columns: List[Dict[str, Any]]) -> str:
    """Create a unique signature for a table schema based on column names and types."""
    # Sort columns by name for consistent signature
    sorted_cols = sorted(columns, key=lambda x: x.get('name', ''))
    
    # Create signature from column names and types
    signature_parts = []
    for col in sorted_cols:
        col_name = col.get('name', '')
        col_type = col.get('type', '')
        signature_parts.append(f"{col_name}:{col_type}")
    
    return "|".join(signature_parts)


def group_tables_by_schema(table_schemas: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
    """Group tables by their schema signature."""
    schema_groups = defaultdict(list)
    
    for table_name, columns in table_schemas.items():
        signature = create_schema_signature(columns)
        schema_groups[signature].append(table_name)
    
    return dict(schema_groups)


def create_wildcard_mapping(schema_groups: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Create wildcard mappings for tables with identical schemas.
    
    Rule: Only map tables that have the same base pattern but different suffixes.
    Wildcard should only replace the last part after the last underscore.
    IMPORTANT: Only group tables that have truly identical schemas (column names, types, meanings).
    """
    wildcard_mappings = {}
    
    for signature, table_names in schema_groups.items():
        if len(table_names) > 1:
            # Multiple tables with same schema - check if they follow a pattern
            sorted_names = sorted(table_names)
            
            # Find common prefix
            common_prefix = ""
            for i in range(min(len(name) for name in sorted_names)):
                if all(name[i] == sorted_names[0][i] for name in sorted_names):
                    common_prefix += sorted_names[0][i]
                else:
                    break
            
            # Only create wildcard if there's a clear pattern with different suffixes
            # Wildcard should only replace the last part after the last underscore
            if common_prefix:
                # Check if tables have same prefix but different suffixes
                suffixes = set()
                for table_name in sorted_names:
                    if table_name.startswith(common_prefix):
                        suffix = table_name[len(common_prefix):]
                        if suffix:  # Only if there's actually a suffix
                            suffixes.add(suffix)
                
                # Only create wildcard if there are multiple different suffixes
                if len(suffixes) > 1:
                    # Find the last underscore in the common prefix
                    last_underscore_pos = common_prefix.rfind('_')
                    if last_underscore_pos != -1:
                        # Split at the last underscore and keep everything before it
                        base_prefix = common_prefix[:last_underscore_pos + 1]  # Include the underscore
                        wildcard_pattern = f"{base_prefix}*"
                        for table_name in table_names:
                            wildcard_mappings[table_name] = wildcard_pattern
    
    return wildcard_mappings


def create_wildcard_mapping_strict(schema_groups: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Create wildcard mappings ONLY for tables with truly identical schemas.
    
    This function is more strict and only groups tables that:
    1. Have identical schemas (column names, types, meanings)
    2. Follow a naming pattern with different suffixes
    3. Have the same base prefix ending with underscore
    """
    wildcard_mappings = {}
    
    for signature, table_names in schema_groups.items():
        if len(table_names) > 1:
            # Multiple tables with same schema - check if they follow a pattern
            sorted_names = sorted(table_names)
            
            # Find common prefix
            common_prefix = ""
            for i in range(min(len(name) for name in sorted_names)):
                if all(name[i] == sorted_names[0][i] for name in sorted_names):
                    common_prefix += sorted_names[0][i]
                else:
                    break
            
            # Only create wildcard if there's a clear pattern with different suffixes
            if common_prefix:
                # Check if tables have same prefix but different suffixes
                suffixes = set()
                for table_name in sorted_names:
                    if table_name.startswith(common_prefix):
                        suffix = table_name[len(common_prefix):]
                        if suffix:  # Only if there's actually a suffix
                            suffixes.add(suffix)
                
                # Only create wildcard if there are multiple different suffixes
                if len(suffixes) > 1:
                    # Find the last underscore in the common prefix
                    last_underscore_pos = common_prefix.rfind('_')
                    if last_underscore_pos != -1:
                        # Split at the last underscore and keep everything before it
                        base_prefix = common_prefix[:last_underscore_pos + 1]  # Include the underscore
                        wildcard_pattern = f"{base_prefix}*"
                        for table_name in table_names:
                            wildcard_mappings[table_name] = wildcard_pattern
    
    return wildcard_mappings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export table mappings for all databases.")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides)")
    p.add_argument("--collection-name", default="spider2_lite_samples",
                   help="MongoDB collection name (default: %(default)s)")
    p.add_argument("--base-dir", type=Path, default=Path("/home/datht/Spider2/spider2-lite"),
                   help="Root directory containing Spider2.0-lite dataset")
    p.add_argument("--output-dir", type=Path, default=Path("table_mapping"),
                   help="Output directory for mapping files (default: %(default)s)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be exported without creating files")
    return p.parse_args()


def get_table_wildcard_mappings(db_id: str, db_type: str, base_dir: Path) -> Dict[str, str]:
    """
    Get table name mappings to wildcard patterns based on schema similarity.
    
    Rule: Only map tables that have the same base pattern but different suffixes.
    Wildcard should only be at the end, replacing the varying suffix part.
    IMPORTANT: Only group tables that have truly identical schemas.
    
    Returns:
        Dict mapping table names to their wildcard patterns
    """
    # Get table schemas based on database type
    if db_type == "sqlite":
        table_schemas = get_table_schema_from_sqlite(db_id, base_dir)
    else:
        table_schemas = get_table_schema_from_json(db_id, db_type, base_dir)
    
    if not table_schemas:
        return {}
    
    # Group tables by schema
    schema_groups = group_tables_by_schema(table_schemas)
    
    # Create wildcard mappings
    wildcard_mappings = create_wildcard_mapping(schema_groups)
    
    return wildcard_mappings


def get_table_wildcard_mappings_strict(db_id: str, db_type: str, base_dir: Path) -> Dict[str, str]:
    """
    Get table name mappings to wildcard patterns with strict schema verification.
    
    This function creates mappings and then filters out any inconsistent patterns
    to ensure only tables with truly identical schemas are grouped.
    
    Returns:
        Dict mapping table names to their wildcard patterns (only consistent ones)
    """
    # Get table schemas based on database type
    if db_type == "sqlite":
        table_schemas = get_table_schema_from_sqlite(db_id, base_dir)
    else:
        table_schemas = get_table_schema_from_json(db_id, db_type, base_dir)
    
    if not table_schemas:
        return {}
    
    # Group tables by schema
    schema_groups = group_tables_by_schema(table_schemas)
    
    # Create initial wildcard mappings
    initial_mappings = create_wildcard_mapping(schema_groups)
    
    if not initial_mappings:
        return {}
    
    # Group by wildcard patterns
    pattern_groups = {}
    for table_name, wildcard_pattern in initial_mappings.items():
        if wildcard_pattern not in pattern_groups:
            pattern_groups[wildcard_pattern] = []
        pattern_groups[wildcard_pattern].append(table_name)
    
    # Verify schema consistency and filter out inconsistent patterns
    strict_mappings = {}
    for wildcard_pattern, table_names in pattern_groups.items():
        if len(table_names) < 2:
            continue
            
        # Get schemas for all tables in this group
        group_schemas = {}
        for table_name in table_names:
            if table_name in table_schemas:
                group_schemas[table_name] = table_schemas[table_name]
        
        if len(group_schemas) < 2:
            continue
            
        # Check if all schemas are identical
        first_table = list(group_schemas.keys())[0]
        first_schema = group_schemas[first_table]
        first_signature = create_schema_signature(first_schema)
        
        all_identical = True
        for table_name, schema in group_schemas.items():
            if table_name == first_table:
                continue
                
            signature = create_schema_signature(schema)
            if signature != first_signature:
                all_identical = False
                break
        
        # Only keep mappings for patterns where all tables have identical schemas
        if all_identical:
            for table_name in table_names:
                strict_mappings[table_name] = wildcard_pattern
    
    return strict_mappings


def get_table_wildcard_mappings_ga360_special(db_id: str, db_type: str, base_dir: Path) -> Dict[str, str]:
    """
    Special handling for specific databases: create mappings for specific table patterns regardless of schema differences.
    
    This function creates wildcard mappings for:
    - ga360: all ga_sessions_* tables to ga_sessions_* pattern
    - covid19_usa, sdoh: zip_codes_*, zcta_*, and zcta5_* tables to their respective patterns
    - census_bureau_acs_1, CENSUS_BUREAU_ACS_2: comprehensive mapping for all census patterns with low variance
    
    Returns:
        Dict mapping table names to their wildcard patterns (special cases)
    """
    # Special handling for ga360: directly map all ga_sessions_* tables
    if db_id == "ga360":
        # Get all JSON files in the ga360 directory to find table names
        db_path = base_dir / "resource" / "databases" / db_type / db_id
        if not db_path.exists():
            return {}
        
        wildcard_mappings = {}
        # Find all ga_sessions_* JSON files
        for json_file in db_path.rglob("ga_sessions_*.json"):
            table_name = json_file.stem  # Get filename without extension
            wildcard_mappings[table_name] = "ga_sessions_*"
        
        if wildcard_mappings:
            print(f"  Special ga360 handling: mapping {len(wildcard_mappings)} ga_sessions_* tables to ga_sessions_* pattern")
        
        return wildcard_mappings
    
    # Special handling for census_bureau_acs_1 and CENSUS_BUREAU_ACS_2: comprehensive mapping with variance checking
    elif db_id in ["census_bureau_acs_1", "CENSUS_BUREAU_ACS_2"]:
        db_path = base_dir / "resource" / "databases" / db_type / db_id
        if not db_path.exists():
            return {}
        
        wildcard_mappings = {}
        
        # Define patterns to check with their expected column counts
        # Use uppercase patterns for CENSUS_BUREAU_ACS_2, lowercase for census_bureau_acs_1
        if db_id == "CENSUS_BUREAU_ACS_2":
            patterns_to_check = {
                "STATE_*": [],
                "COUNTY_*": [],
                "PLACE_*": [],
                "PUMA_*": [],
                "CONGRESSIONALDISTRICT_*": [],
                "CBSA_*": [],
                "ZIP_CODES_*": [],
                "ZCTA_*": [],
                "CENSUSTRACT_*": [],
                "SCHOOLDISTRICTELEMENTARY_*": [],
                "SCHOOLDISTRICTSECONDARY_*": [],
                "SCHOOLDISTRICTUNIFIED_*": []
            }
        else:
            patterns_to_check = {
                "state_*": [],
                "county_*": [],
                "place_*": [],
                "puma_*": [],
                "congressionaldistrict_*": [],
                "cbsa_*": [],
                "zip_codes_*": [],
                "zcta_*": [],
                "censustract_*": [],
                "schooldistrictelementary_*": [],
                "schooldistrictsecondary_*": [],
                "schooldistrictunified_*": []
            }
        
        # Collect all tables and their column counts
        table_column_counts = {}
        for json_file in db_path.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    schema_data = json.load(f)
                table_name = schema_data.get("table_name", "")
                column_names = schema_data.get("column_names", [])
                column_count = len(column_names)
                table_column_counts[table_name] = column_count
            except Exception as e:
                continue
        
        # For CENSUS_BUREAU_ACS_2, also check subdirectories
        if db_id == "CENSUS_BUREAU_ACS_2":
            # Check CENSUS_BUREAU_ACS subdirectory
            acs_path = db_path / "CENSUS_BUREAU_ACS"
            if acs_path.exists():
                print(f"  Found CENSUS_BUREAU_ACS subdirectory with {len(list(acs_path.glob('*.json')))} files")
                for json_file in acs_path.rglob("*.json"):
                    try:
                        with open(json_file, 'r') as f:
                            schema_data = json.load(f)
                        table_name = schema_data.get("table_name", "")
                        column_names = schema_data.get("column_names", [])
                        column_count = len(column_names)
                        table_column_counts[table_name] = column_count
                    except Exception as e:
                        continue
        
        # Group tables by pattern and check variance
        for pattern, tables in patterns_to_check.items():
            base_pattern = pattern.replace("*", "")
            pattern_tables = []
            
            for table_name, column_count in table_column_counts.items():
                if table_name.startswith(base_pattern):
                    pattern_tables.append((table_name, column_count))
            
            if len(pattern_tables) >= 2:
                # Check variance in column counts
                column_counts = [count for _, count in pattern_tables]
                min_count = min(column_counts)
                max_count = max(column_counts)
                variance = (max_count - min_count) / min_count if min_count > 0 else 0
                
                # Only include patterns with low variance (< 10%)
                if variance < 0.1:
                    for table_name, _ in pattern_tables:
                        wildcard_mappings[table_name] = pattern
                    
                    print(f"  Special census_bureau_acs_1 handling: {pattern} - {len(pattern_tables)} tables, variance: {variance:.2%}")
        
        # Add existing patterns from the original mapping
        existing_patterns = {
            "census_tracts_*": [],
            "blockgroup_*": [],
            "congress_district_*": []
        }
        
        for pattern, tables in existing_patterns.items():
            base_pattern = pattern.replace("*", "")
            pattern_tables = []
            
            for table_name, column_count in table_column_counts.items():
                if table_name.startswith(base_pattern):
                    pattern_tables.append((table_name, column_count))
            
            if pattern_tables:
                for table_name, _ in pattern_tables:
                    wildcard_mappings[table_name] = pattern
        
        if wildcard_mappings:
            total_mapped = len(wildcard_mappings)
            print(f"  Special {db_id} handling: mapping {total_mapped} tables total")
        
        return wildcard_mappings
    
    # Special handling for covid19_usa and sdoh: map census bureau tables
    elif db_id in ["covid19_usa", "sdoh"]:
        db_path = base_dir / "resource" / "databases" / db_type / db_id
        if not db_path.exists():
            return {}
        
        wildcard_mappings = {}
        
        # Map zip_codes_* tables
        zip_files = list(db_path.rglob("zip_codes_*.json"))
        for json_file in zip_files:
            table_name = json_file.stem
            wildcard_mappings[table_name] = "zip_codes_*"
        
        # Map zcta_* tables (excluding zcta5_*)
        zcta_files = list(db_path.rglob("zcta_*.json"))
        for json_file in zcta_files:
            table_name = json_file.stem
            if not table_name.startswith("zcta5_"):  # Exclude zcta5_* tables
                wildcard_mappings[table_name] = "zcta_*"
        
        # Map zcta5_* tables
        zcta5_files = list(db_path.rglob("zcta5_*.json"))
        for json_file in zcta5_files:
            table_name = json_file.stem
            wildcard_mappings[table_name] = "zcta5_*"
        
        # Map additional major patterns for covid19_usa
        major_patterns = [
            "blockgroup_*",
            "cbsa_*", 
            "censustract_*",
            "congressionaldistrict_*",
            "county_*",
            "place_*",
            "puma_*",
            "schooldistrictelementary_*",
            "schooldistrictsecondary_*",
            "schooldistrictunified_*",
            "state_*"
        ]
        
        for pattern in major_patterns:
            base_pattern = pattern.replace("*", "")
            pattern_files = list(db_path.rglob(f"{base_pattern}*.json"))
            for json_file in pattern_files:
                table_name = json_file.stem
                if table_name not in wildcard_mappings:  # Don't overwrite existing mappings
                    wildcard_mappings[table_name] = pattern
        
        if wildcard_mappings:
            zip_count = len([f for f in zip_files])
            zcta_count = len([f for f in zcta_files if not Path(f).stem.startswith("zcta5_")])
            zcta5_count = len(zcta5_files)
            total_mapped = len(wildcard_mappings)
            print(f"  Special {db_id} handling: mapping {total_mapped} tables total")
            print(f"    - {zip_count} zip_codes_* tables")
            print(f"    - {zcta_count} zcta_* tables") 
            print(f"    - {zcta5_count} zcta5_* tables")
            print(f"    - {total_mapped - zip_count - zcta_count - zcta5_count} other pattern tables")
        
        return wildcard_mappings
    
    # For other databases, use the normal strict approach
    return get_table_wildcard_mappings_100_percent_strict(db_id, db_type, base_dir)


def get_table_wildcard_mappings_100_percent_strict(db_id: str, db_type: str, base_dir: Path) -> Dict[str, str]:
    """
    Get table name mappings to wildcard patterns with 100% strict schema verification.
    
    This function ensures that ONLY tables with truly identical schemas are grouped.
    It double-checks each group to guarantee 100% consistency.
    
    Returns:
        Dict mapping table names to their wildcard patterns (100% consistent)
    """
    # Get table schemas based on database type
    if db_type == "sqlite":
        table_schemas = get_table_schema_from_sqlite(db_id, base_dir)
    else:
        table_schemas = get_table_schema_from_json(db_id, db_type, base_dir)
    
    if not table_schemas:
        return {}
    
    # Group tables by schema
    schema_groups = group_tables_by_schema(table_schemas)
    
    # Create initial wildcard mappings
    initial_mappings = create_wildcard_mapping(schema_groups)
    
    if not initial_mappings:
        return {}
    
    # Group by wildcard patterns
    pattern_groups = {}
    for table_name, wildcard_pattern in initial_mappings.items():
        if wildcard_pattern not in pattern_groups:
            pattern_groups[wildcard_pattern] = []
        pattern_groups[wildcard_pattern].append(table_name)
    
    # Verify schema consistency and filter out inconsistent patterns
    strict_mappings = {}
    for wildcard_pattern, table_names in pattern_groups.items():
        if len(table_names) < 2:
            continue
            
        # Get schemas for all tables in this group
        group_schemas = {}
        for table_name in table_names:
            if table_name in table_schemas:
                group_schemas[table_name] = table_schemas[table_name]
        
        if len(group_schemas) < 2:
            continue
            
        # Double-check: Verify that all tables in this group have identical schemas
        first_table = list(group_schemas.keys())[0]
        first_schema = group_schemas[first_table]
        first_signature = create_schema_signature(first_schema)
        
        all_identical = True
        inconsistent_tables = []
        
        for table_name, schema in group_schemas.items():
            if table_name == first_table:
                continue
                
            signature = create_schema_signature(schema)
            if signature != first_signature:
                all_identical = False
                inconsistent_tables.append(table_name)
        
        # Only keep mappings for patterns where ALL tables have identical schemas
        if all_identical:
            for table_name in table_names:
                strict_mappings[table_name] = wildcard_pattern
        else:
            # Debug: Print inconsistent tables for this pattern
            print(f"DEBUG: Pattern {wildcard_pattern} has inconsistent schemas:")
            print(f"  First table ({first_table}) signature: {first_signature}")
            for table_name in inconsistent_tables[:3]:  # Show only first 3 for brevity
                schema = group_schemas[table_name]
                signature = create_schema_signature(schema)
                print(f"  Inconsistent table ({table_name}) signature: {signature}")
            if len(inconsistent_tables) > 3:
                print(f"  ... and {len(inconsistent_tables) - 3} more inconsistent tables")
            # IMPORTANT: Do NOT add any mappings for this pattern
    
    return strict_mappings


def verify_schema_consistency(db_id: str, db_type: str, base_dir: Path, pattern_groups: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """
    Verify that tables within each wildcard pattern group have identical schemas.
    
    Returns:
        Dict with verification results for each pattern
    """
    verification_results = {}
    
    # Get table schemas
    if db_type == "sqlite":
        table_schemas = get_table_schema_from_sqlite(db_id, base_dir)
    else:
        table_schemas = get_table_schema_from_json(db_id, db_type, base_dir)
    
    if not table_schemas:
        return verification_results
    
    for wildcard_pattern, table_names in pattern_groups.items():
        if len(table_names) < 2:
            continue  # Skip single tables
            
        # Get schemas for all tables in this group
        group_schemas = {}
        for table_name in table_names:
            if table_name in table_schemas:
                group_schemas[table_name] = table_schemas[table_name]
        
        if len(group_schemas) < 2:
            continue
            
        # Check if all schemas are identical
        first_table = list(group_schemas.keys())[0]
        first_schema = group_schemas[first_table]
        
        # Create schema signature for comparison
        first_signature = create_schema_signature(first_schema)
        
        all_identical = True
        inconsistent_tables = []
        detailed_analysis = {}
        
        # Get column names from first schema
        first_columns = set()
        if isinstance(first_schema, list):
            for col in first_schema:
                if isinstance(col, dict) and 'name' in col:
                    first_columns.add(col['name'])
                elif isinstance(col, str):
                    first_columns.add(col)
        
        for table_name, schema in group_schemas.items():
            if table_name == first_table:
                continue
                
            signature = create_schema_signature(schema)
            if signature != first_signature:
                all_identical = False
                inconsistent_tables.append(table_name)
                
                # Detailed analysis for this inconsistent table
                current_columns = set()
                if isinstance(schema, list):
                    for col in schema:
                        if isinstance(col, dict) and 'name' in col:
                            current_columns.add(col['name'])
                        elif isinstance(col, str):
                            current_columns.add(col)
                
                # Find differences
                extra_columns = current_columns - first_columns
                missing_columns = first_columns - current_columns
                
                detailed_analysis[table_name] = {
                    "column_count": len(current_columns),
                    "baseline_column_count": len(first_columns),
                    "extra_columns": list(extra_columns),
                    "missing_columns": list(missing_columns),
                    "total_differences": len(extra_columns) + len(missing_columns)
                }
        
        verification_results[wildcard_pattern] = {
            "total_tables": len(group_schemas),
            "all_identical": all_identical,
            "inconsistent_tables": inconsistent_tables,
            "sample_schema": first_schema[:5] if first_schema else [],  # First 5 columns as sample
            "baseline_column_count": len(first_columns),
            "detailed_analysis": detailed_analysis
        }
    
    return verification_results


def analyze_semantic_differences(extra_columns: List[str], missing_columns: List[str]) -> str:
    """
    Analyze semantic differences between column sets to provide meaningful insights.
    
    Args:
        extra_columns: List of extra columns in the current table
        missing_columns: List of missing columns in the current table
        
    Returns:
        String describing the semantic differences
    """
    if not extra_columns and not missing_columns:
        return ""
    
    insights = []
    
    # Analyze naming pattern differences
    male_age_patterns = []
    for col in extra_columns + missing_columns:
        if 'male_60' in col or 'male_62' in col:
            male_age_patterns.append(col)
    
    if male_age_patterns:
        if any('male_60_to_61' in col for col in male_age_patterns) and any('male_60_61' in col for col in male_age_patterns):
            insights.append("Column naming variation: 'male_60_61' vs 'male_60_to_61' pattern")
        if any('male_62_to_64' in col for col in male_age_patterns) and any('male_62_64' in col for col in male_age_patterns):
            insights.append("Column naming variation: 'male_62_64' vs 'male_62_to_64' pattern")
    
    # Analyze marital status columns
    marital_columns = [col for col in extra_columns + missing_columns if any(word in col.lower() for word in ['married', 'divorced', 'widowed', 'separated'])]
    if marital_columns:
        insights.append("Marital status data availability varies")
    
    # Analyze labor force columns
    labor_columns = [col for col in extra_columns + missing_columns if any(word in col.lower() for word in ['labor', 'employed', 'unemployed', 'armed_forces'])]
    if labor_columns:
        insights.append("Labor force data availability varies")
    
    # Analyze commute columns
    commute_columns = [col for col in extra_columns + missing_columns if 'commute' in col.lower()]
    if commute_columns:
        insights.append("Commute time data availability varies")
    
    # Analyze demographic columns
    demo_columns = [col for col in extra_columns + missing_columns if any(word in col.lower() for word in ['hispanic', 'asian', 'black', 'amerindian'])]
    if demo_columns:
        insights.append("Demographic breakdown availability varies")
    
    # Analyze education columns
    edu_columns = [col for col in extra_columns + missing_columns if any(word in col.lower() for word in ['degree', 'diploma', 'education'])]
    if edu_columns:
        insights.append("Education data availability varies")
    
    # Analyze geographic columns
    geo_columns = [col for col in extra_columns + missing_columns if any(word in col.lower() for word in ['geoid', 'geo_id'])]
    if geo_columns:
        insights.append("Geographic identifier format varies")
    
    # Analyze year-based patterns
    if len(extra_columns) > 0 and len(missing_columns) > 0:
        insights.append(f"Schema evolution: {len(extra_columns)} new columns, {len(missing_columns)} removed columns")
    
    return "; ".join(insights) if insights else "Column structure differences detected"


def export_mappings_to_json(db_id: str, db_type: str, base_dir: Path, output_file: Path, dry_run: bool = False, strict: bool = True) -> Dict[str, Any]:
    """Export table mappings to a JSON file."""
    if strict:
        # Use special handling for ga360, normal strict handling for others
        mappings = get_table_wildcard_mappings_ga360_special(db_id, db_type, base_dir)
    else:
        mappings = get_table_wildcard_mappings(db_id, db_type, base_dir)
    
    if not mappings:
        return {}
    
    # Create a structured export with metadata
    export_data = {
        "database_id": db_id,
        "database_type": db_type,
        "total_tables": len(mappings),
        "mappings": mappings,
        "wildcard_patterns": {},
        "strict_mode": strict
    }
    
    # Group by wildcard patterns for easier lookup
    pattern_groups = {}
    for table_name, wildcard_pattern in mappings.items():
        if wildcard_pattern not in pattern_groups:
            pattern_groups[wildcard_pattern] = []
        pattern_groups[wildcard_pattern].append(table_name)
    
    export_data["wildcard_patterns"] = pattern_groups
    
    # Verify schema consistency (should all be consistent in strict mode)
    verification_results = verify_schema_consistency(db_id, db_type, base_dir, pattern_groups)
    export_data["schema_verification"] = verification_results
    
    # Write to JSON file
    if not dry_run:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return export_data


def main() -> None:
    args = parse_args()
    
    # Create output directory
    if not args.dry_run:
        args.output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    
    # Connect to MongoDB
    client = MongoClient(args.mongo_uri)
    coll = client["mats"][args.collection_name]
    
    # Get all samples
    samples = list(coll.find({}))
    print(f"Processing {len(samples)} samples...")
    
    # Group samples by database
    db_groups = defaultdict(list)
    for sample in samples:
        db_id = sample.get("db_id")
        db_type = sample.get("db_type")
        if db_id and db_type:
            db_groups[(db_id, db_type)].append(sample)
    
    print(f"Found {len(db_groups)} unique databases")
    
    # Process each database
    exported_count = 0
    total_mappings = 0
    
    for (db_id, db_type), db_samples in db_groups.items():
        print(f"\nProcessing {db_id} ({db_type}) with {len(db_samples)} samples...")
        
        try:
            output_file = args.output_dir / f"{db_id}.json"
            export_data = export_mappings_to_json(db_id, db_type, args.base_dir, output_file, args.dry_run, strict=True)
            
            if export_data:
                total_tables = export_data["total_tables"]
                num_patterns = len(export_data["wildcard_patterns"])
                
                print(f"  Exported: {total_tables} tables mapped to {num_patterns} wildcard patterns")
                
                if not args.dry_run:
                    print(f"  File: {output_file}")
                
                # Show some examples
                if export_data["wildcard_patterns"]:
                    print("  Examples:")
                    for pattern, tables in list(export_data["wildcard_patterns"].items())[:3]:
                        print(f"    {pattern}: {len(tables)} tables")
                
                exported_count += 1
                total_mappings += total_tables
            else:
                print(f"  No mappings found")
                
        except Exception as e:
            print(f"  Error processing {db_id}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Databases processed: {len(db_groups)}")
    print(f"Databases with mappings: {exported_count}")
    print(f"Total tables mapped: {total_mappings}")
    
    if not args.dry_run:
        print(f"Output directory: {args.output_dir}")
        print(f"Mapping files created: {exported_count}")
    else:
        print("DRY RUN - No files created")
    
    # Schema verification summary
    print(f"\n{'='*60}")
    print("SCHEMA VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    total_patterns = 0
    consistent_patterns = 0
    inconsistent_patterns = 0
    
    for (db_id, db_type), db_samples in db_groups.items():
        try:
            # In dry-run mode, we need to regenerate the verification data
            if args.dry_run:
                # Re-run the export to get verification data
                temp_export_data = export_mappings_to_json(db_id, db_type, args.base_dir, Path("/tmp/temp.json"), dry_run=True, strict=True)
                verification = temp_export_data.get("schema_verification", {}) if temp_export_data else {}
            else:
                output_file = args.output_dir / f"{db_id}.json"
                if output_file.exists():
                    with open(output_file, 'r', encoding='utf-8') as f:
                        export_data = json.load(f)
                    verification = export_data.get("schema_verification", {})
                else:
                    verification = {}
            
            if verification:
                print(f"\n{db_id} ({db_type}):")
                for pattern, result in verification.items():
                    total_patterns += 1
                    if result["all_identical"]:
                        consistent_patterns += 1
                        print(f"  ✅ {pattern}: {result['total_tables']} tables - ALL IDENTICAL")
                    else:
                        inconsistent_patterns += 1
                        print(f"  ❌ {pattern}: {result['total_tables']} tables - INCONSISTENT")
                        print(f"     Inconsistent tables: {result['inconsistent_tables']}")
                        
                        # Show detailed analysis for inconsistent patterns
                        if "detailed_analysis" in result and result["detailed_analysis"]:
                            print(f"     Baseline schema: {result.get('baseline_column_count', 'N/A')} columns")
                            print(f"     Detailed differences:")
                            for table_name, analysis in result["detailed_analysis"].items():
                                print(f"       {table_name} ({analysis['column_count']} columns):")
                                # if analysis['extra_columns']:
                                #     print(f"         + Extra columns ({len(analysis['extra_columns'])}): {analysis['extra_columns'][:5]}{'...' if len(analysis['extra_columns']) > 5 else ''}")
                                # if analysis['missing_columns']:
                                #     print(f"         - Missing columns ({len(analysis['missing_columns'])}): {analysis['missing_columns'][:5]}{'...' if len(analysis['missing_columns']) > 5 else ''}")
                                # print(f"         Total differences: {analysis['total_differences']} columns")
                                
                                # Show semantic analysis for common differences
                                if analysis['extra_columns'] or analysis['missing_columns']:
                                    semantic_analysis = analyze_semantic_differences(analysis['extra_columns'], analysis['missing_columns'])
                                    if semantic_analysis:
                                        print(f"         Semantic analysis: {semantic_analysis}")
        except Exception as e:
            print(f"Error reading verification for {db_id}: {e}")
    
    print(f"\nVerification Results:")
    print(f"  Total wildcard patterns: {total_patterns}")
    print(f"  Consistent patterns: {consistent_patterns}")
    print(f"  Inconsistent patterns: {inconsistent_patterns}")
    print(f"  Consistency rate: {consistent_patterns/total_patterns*100:.1f}%" if total_patterns > 0 else "  No patterns to verify")
    
    client.close()


if __name__ == "__main__":
    main() 