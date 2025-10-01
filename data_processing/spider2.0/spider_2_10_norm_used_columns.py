#!/usr/bin/env python3
"""
spider_2_10_norm_used_columns.py
─────────────────────────────────
Script to analyze used_columns and identify normalization candidates.
Can backup used_columns to openai_used_columns and analyze what could be normalized.

Usage
-----
python spider_2_10_norm_used_columns.py                    # backup used_columns to openai_used_columns
python spider_2_10_norm_used_columns.py --skip-backup      # just show used_columns without backup
python spider_2_10_norm_used_columns.py --analyze-normalization --skip-backup  # show normalization candidates
python spider_2_10_norm_used_columns.py --analyze-normalization --normalize --skip-backup  # normalize and update MongoDB
python spider_2_10_norm_used_columns.py --show-missing --skip-backup  # show columns not in schema
python spider_2_10_norm_used_columns.py --fuzzy-match --skip-backup  # show fuzzy matches for missing columns
python spider_2_10_norm_used_columns.py --analyze-normalization --fuzzy-normalize --skip-backup  # apply fuzzy normalization
python spider_2_10_norm_used_columns.py --target-ids 5 171 172  # process specific document IDs
python spider_2_10_norm_used_columns.py --limit 10         # test with 10 samples
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from dotenv import load_dotenv
from pymongo import MongoClient


def analyze_normalization_candidates(used_columns: list, schema: list) -> dict:
    """
    Analyze which used_columns could be normalized by finding prefix matches in schema.
    
    Args:
        used_columns: List of columns from used_columns field
        schema: List of columns from schema field
        
    Returns:
        Dictionary with normalization analysis results
    """
    candidates = []
    schema_set = set(schema)
    
    # Check if any used_columns have "bigquery-public-data." prefix
    has_bigquery_prefix_in_used = any(col.startswith("bigquery-public-data.") for col in used_columns)
    
    for col in used_columns:
        if col in schema_set:
            # Column exists in schema, no normalization needed
            continue
        
        # Rule 2: Remove "bigquery-public-data." prefix if used_columns have it
        if has_bigquery_prefix_in_used and col.startswith("bigquery-public-data."):
            normalized_col = col.replace("bigquery-public-data.", "", 1)
            candidates.append({
                'original': col,
                'suggested': normalized_col,
                'reason': 'Remove "bigquery-public-data." prefix'
            })
            continue
        
        # Rule 3: Remove redundant prefixes by comparing with schema
        parts = col.split('.')
        if len(parts) >= 3:  # Need at least prefix.table.column
            # Check if the first prefix exists in schema
            first_prefix = parts[0]
            first_prefix_in_schema = any(col.startswith(f"{first_prefix}.") for col in schema)
            
            if not first_prefix_in_schema:
                # Only try to remove prefixes if the first prefix is NOT in schema
                for i in range(1, len(parts) - 1):  # Try removing prefixes one by one
                    test_col = '.'.join(parts[i:])  # Remove first i parts
                    # Double-check: ensure the test_col actually exists in schema
                    if test_col in schema_set:
                        candidates.append({
                            'original': col,
                            'suggested': test_col,
                            'reason': 'Remove redundant prefix (found in schema)'
                        })
                        break
            
        # Rule 1: Split by dots to find potential prefixes (nested column flattening)
        if len(parts) < 3:  # Need at least table.column.subcolumn for normalization
            continue
            
        # Try to find prefix matches for nested columns
        for i in range(2, len(parts)):  # Start from table.column
            prefix = '.'.join(parts[:i])
            if prefix in schema_set:
                candidates.append({
                    'original': col,
                    'suggested': prefix,
                    'reason': f'Prefix "{prefix}" found in schema'
                })
                break
    
    return {
        'total_used_columns': len(used_columns),
        'columns_in_schema': len([col for col in used_columns if col in schema_set]),
        'normalization_candidates': candidates,
        'candidates_count': len(candidates)
    }


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


def find_fuzzy_matches(used_columns: list, schema: list) -> dict:
    """
    Find fuzzy matches for columns that couldn't be normalized by exact rules.
    
    Args:
        used_columns: List of columns from used_columns field
        schema: List of columns from schema field
        
    Returns:
        Dictionary with fuzzy matching results
    """
    schema_set = set(schema)
    fuzzy_matches = []
    
    for col in used_columns:
        if col in schema_set:
            # Column exists in schema, no fuzzy matching needed
            continue
            
        # Calculate similarity scores with all schema columns
        best_matches = []
        for schema_col in schema:
            score = calculate_similarity_score(col, schema_col)
            if score > 0:  # Only include matches with some similarity
                best_matches.append({
                    'schema_column': schema_col,
                    'score': score,
                    'reason': get_similarity_reason(col, schema_col, score)
                })
        
        # Sort by score (highest first) and take top 3
        best_matches.sort(key=lambda x: x['score'], reverse=True)
        top_matches = best_matches[:3]
        
        if top_matches:
            fuzzy_matches.append({
                'original': col,
                'best_matches': top_matches
            })
    
    return {
        'total_used_columns': len(used_columns),
        'columns_in_schema': len([col for col in used_columns if col in schema_set]),
        'fuzzy_matches': fuzzy_matches,
        'fuzzy_matches_count': len(fuzzy_matches)
    }


def calculate_similarity_score(candidate: str, schema_col: str) -> int:
    """
    Calculate similarity score between candidate and schema column.
    
    Args:
        candidate: Column from used_columns
        schema_col: Column from schema
        
    Returns:
        Similarity score (0-100)
    """
    score = 0
    
    # Exact match
    if candidate == schema_col:
        return 100
    
    # Wildcard pattern matching
    if has_wildcard_pattern_match(candidate, schema_col):
        score += 70
    
    # Contains match (candidate contains schema_col or vice versa)
    if candidate in schema_col or schema_col in candidate:
        score += 80
    
    # Token overlap
    candidate_tokens = set(candidate.split('.'))
    schema_tokens = set(schema_col.split('.'))
    common_tokens = candidate_tokens.intersection(schema_tokens)
    score += len(common_tokens) * 20
    
    # Edit distance (for similar strings)
    if len(candidate) > 3 and len(schema_col) > 3:
        edit_distance = calculate_edit_distance(candidate, schema_col)
        max_length = max(len(candidate), len(schema_col))
        if edit_distance < max_length:
            score += max(0, 50 - (edit_distance * 5))
    
    return min(100, score)


def has_wildcard_pattern_match(candidate: str, schema_col: str) -> bool:
    """
    Check if candidate and schema_col have matching wildcard patterns.
    
    Args:
        candidate: Column from used_columns
        schema_col: Column from schema
        
    Returns:
        True if wildcard patterns match
    """
    # Replace wildcards with a placeholder for comparison
    candidate_clean = candidate.replace('*', 'WILDCARD')
    schema_clean = schema_col.replace('*', 'WILDCARD')
    
    # Check if they have the same structure
    if candidate_clean == schema_clean:
        return True
    
    # Check if one is a subset of the other
    candidate_parts = candidate.split('.')
    schema_parts = schema_col.split('.')
    
    if len(candidate_parts) == len(schema_parts):
        matches = 0
        for i in range(len(candidate_parts)):
            if (candidate_parts[i] == schema_parts[i] or 
                candidate_parts[i] == '*' or 
                schema_parts[i] == '*'):
                matches += 1
        
        # If most parts match, consider it a wildcard pattern match
        return matches >= len(candidate_parts) * 0.8
    
    return False


def calculate_edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return calculate_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def get_similarity_reason(candidate: str, schema_col: str, score: int) -> str:
    """
    Get human-readable reason for similarity score.
    
    Args:
        candidate: Column from used_columns
        schema_col: Column from schema
        score: Similarity score
        
    Returns:
        Reason string
    """
    if score == 100:
        return "exact match"
    elif score >= 80:
        return "contains match"
    elif score >= 70:
        return "wildcard pattern match"
    elif score >= 60:
        return "high token overlap"
    elif score >= 40:
        return "moderate similarity"
    else:
        return "low similarity"


def normalize_using_fuzzy_matches(used_columns: list, schema: list) -> tuple[list, list]:
    """
    Normalize used_columns using fuzzy matching without score filtering.
    
    Args:
        used_columns: List of columns to normalize
        schema: List of schema columns for validation
        
    Returns:
        Tuple of (normalized_columns, normalization_mappings)
    """
    schema_set = set(schema)
    normalized_columns = []
    normalization_mappings = []
    
    for col in used_columns:
        if col in schema_set:
            # Column exists in schema, keep as is
            normalized_columns.append(col)
            continue
        
        # Find best fuzzy match
        best_match = None
        best_score = 0
        
        for schema_col in schema:
            score = calculate_similarity_score(col, schema_col)
            if score > best_score:
                best_score = score
                best_match = schema_col
        
        if best_match:
            # Use the fuzzy match
            normalized_columns.append(best_match)
            normalization_mappings.append({
                'original': col,
                'normalized': best_match,
                'score': best_score,
                'method': 'fuzzy_match'
            })
        else:
            # Keep original if no match found
            normalized_columns.append(col)
    
    return normalized_columns, normalization_mappings


def normalize_used_columns(used_columns: list, schema: list) -> tuple[list, list]:
    """
    Normalize used_columns by flattening nested columns to their parent level.
    
    Args:
        used_columns: List of columns to normalize
        schema: List of schema columns for validation
        
    Returns:
        Tuple of (normalized_columns, normalization_mappings)
    """
    schema_set = set(schema)
    normalized_columns = []
    normalization_mappings = []
    
    # Check if any used_columns have "bigquery-public-data." prefix
    has_bigquery_prefix_in_used = any(col.startswith("bigquery-public-data.") for col in used_columns)
    
    for col in used_columns:
        normalized = False  # Initialize normalized flag for each column
        
        if col in schema_set:
            # Column exists in schema, keep as is
            normalized_columns.append(col)
            continue
        
        # Rule 2: Remove "bigquery-public-data." prefix if used_columns have it
        if has_bigquery_prefix_in_used and col.startswith("bigquery-public-data."):
            normalized_col = col.replace("bigquery-public-data.", "", 1)
            normalized_columns.append(normalized_col)
            normalization_mappings.append({
                'original': col,
                'normalized': normalized_col
            })
            continue
        
        # Rule 3: Remove redundant prefixes by comparing with schema
        parts = col.split('.')
        if len(parts) >= 3:  # Need at least prefix.table.column
            # Check if the first prefix exists in schema
            first_prefix = parts[0]
            first_prefix_in_schema = any(col.startswith(f"{first_prefix}.") for col in schema)
            
            if not first_prefix_in_schema:
                # Only try to remove prefixes if the first prefix is NOT in schema
                for i in range(1, len(parts) - 1):  # Try removing prefixes one by one
                    test_col = '.'.join(parts[i:])  # Remove first i parts
                    # Double-check: ensure the test_col actually exists in schema
                    if test_col in schema_set:
                        normalized_columns.append(test_col)
                        normalization_mappings.append({
                            'original': col,
                            'normalized': test_col
                        })
                        normalized = True
                        break
            
        if normalized:
            continue
            
        # Rule 1: Split by dots to find potential prefixes (nested column flattening)
        if len(parts) < 3:  # Need at least table.column.subcolumn for normalization
            normalized_columns.append(col)
            continue
            
        # Try to find prefix matches for nested columns
        for i in range(2, len(parts)):  # Start from table.column
            prefix = '.'.join(parts[:i])
            if prefix in schema_set:
                normalized_columns.append(prefix)
                normalization_mappings.append({
                    'original': col,
                    'normalized': prefix
                })
                normalized = True
                break
        
        if not normalized:
            # Keep original if no normalization possible
            normalized_columns.append(col)
    
    return normalized_columns, normalization_mappings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find used_columns that don't exist in schema.")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides).")
    p.add_argument("--collection-name", default="spider2_lite_samples",
                   help="MongoDB collection name (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of documents to process (for testing)")
    p.add_argument("--skip-backup", action="store_true",
                   help="Skip backup to openai_used_columns (just show used_columns)")
    p.add_argument("--analyze-normalization", action="store_true",
                   help="Analyze candidates for normalization (show what could be normalized)")
    p.add_argument("--normalize", action="store_true",
                   help="Actually perform normalization and update MongoDB (requires --analyze-normalization)")
    p.add_argument("--show-missing", action="store_true",
                   help="Show columns that are not in schema (missing columns)")
    p.add_argument("--fuzzy-match", action="store_true",
                   help="Show fuzzy matches for columns not in schema")

    p.add_argument("--target-ids", nargs='+', type=int,
                   help="Specific document IDs to process (e.g., --target-ids 5 171 172 178 194 228 231 239 323 361 386 389 390 391 392)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv("/home/datht/mats/.env")
    
    if args.skip_backup:
        print("=== Show used_columns (backup skipped) ===")
    else:
        print("=== Backup used_columns to openai_used_columns ===")
    
    client = MongoClient(args.mongo_uri)
    coll = client["mats"][args.collection_name]
    
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
    
    print(f"Found {len(docs_with_used_columns)} documents with used_columns")
    
    if args.skip_backup:
        # Just show used_columns without backup
        for doc in docs_with_used_columns:
            doc_id = doc["_id"]
            db_id = doc["db_id"]
            db_type = doc["db_type"]
            used_columns = doc.get("used_columns", [])
            schema = doc.get("schema", [])
            
            if args.analyze_normalization and schema:
                # Analyze normalization candidates
                analysis = analyze_normalization_candidates(used_columns, schema)
                
                # Always show the document header
                print(f"\n--- Document {doc_id} ({db_id}, {db_type}) ---")
                
                if analysis['candidates_count'] > 0:
                    print(f"    📋 Exact normalization candidates ({analysis['candidates_count']}):")
                    for candidate in analysis['normalization_candidates']:
                        print(f"      {candidate['original']} -> {candidate['suggested']}")
                else:
                    print(f"    📋 No exact normalization candidates found")
                
                # Perform normalization if requested
                if args.normalize:
                    # First try exact normalization
                    normalized_columns, normalization_mappings = normalize_used_columns(used_columns, schema)
                    
                    # If no exact normalization was possible, try fuzzy normalization
                    if len(normalization_mappings) == 0:
                        print(f"    🔍 No exact normalization possible, trying fuzzy matching...")
                        fuzzy_normalized_columns, fuzzy_normalization_mappings = normalize_using_fuzzy_matches(used_columns, schema)
                        
                        if len(fuzzy_normalization_mappings) > 0:
                            # Use fuzzy normalization results
                            normalized_columns = fuzzy_normalized_columns
                            normalization_mappings = [{"original": m["original"], "normalized": m["normalized"], "method": "fuzzy_match", "score": m["score"]} for m in fuzzy_normalization_mappings]
                            print(f"    🔍 Applied fuzzy normalization with {len(fuzzy_normalization_mappings)} mappings")
                    
                    # Update MongoDB
                    coll.update_one(
                        {"_id": doc_id},
                        {
                            "$set": {
                                "used_columns": normalized_columns,
                                "normalized_at": dt.datetime.utcnow(),
                                "normalization_mappings": normalization_mappings
                            }
                        }
                    )
                    print(f"    ✅ Normalized and updated MongoDB")
                    print(f"    Original count: {len(used_columns)}, Normalized count: {len(normalized_columns)}")
            

            
            # Show missing columns analysis if requested
            if args.show_missing and schema:
                missing_analysis = analyze_missing_in_schema(used_columns, schema)
                if missing_analysis['missing_count'] > 0:
                    if not args.analyze_normalization or analysis['candidates_count'] == 0:
                        print(f"\n--- Document {doc_id} ({db_id}, {db_type}) ---")
                    print(f"    ❌ Missing in schema ({missing_analysis['missing_count']}):")
                    for missing_col in missing_analysis['missing_columns']:
                        print(f"      {missing_col}")
            
            # Show fuzzy matches if requested
            if args.fuzzy_match and schema:
                fuzzy_analysis = find_fuzzy_matches(used_columns, schema)
                if fuzzy_analysis['fuzzy_matches_count'] > 0:
                    if not args.analyze_normalization and not args.show_missing:
                        print(f"\n--- Document {doc_id} ({db_id}, {db_type}) ---")
                    print(f"    🔍 Fuzzy match suggestions ({fuzzy_analysis['fuzzy_matches_count']}):")
                    for match in fuzzy_analysis['fuzzy_matches']:
                        original = match['original']
                        best_match = match['best_matches'][0]  # Top match
                        print(f"      {original} -> {best_match['schema_column']} (score: {best_match['score']}, {best_match['reason']})")
            
            elif not args.analyze_normalization and not args.show_missing and not args.fuzzy_match:
                print(f"\n--- Document {doc_id} ({db_id}, {db_type}) ---")
                print(f"  Used columns: {used_columns}")
    else:
        # Backup used_columns to openai_used_columns
        backup_count = 0
        for doc in docs_with_used_columns:
            doc_id = doc["_id"]
            db_id = doc["db_id"]
            db_type = doc["db_type"]
            used_columns = doc.get("used_columns", [])
            
            print(f"\n--- Document {doc_id} ({db_id}, {db_type}) ---")
            print(f"  Current used_columns: {used_columns}")
            
            # Check if openai_used_columns already exists
            if "openai_used_columns" not in doc:
                # Backup current used_columns to openai_used_columns
                coll.update_one(
                    {"_id": doc_id},
                    {"$set": {"openai_used_columns": used_columns}}
                )
                print(f"  ✅ Backed up to openai_used_columns")
                backup_count += 1
            else:
                print(f"  ⚠️  openai_used_columns already exists, skipping")
        
        print(f"\n=== BACKUP SUMMARY ===")
        print(f"Total documents processed: {len(docs_with_used_columns)}")
        print(f"Documents backed up: {backup_count}")
        print(f"Documents already had backup: {len(docs_with_used_columns) - backup_count}")
    
    client.close()


if __name__ == "__main__":
    main()
