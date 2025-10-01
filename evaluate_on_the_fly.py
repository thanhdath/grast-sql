#!/usr/bin/env python3
"""
evaluate_on_the_fly_embeddings.py
────────────────────────────────
Embed each (query, column) pair on-the-fly, run the frozen-encoder GNN, and
evaluate either top-k or threshold-based precision / recall.

Updated 2025-07-23
──────────────────
• Inserts per-sample prediction **and** ground-truth columns into
  MongoDB → mats.grast (host 192.168.1.108:27027).

Updated 2025-01-XX
──────────────────
• Added --evaluation_mode option:
  - end2end: current flow with graph transformers (default)
  - encoder_only: encoder + steiner tree with threshold (no graph transformers)
"""

from __future__ import annotations
import warnings
# Suppress truncation strategy warnings from transformers
warnings.filterwarnings("ignore", message=".*truncation strategy.*")
warnings.filterwarnings("ignore", message=".*We need to remove.*")
import argparse, json, pickle, time
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Any

import numpy as np
import torch
from torch.cuda.amp.autocast_mode import autocast
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pymongo import MongoClient

# project-local imports ------------------------------------------------------
from frozen_encoder_trainer.init_embeddings import EmbeddingInitializer, make_desc
from frozen_encoder_trainer.train_with_frozen_embeddings import (
    graph_to_data_with_embeddings,
    GraphColumnRetrieverFrozen,
    DEVICE,
)

DB_NAME = "mats"                             # ← same DB name as exporter
COLL_TEMPLATE = {                            # (dataset, split) → collection
    ("bird",   "train"): "train_samples",
    ("bird",   "dev")  : "dev_samples",
    ("spider", "train"): "spider_train_samples",
    ("spider", "dev")  : "spider_dev_samples",
    ("spider2", "train"): "spider2_lite_samples",  # Spider 2.0 uses single collection
    ("spider2", "dev")  : "spider2_lite_samples",  # Spider 2.0 uses single collection
}

# ─────────────────── CLI ───────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate on-the-fly embeddings")
    # dataset / paths --------------------------------------------------
    p.add_argument("--dataset",     choices=["bird", "spider", "spider2"], required=True)
    p.add_argument("--split",       choices=["train", "dev"],   default="dev")
    p.add_argument("--pkl_path",    type=Path, required=True,
                   help="Pickle produced by export_samples_graphs.py")
    p.add_argument("--checkpoint",  type=Path, required=False,
                   help="GraphColumnRetrieverFrozen checkpoint *.pt (required for end2end mode)")
    p.add_argument("--embeddings_dir", type=Path, required=False,
                   help="Directory containing pre-initialized embeddings (optional)")
    # embedding options -----------------------------------------------
    p.add_argument("--reranker_type", choices=["standard", "layerwise", "st", "qwen"], default="standard")
    p.add_argument("--encoder_path",  type=str,
                   help="Path to BGE-m3 (standard) or MiniCPM (layerwise) model")
    p.add_argument("--cut_layer",     type=int, default=39,
                   help="MiniCPM hidden layer to extract (layerwise only)")
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--max_length",    type=int, default=512)
    # GNN options ----------------------------------------------------\--
    p.add_argument("--hidden_dim",    type=int, default=1024)
    p.add_argument("--num_layers",    type=int, default=3)
    # evaluation options ----------------------------------------------
    p.add_argument("--evaluation_mode", choices=["end2end", "encoder_only"], default="end2end",
                   help="end2end: current flow with graph transformers; encoder_only: encoder + steiner tree with threshold")
    p.add_argument("--k",             nargs="+", type=int, default=[10, 20],
                   help="Fixed top-k values (e.g., 10, 20, 50, 100)")
    p.add_argument("--k_percent",     nargs="+", type=float, default=None,
                   help="Percentage-based top-k values (e.g., 10.0, 15.0, 20.0 for 10%, 15%, 20%)")
    p.add_argument("--threshold",     type=float,
                   help="Score threshold; if provided, top-k metrics are skipped")
    p.add_argument("--exclude_foreign_keys", action="store_true",
                   help="Exclude foreign key columns from precision/recall computation")

    # misc -------------------------------------------------------------
    p.add_argument("--mongo_uri",     type=str,
                   default="mongodb://192.168.1.108:27017",
                   help="Mongo URI for reading sample metadata")
    p.add_argument("--log_dir",       type=Path, default=Path("logs/on_the_fly_eval"),
                   help="Directory to save all log files and plots")
    # prediction-write Mongo ------------------------------------------
    p.add_argument("--pred_mongo_uri", type=str,
                   default="mongodb://192.168.1.108:27017",
                   help="Mongo URI to store predictions")
    p.add_argument("--pred_collection", type=str,
                   default="grast",
                   help="Collection name for predictions (DB fixed to 'mats')")
    return p.parse_args()

# ───────────────── helpers ─────────────────
def get_steiner_subgraph(G: nx.Graph, terminals: Sequence[str]) -> nx.Graph:
    G_u = G.to_undirected()
    forest = nx.Graph()
    terms = set(terminals)
    for comp in nx.connected_components(G_u):
        comp_terms = terms & comp
        if not comp_terms:
            continue
        sub = G_u.subgraph(comp).copy()
        if len(comp_terms) == 1:
            node = next(iter(comp_terms))
            forest.add_node(node, **G.nodes[node])
        else:
            forest = nx.compose(forest, steiner_tree(sub, comp_terms))
    return forest


def pr(preds: Sequence[str], gold: Sequence[str]) -> Tuple[float, float]:
    # De-duplicate gold columns before computing precision/recall
    gold_unique = list(dict.fromkeys(gold))  # Preserve order while removing duplicates
    tp = len(set(preds) & set(gold_unique))
    p  = tp / len(preds) if preds else 1.0
    r  = tp / len(gold_unique)  if gold_unique  else 1.0
    return p, r


def pr_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> Tuple[float, float]:
    return pr(pred[:k], gold)


def get_missing_columns(preds: Sequence[str], gold: Sequence[str]) -> List[str]:
    """
    Calculate missing columns using de-duplicated gold columns.
    
    Args:
        preds: Predicted columns
        gold: Ground truth columns (duplicates will be removed)
        
    Returns:
        List of missing unique columns
    """
    # De-duplicate gold columns to match the precision/recall calculation
    gold_unique = list(dict.fromkeys(gold))  # Preserve order while removing duplicates
    return list(set(gold_unique) - set(preds))





def load_ground_truth_sql(dataset: str, split: str, mongo_uri: str) -> Dict[Any, str]:
    """Load ground truth SQL from MongoDB for all samples."""
    coll_name = COLL_TEMPLATE[(dataset, split)]
    if dataset == "spider":
        key = "query"
    elif dataset == "bird":
        key = "SQL"
    elif dataset == "spider2":
        key = "SQL"  # Spider 2.0 uses "query" field like original Spider

    with MongoClient(mongo_uri) as client:
        coll = client[DB_NAME][coll_name]
        # Create mapping from sample_id to SQL
        sql_map = {}
        for doc in coll.find({}, {"_id": 1, key: 1}):
            sql_map[doc["_id"]] = doc.get(key, "")
    print(f"Loaded SQL map from '{DB_NAME}.{coll_name}' ({len(sql_map)} entries)")
    return sql_map


def filter_foreign_key_columns(G: nx.Graph, columns: Sequence[str]) -> List[str]:
    """Filter out foreign key columns from the given list."""
    return [col for col in columns if not G.nodes[col].get("is_in_foreign_key", False)]


def pr_exclude_fk(preds: Sequence[str], gold: Sequence[str], G: nx.Graph) -> Tuple[float, float]:
    """Compute precision/recall excluding foreign key columns."""
    # Filter out foreign key columns
    preds_filtered = filter_foreign_key_columns(G, preds)
    gold_filtered = filter_foreign_key_columns(G, gold)
    
    # De-duplicate gold columns before computing precision/recall
    gold_filtered_unique = list(dict.fromkeys(gold_filtered))  # Preserve order while removing duplicates
    
    tp = len(set(preds_filtered) & set(gold_filtered_unique))
    p  = tp / len(preds_filtered) if preds_filtered else 1.0
    r  = tp / len(gold_filtered_unique)  if gold_filtered_unique  else 1.0
    return p, r


def pr_at_k_exclude_fk(pred: Sequence[str], gold: Sequence[str], k: int, G: nx.Graph) -> Tuple[float, float]:
    """Compute precision/recall at k excluding foreign key columns."""
    return pr_exclude_fk(pred[:k], gold, G)


def get_database_stats(db_id: str, dataset: str, split: str) -> Tuple[int, int]:
    """
    Get actual table and column counts for a database using the database analyzer approach.
    
    Args:
        db_id: Database identifier
        dataset: Dataset name ("bird", "spider", or "spider2")
        
    Returns:
        Tuple of (table_count, column_count)
    """
    import sqlite3
    import re
    from pathlib import Path
    
    # For Spider 2.0, get stats from MongoDB instead of file system
    if dataset == "spider2":
        # Get stats from MongoDB using the schema information
        coll_name = COLL_TEMPLATE[(dataset, split)]
        with MongoClient("mongodb://192.168.1.108:27017") as client:
            coll = client[DB_NAME][coll_name]
            # Find one document with this db_id to get schema info
            doc = coll.find_one({"db_id": db_id})
            if doc:
                col_info = doc.get("column_info", {})
                # Count columns
                column_count = len(col_info)
                # Count unique tables
                tables = set()
                for col_name in col_info.keys():
                    table = col_name.split(".")[0]
                    tables.add(table)
                table_count = len(tables)
                return table_count, column_count
        return 0, 0
    
    # For other datasets, use file system approach
    if dataset == "spider":
        db_dir = Path("/home/datht/mats/data/sft_data_collections/spider/database") / db_id
    elif dataset == "bird" and split == "dev":
        db_dir = Path("/home/datht/mats/data/bird-062024/dev/dev_databases") / db_id
    elif dataset == "bird" and split == "train":
        db_dir = Path("/home/datht/mats/data/bird-062024/train/train_databases") / db_id
    else:
        return 0, 0
    
    if not db_dir.exists():
        return 0, 0
    
    # Try SQLite database first (more accurate)
    sqlite_files = list(db_dir.glob("*.sqlite"))
    if sqlite_files:
        try:
            conn = sqlite3.connect(sqlite_files[0])
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            table_count = len(tables)
            
            column_count = 0
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info(`{table_name}`);")
                columns = cursor.fetchall()
                column_count += len(columns)
            
            conn.close()
            return table_count, column_count
            
        except Exception as e:
            print(f"Error analyzing SQLite database {db_id}: {e}")
    
    return 0, 0

# ───────────────── main ────────────────────
def main() -> None:
    args = cli()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # 0) Mongo lookup map (_id → db_id) --------------------------------
    coll_name = COLL_TEMPLATE[(args.dataset, args.split)]
    with MongoClient(args.mongo_uri) as client:
        coll = client[DB_NAME][coll_name]
        id2db: Dict[Any, str] = {
            doc["_id"]: doc.get("db_id", "unknown")
            for doc in coll.find({}, {"db_id": 1})
        }
    print(f"Loaded db_id map from '{DB_NAME}.{coll_name}'  ({len(id2db)} entries)")

    # Load ground truth SQL
    sql_map = load_ground_truth_sql(args.dataset, args.split, args.mongo_uri)

    # --- prediction-write client -------------------------------------
    pred_client = MongoClient(args.pred_mongo_uri)
    pred_coll   = pred_client[DB_NAME][args.pred_collection]

    # 1) embedding initialiser -----------------------------------------
    default_enc = ("/home/datht/huggingface/BAAI/bge-m3" if args.reranker_type == "standard"
                   else "/home/datht/graph-schema/embedder/finetuned-reranker-v2-minicpm-layerwise-bird-lora/merged_model/")
    if args.reranker_type == "qwen":
        # Default to hub model; can be overridden via --encoder_path
        default_enc = "Qwen/Qwen3-Reranker-0.6B"
    
    # Load pre-initialized embeddings if available
    pre_embeddings = {}
    if args.embeddings_dir and args.embeddings_dir.exists():
        embedding_files = list(args.embeddings_dir.glob("*.pkl"))
        if embedding_files:
            with open(embedding_files[0], 'rb') as f:
                pre_embeddings = pickle.load(f)
            print(f"✓ Loaded {len(pre_embeddings)} pre-initialized embeddings from {embedding_files[0]}")
    
    # Initialize embedding model only if needed
    emb_init = None
    if not pre_embeddings:
        emb_init = EmbeddingInitializer(
            reranker_type=args.reranker_type,
            model_path   =args.encoder_path or default_enc,
            cut_layer    =args.cut_layer if args.reranker_type == "layerwise" else None,
            batch_size   =args.batch_size,
            max_length   =args.max_length,
            device       ="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        print("✓ Using on-the-fly embedding generation")
    else:
        print("✓ Using pre-initialized embeddings")

    # For encoder-only mode, also load the reranker like in 2_3_evaluate_reranker_and_steiner_tree.py
    reranker = None
    layer_kw = None
    if args.evaluation_mode == "encoder_only":
        if args.reranker_type == "standard":
            from FlagEmbedding import FlagReranker
            reranker = FlagReranker(str(args.encoder_path or default_enc),
                                    query_max_length=args.max_length,
                                    passage_max_length=args.max_length,
                                    batch_size=args.batch_size,
                                    use_fp16=True, devices=["cuda:0" if torch.cuda.is_available() else "cpu"])
            layer_kw = None
        elif args.reranker_type == "llm":
            from FlagEmbedding import FlagLLMReranker
            reranker = FlagLLMReranker(str(args.encoder_path or default_enc),
                                       query_max_length=args.max_length,
                                       passage_max_length=args.max_length,
                                       batch_size=args.batch_size,
                                       use_fp16=True, devices=["cuda:0" if torch.cuda.is_available() else "cpu"])
            layer_kw = None
        elif args.reranker_type == "layerwise":
            from FlagEmbedding import LayerWiseFlagLLMReranker
            reranker = LayerWiseFlagLLMReranker(str(args.encoder_path or default_enc),
                                                query_max_length=args.max_length,
                                                passage_max_length=args.max_length,
                                                batch_size=args.batch_size,
                                                use_fp16=True, devices=["cuda:0" if torch.cuda.is_available() else "cpu"])
            layer_kw = {"cutoff_layers": [args.cut_layer]} if args.cut_layer else {}
        elif args.reranker_type == "st":
            from sentence_transformers.cross_encoder import CrossEncoder
            reranker = CrossEncoder(str(args.encoder_path or default_enc), 
                                   max_length=args.max_length, 
                                   device="cuda:0" if torch.cuda.is_available() else "cpu")
            layer_kw = None
        elif args.reranker_type == "qwen":
            # Encoder-only scoring is not supported for qwen via vLLM pooling.
            # Use end2end mode with GNN, which uses vLLM embeddings through EmbeddingInitializer.
            raise ValueError("encoder_only mode is not supported for reranker_type='qwen'. Use --evaluation_mode end2end.")
        print(f"✓ Loaded reranker for encoder-only mode: {args.reranker_type}")
        
        # Configure tokenizer to avoid truncation warnings
        if reranker is not None and hasattr(reranker, 'tokenizer'):
            reranker.tokenizer.truncation_side = 'left'
            reranker.tokenizer.padding_side = 'right'
            # Set truncation strategy to longest_first to avoid warnings
            if hasattr(reranker.tokenizer, 'truncation_strategy'):
                reranker.tokenizer.truncation_strategy = 'longest_first'

    # 2) load triples ---------------------------------------------------
    triples = pickle.load(open(args.pkl_path, "rb"))
    print(f"Loaded {len(triples)} triples from {args.pkl_path}")
    
    # Note: wrong_label=True samples are already filtered out when building graphs in 1_build_graph.py
    # No need to filter again here

    # 3) frozen column retriever (only for end2end mode) --------------
    gnn = None
    if args.evaluation_mode == "end2end":
        # Get embedding dimension
        if pre_embeddings:
            first_embedding = next(iter(pre_embeddings.values()))
            if isinstance(first_embedding, dict):
                embed_dim = first_embedding['embeddings'].shape[-1]
            else:
                embed_dim = first_embedding.shape[-1]
        else:
            embed_dim = emb_init.embed_dim()
        
        gnn = GraphColumnRetrieverFrozen(
            embed_dim = embed_dim,
            hid_dim   = args.hidden_dim,
            num_layers= args.num_layers,
        )
        # Ensure DEVICE is a string or int for .to()
        # Convert DEVICE to string if needed
        if hasattr(DEVICE, 'type'):
            device_str = DEVICE.type if DEVICE.index is None else f"{DEVICE.type}:{DEVICE.index}"
        else:
            device_str = str(DEVICE)
        gnn = gnn.to(device_str)
        # Robust checkpoint loading across torch versions (2.6 defaults weights_only=True)
        try:
            chk = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)  # type: ignore[call-arg]
        except TypeError:
            # Older torch without weights_only
            chk = torch.load(args.checkpoint, map_location=DEVICE)
        except Exception:
            # Allowlist numpy scalar if required, then retry
            try:
                from torch.serialization import add_safe_globals  # type: ignore
                try:
                    from numpy.core.multiarray import scalar as _np_scalar  # type: ignore
                    add_safe_globals([_np_scalar])
                except Exception:
                    pass
            except Exception:
                pass
            chk = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)  # type: ignore[call-arg]
        gnn.load_state_dict(chk.get("model_state_dict", chk), strict=False)
        gnn.eval();  print("✓ GNN checkpoint loaded")
    else:
        print("✓ Encoder-only mode: skipping GNN loading")
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 4) containers -----------------------------------------------------
    # Validate arguments based on evaluation mode
    if args.evaluation_mode == "end2end":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for end2end mode")
        print(f"✓ End2end mode: will use GNN checkpoint {args.checkpoint}")
    elif args.evaluation_mode == "encoder_only":
        if args.threshold is None:
            print("Warning: encoder_only mode requires threshold. Setting default threshold to 0.0")
            args.threshold = 0.0
        print(f"✓ Encoder-only mode: will use threshold {args.threshold} with steiner tree")
    
    if args.exclude_foreign_keys:
        print("✓ Foreign key columns will be excluded from precision/recall computation")
    
    use_thr = args.threshold is not None
    
    # Process fixed k values
    fixed_ks = sorted(args.k) if not use_thr else []
    
    # Process percentage-based k values
    percent_ks = []
    if args.k_percent is not None and not use_thr:
        percent_ks = sorted(args.k_percent)
    
    # Combine all k values for stats initialization
    all_ks = fixed_ks + [f"{p}%" for p in percent_ks]
    stats = {k: dict(cp=[], cr=[]) for k in all_ks}
    stats_st = {k: dict(cp=[], cr=[]) for k in all_ks}
    thr_raw_p = []; thr_raw_r = []; thr_st_p = []; thr_st_r = []

    timings: List[Dict[str, Any]] = []
    per_db: Dict[str, Dict[str, Any]] = {}
    
    # Cache for database stats to avoid repeated lookups
    db_stats_cache: Dict[str, Tuple[int, int]] = {}

    gold_scores, non_gold_scores = [], []
    # Use dictionaries to maintain proper ordering and alignment
    sample_data = {}  # sample_id -> {scores, labels, names, graph, gold_cols}

    # Comment out unused pred_docs to satisfy linter
    # pred_docs: List[Dict[str, Any]] = []  # ← accumulate prediction docs
    
    # Container for samples with recall < 1.0
    low_recall_samples: List[Dict[str, Any]] = []
    


    # 5) iterate samples ------------------------------------------------
    for q, G, gold_cols, sample_id in tqdm(triples, desc="Samples"):
        db_id = id2db.get(sample_id, "unknown")

        names = sorted(G.nodes())
        descs = [make_desc(G.nodes[n]) for n in names]

        t0 = time.perf_counter()
        # Get embeddings
        if pre_embeddings:
            # Find matching pre-initialized embedding by query
            embeds = None
            for key, embedding_data in pre_embeddings.items():
                if isinstance(embedding_data, dict) and embedding_data.get('query') == q:
                    embeds = embedding_data['embeddings']
                    break
            if embeds is None:
                raise ValueError(f"No pre-initialized embedding found for query: {q[:100]}...")
        else:
            # Generate embeddings on-the-fly
            embeds_list = []
            for i in range(0, len(descs), args.batch_size):
                batch_descs = descs[i:i + args.batch_size]
                batch_embeds = emb_init.encode_pairs([q] * len(batch_descs), [batch_descs])
                embeds_list.append(batch_embeds)
            embeds = torch.cat(embeds_list, dim=0) if embeds_list else torch.empty((0, emb_init.embed_dim()))
        t_embed = time.perf_counter() - t0

        if args.evaluation_mode == "end2end":
            # End-to-end mode: use GNN
            pyg = graph_to_data_with_embeddings(G, q, gold_cols, embeds).to(device_str)
            t1 = time.perf_counter()
            with torch.no_grad(), autocast():
                logits = gnn(pyg)
            t_infer = time.perf_counter() - t1
        else:
            # Encoder-only mode: use reranker scores directly (like 2_3_evaluate_reranker_and_steiner_tree.py)
            t1 = time.perf_counter()
            # Get column descriptions
            descs = [make_desc(G.nodes[n]) for n in names]
            # Use reranker.compute_score exactly like in 2_3_evaluate_reranker_and_steiner_tree.py
            if reranker is None:
                raise ValueError("Reranker not loaded for encoder-only mode")
            scores_all: List[float] = []
            for i in range(0, len(descs), args.batch_size):
                batch_descs = descs[i:i + args.batch_size]
                if args.reranker_type == "st":
                    # CrossEncoder expects pairs as [sentence1, sentence2]
                    pairs = [[q, desc] for desc in batch_descs]
                    batch_scores = reranker.predict(pairs)
                    if isinstance(batch_scores, np.ndarray):
                        batch_scores = batch_scores.tolist()
                    scores_all.extend(batch_scores)
                else:
                    # FlagEmbedding rerankers
                    pairs = [(q, desc) for desc in batch_descs]
                    kw = {"normalize": False, **(layer_kw or {})}
                    
                    # Debug: Print sample info before computing scores
                    if i == 0:  # Only print for first batch of each sample
                        # Get tokenizer for token counting
                        tokenizer = reranker.tokenizer if hasattr(reranker, 'tokenizer') else None
                        
                        # Count tokens in query
                        query_tokens = len(tokenizer.encode(q)) if tokenizer else len(q.split())
                        
                        # Count tokens in all descriptions and find the longest
                        desc_token_counts = []
                        longest_desc_idx = 0
                        longest_desc_tokens = 0
                        if descs:
                            for j, desc in enumerate(descs):
                                desc_tokens = len(tokenizer.encode(desc)) if tokenizer else len(desc.split())
                                desc_token_counts.append(desc_tokens)
                                if desc_tokens > longest_desc_tokens:
                                    longest_desc_tokens = desc_tokens
                                    longest_desc_idx = j
                        
                        print(f"\n[DEBUG] Sample {sample_id} (db_id: {db_id}):")
                        print(f"  Query: {query_tokens} tokens ({len(q)} chars)")
                        print(f"  Number of columns: {len(names)}")
                        if descs:
                            print(f"  Longest desc: {longest_desc_tokens} tokens (column {longest_desc_idx}: {names[longest_desc_idx]})")
                            print(f"  Total tokens (query + longest_desc): {query_tokens + longest_desc_tokens}")
                            print(f"  Token count range: {min(desc_token_counts)} - {max(desc_token_counts)}")
                        print(f"  Query preview: {q[:100]}{'...' if len(q) > 100 else ''}")
                        if descs:
                            print(f"  Longest desc preview: {descs[longest_desc_idx][:100]}{'...' if len(descs[longest_desc_idx]) > 100 else ''}")
                    
                    scores = reranker.compute_score(pairs, **kw)
                    if scores is None:
                        scores_all.extend([0.0] * len(pairs))
                    elif isinstance(scores, list):
                        scores_all.extend(scores)
                    else:
                        scores_all.extend(scores.tolist())
            logits = torch.tensor(scores_all, device=device_str, dtype=torch.float32)
            t_infer = time.perf_counter() - t1

        # timing bookkeeping ------------------------------------------
        timings.append({"db_id": db_id, "embed_s": t_embed, "infer_s": t_infer})
        
        # Get actual database stats (cached to avoid repeated lookups)
        if db_id not in db_stats_cache:
            db_stats_cache[db_id] = get_database_stats(db_id, args.dataset, args.split)
        
        actual_tables, actual_cols = db_stats_cache[db_id]
        
        db_stat = per_db.setdefault(
            db_id,
            {"embed_times": [], "infer_times": [], "actual_tables": actual_tables, "actual_cols": actual_cols},
        )
        db_stat["embed_times"].append(t_embed)
        db_stat["infer_times"].append(t_infer)

        # scores / labels for AUCs ------------------------------------
        scores_np = logits.cpu().numpy()
        labels_np = np.array([1 if n in gold_cols else 0 for n in names])
        
        # Store data in dictionary for proper alignment
        sample_data[sample_id] = {
            'scores': scores_np,
            'labels': labels_np,
            'names': names,
            'graph': G,
            'gold_cols': gold_cols
        }

        for n, s in zip(names, scores_np):
            (gold_scores if n in gold_cols else non_gold_scores).append(float(s))

        order = scores_np.argsort()[::-1]
        preds_all = [names[i] for i in order]

        # -- build prediction doc ------------------------------------
        pred_doc = {
            "sample_id": sample_id,                        # logical sample id
            "db_id"    : db_id,
            "query"    : q,
            "pred_cols": preds_all,
            "scores"   : scores_np[order].astype(float).tolist(),
            "gold_cols": list(gold_cols),
            "created"  : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset"  : args.dataset,
            "split"    : args.split,
        }
        # Upsert prediction immediately after processing each sample
        # NOTE: Requires a compound unique index on (sample_id, dataset, split) in MongoDB:
        #   db.grast.createIndex({ sample_id: 1, dataset: 1, split: 1 }, { unique: true })
        pred_coll.replace_one(
            {"sample_id": pred_doc["sample_id"], "dataset": pred_doc["dataset"], "split": pred_doc["split"]},
            pred_doc,
            upsert=True
        )

        # evaluation bookkeeping -------------------------------------
        if use_thr:
            preds_thr = [n for n, sc in zip(names, scores_np) if sc >= args.threshold]
            
            # Use appropriate precision/recall function based on exclude_foreign_keys flag
            if args.exclude_foreign_keys:
                p_raw, r_raw = pr_exclude_fk(preds_thr, gold_cols, G)
            else:
                p_raw, r_raw = pr(preds_thr, gold_cols)

            st_nodes = list(get_steiner_subgraph(G, preds_thr).nodes())
            

            
            if args.exclude_foreign_keys:
                p_st, r_st = pr_exclude_fk(st_nodes, gold_cols, G)
            else:
                # Use pr_at_k to ensure consistency with raw calculation
                # For threshold mode, use the same number of predictions as the raw calculation
                p_st, r_st = pr_at_k(st_nodes, gold_cols, len(preds_thr))

            thr_raw_p.append(p_raw); thr_raw_r.append(r_raw)
            thr_st_p.append(p_st);  thr_st_r.append(r_st)
            
                        # Check for low recall samples
            if r_raw < 1.0:
                # Create column details with descriptions and scores
                column_details = []
                for i, name in enumerate(names):
                    column_details.append({
                        "column_name": name,
                        "description": descs[i],
                        "score": float(scores_np[i]),
                        "is_gold": name in gold_cols,
                        "is_predicted": name in preds_thr
                    })
                
                low_recall_samples.append({
                            "sample_id": sample_id,
                            "db_id": db_id,
                            "question": q,
                            "ground_truth_columns": list(gold_cols),
                            "ground_truth_columns_unique": list(dict.fromkeys(gold_cols)),  # De-duplicated version
                            "ground_truth_sql": sql_map.get(sample_id, ""),
                            "predicted_columns": preds_thr,
                            "recall": r_raw,
                            "precision": p_raw,
                            "threshold": args.threshold,
                            "evaluation_mode": args.evaluation_mode,
                            "exclude_foreign_keys": args.exclude_foreign_keys,
                            "missing_columns": get_missing_columns(preds_thr, gold_cols),
                            "column_details": column_details
                        })
        else:
            # Only for end2end mode with top-k evaluation
            if args.evaluation_mode == "end2end":
                for k in all_ks:
                    # Handle percentage-based k values
                    if isinstance(k, str) and k.endswith('%'):
                        # Calculate k based on percentage of total columns
                        percent = float(k[:-1]) / 100.0
                        actual_k = max(1, int(len(names) * percent))
                    else:
                        # Fixed k value
                        actual_k = int(k)
                    
                    # Use appropriate precision/recall function based on exclude_foreign_keys flag
                    if args.exclude_foreign_keys:
                        cp, cr = pr_at_k_exclude_fk(preds_all, gold_cols, actual_k, G)
                    else:
                        cp, cr = pr_at_k(preds_all, gold_cols, actual_k)
                    
                    stats[k]["cp"].append(cp); stats[k]["cr"].append(cr)

                    st_nodes = list(get_steiner_subgraph(G, preds_all[:actual_k]).nodes())
                    

                    
                    if args.exclude_foreign_keys:
                        p_st, r_st = pr_exclude_fk(st_nodes, gold_cols, G)
                    else:
                        # Use pr_at_k to ensure consistency with raw calculation
                        # This ensures we use the same number of predictions as the raw calculation
                        p_st, r_st = pr_at_k(st_nodes, gold_cols, actual_k)
                    
                    stats_st[k]["cp"].append(p_st)
                    stats_st[k]["cr"].append(r_st)
                    
                    # Check for low recall samples at each k
                    if cr < 1.0:
                        # Create column details with descriptions and scores
                        column_details = []
                        for i, name in enumerate(names):
                            column_details.append({
                                "column_name": name,
                                "description": descs[i],
                                "score": float(scores_np[i]),
                                "is_gold": name in gold_cols,
                                "is_predicted": name in preds_all[:actual_k]
                            })
                        
                        low_recall_samples.append({
                            "sample_id": sample_id,
                            "db_id": db_id,
                            "question": q,
                            "ground_truth_columns": list(gold_cols),
                            "ground_truth_columns_unique": list(dict.fromkeys(gold_cols)),  # De-duplicated version
                            "ground_truth_sql": sql_map.get(sample_id, ""),
                            "predicted_columns": preds_all[:actual_k],
                            "recall": cr,
                            "precision": cp,
                            "k": k,
                            "actual_k": actual_k,
                            "total_columns": len(names),
                            "evaluation_mode": args.evaluation_mode,
                            "exclude_foreign_keys": args.exclude_foreign_keys,
                            "missing_columns": get_missing_columns(preds_all[:actual_k], gold_cols),
                            "column_details": column_details
                        })

    # ── save low recall samples log ──────────────────────────────────
    if low_recall_samples:
        # Update filename to indicate if foreign keys are excluded
        suffix = "_no_fk" if args.exclude_foreign_keys else ""
        low_recall_log = args.log_dir / f"low_recall_samples_{args.dataset}_{args.split}_{args.evaluation_mode}{suffix}.json"
        with low_recall_log.open("w", encoding="utf-8") as fh:
            json.dump(low_recall_samples, fh, indent=2, ensure_ascii=False)
        print(f"\n✓ Logged {len(low_recall_samples)} samples with recall < 1.0 → {low_recall_log.resolve()}")
        
        # Also save a summary
        summary = {
            "total_samples": len(triples),
            "low_recall_samples": len(low_recall_samples),
            "low_recall_percentage": len(low_recall_samples) / len(triples) * 100,
            "evaluation_mode": args.evaluation_mode,
            "exclude_foreign_keys": args.exclude_foreign_keys,
            "evaluation_type": "threshold" if use_thr else "top_k",
            "threshold": args.threshold if use_thr else None,
            "k_values": all_ks if not use_thr else None
        }
        summary_log = args.log_dir / f"low_recall_summary_{args.dataset}_{args.split}_{args.evaluation_mode}{suffix}.json"
        with summary_log.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"✓ Summary saved → {summary_log.resolve()}")
    else:
        print("\n✓ All samples achieved perfect recall!")

    # ── timing report ────────────────────────────────────────────────
    print("\nPer-DB timing / size")
    print("db_id                 #tables  #cols   avg_embed(s)  avg_infer(s)")
    for db_id, d in per_db.items():
        print(f"{db_id:<20} {d['actual_tables']:>7} {d['actual_cols']:>7} "
              f"{np.mean(d['embed_times']):>12.3f} {np.mean(d['infer_times']):>12.3f}")

    overall_e = np.mean([t["embed_s"] for t in timings])
    overall_i = np.mean([t["infer_s"] for t in timings])
    print(f"\nOverall avg embed  : {overall_e:.3f} s")
    print(f"Overall avg infer  : {overall_i:.3f} s")

    # ── AUCs ─────────────────────────────────────────────────────────
    # Extract all scores and labels from sample_data for AUC calculation
    all_scores_flat = np.concatenate([data['scores'] for data in sample_data.values()])
    all_labels_flat = np.concatenate([data['labels'] for data in sample_data.values()])
    auc_roc = roc_auc_score(all_labels_flat, all_scores_flat)
    prec_mi, rec_mi, _ = precision_recall_curve(all_labels_flat, all_scores_flat)
    auc_pr = auc(rec_mi, prec_mi)
    print(f"\nMicro ROC-AUC = {auc_roc:.4f}")
    print(f"Micro PR-AUC  = {auc_pr:.4f}")

    # ── Save score distribution data ─────────────────────────────────
    # Save raw score data for plotting and analysis
    suffix = "_no_fk" if args.exclude_foreign_keys else ""
    score_data_path = args.log_dir / f"score_distribution_data_{args.dataset}_{args.split}_{args.evaluation_mode}{suffix}.json"
    
    # Prepare score distribution data
    score_distribution_data = {
        "metadata": {
            "dataset": args.dataset,
            "split": args.split,
            "evaluation_mode": args.evaluation_mode,
            "exclude_foreign_keys": args.exclude_foreign_keys,
            "total_samples": len(triples),
            "total_columns": len(all_scores_flat),
            "gold_columns": len(gold_scores),
            "non_gold_columns": len(non_gold_scores),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr)
        },
        "score_statistics": {
            "gold_scores": {
                "mean": float(np.mean(gold_scores)),
                "std": float(np.std(gold_scores)),
                "min": float(np.min(gold_scores)),
                "max": float(np.max(gold_scores)),
                "median": float(np.median(gold_scores)),
                "count": len(gold_scores)
            },
            "non_gold_scores": {
                "mean": float(np.mean(non_gold_scores)),
                "std": float(np.std(non_gold_scores)),
                "min": float(np.min(non_gold_scores)),
                "max": float(np.max(non_gold_scores)),
                "median": float(np.median(non_gold_scores)),
                "count": len(non_gold_scores)
            },
            "all_scores": {
                "mean": float(np.mean(all_scores_flat)),
                "std": float(np.std(all_scores_flat)),
                "min": float(np.min(all_scores_flat)),
                "max": float(np.max(all_scores_flat)),
                "median": float(np.median(all_scores_flat)),
                "count": len(all_scores_flat)
            }
        },
        "raw_scores": {
            "gold_scores": gold_scores,
            "non_gold_scores": non_gold_scores,
            "all_scores": all_scores_flat.tolist(),
            "all_labels": all_labels_flat.tolist()
        },
        "per_sample_scores": {
            sample_id: {
                "scores": data['scores'].tolist(),
                "labels": data['labels'].tolist(),
                "names": data['names'],
                "gold_columns": list(data['gold_cols']),
                "db_id": id2db.get(sample_id, "unknown")
            } for sample_id, data in sample_data.items()
        }
    }
    
    with score_data_path.open("w", encoding="utf-8") as fh:
        json.dump(score_distribution_data, fh, indent=2)
    print(f"✓ Score distribution data saved → {score_data_path.resolve()}")
    
    # Also save a simplified CSV version for easy plotting
    csv_score_path = args.log_dir / f"score_distribution_{args.dataset}_{args.split}_{args.evaluation_mode}{suffix}.csv"
    with csv_score_path.open("w", encoding="utf-8") as fh:
        fh.write("score,label,type\n")
        for score in gold_scores:
            fh.write(f"{score},1,gold\n")
        for score in non_gold_scores:
            fh.write(f"{score},0,non_gold\n")
        print(f"✓ Score distribution CSV saved → {csv_score_path.resolve()}")
    
    # ── Save per-database score distribution data ───────────────────
    # Group scores by database for analysis
    db_score_data = {}
    for sample_id, data in sample_data.items():
        db_id = id2db.get(sample_id, "unknown")
        if db_id not in db_score_data:
            db_score_data[db_id] = {
                "scores": [],
                "labels": [],
                "gold_scores": [],
                "non_gold_scores": [],
                "sample_count": 0,
                "column_count": 0
            }
        
        scores = data['scores']
        labels = data['labels']
        names = data['names']
        gold_cols = data['gold_cols']
        
        db_score_data[db_id]["scores"].extend(scores.tolist())
        db_score_data[db_id]["labels"].extend(labels.tolist())
        db_score_data[db_id]["sample_count"] += 1
        db_score_data[db_id]["column_count"] += len(names)
        
        # Separate gold and non-gold scores
        for name, score in zip(names, scores):
            if name in gold_cols:
                db_score_data[db_id]["gold_scores"].append(float(score))
            else:
                db_score_data[db_id]["non_gold_scores"].append(float(score))
    
    # Calculate statistics for each database
    for db_id, data in db_score_data.items():
        if data["gold_scores"]:
            data["gold_stats"] = {
                "mean": float(np.mean(data["gold_scores"])),
                "std": float(np.std(data["gold_scores"])),
                "min": float(np.min(data["gold_scores"])),
                "max": float(np.max(data["gold_scores"])),
                "median": float(np.median(data["gold_scores"])),
                "count": len(data["gold_scores"])
            }
        else:
            data["gold_stats"] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "count": 0}
        
        if data["non_gold_scores"]:
            data["non_gold_stats"] = {
                "mean": float(np.mean(data["non_gold_scores"])),
                "std": float(np.std(data["non_gold_scores"])),
                "min": float(np.min(data["non_gold_scores"])),
                "max": float(np.max(data["non_gold_scores"])),
                "median": float(np.median(data["non_gold_scores"])),
                "count": len(data["non_gold_scores"])
            }
        else:
            data["non_gold_stats"] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "count": 0}
        
        # Calculate overall stats
        all_scores = data["scores"]
        data["overall_stats"] = {
            "mean": float(np.mean(all_scores)),
            "std": float(np.std(all_scores)),
            "min": float(np.min(all_scores)),
            "max": float(np.max(all_scores)),
            "median": float(np.median(all_scores)),
            "count": len(all_scores)
        }
    
    # Save per-database score data
    db_score_path = args.log_dir / f"per_database_scores_{args.dataset}_{args.split}_{args.evaluation_mode}{suffix}.json"
    with db_score_path.open("w", encoding="utf-8") as fh:
        json.dump(db_score_data, fh, indent=2)
    print(f"✓ Per-database score data saved → {db_score_path.resolve()}")
    
    # ── Precision / Recall results ──────────────────────────────────
    print(f"\nEvaluation Mode: {args.evaluation_mode}")
    if args.exclude_foreign_keys:
        print("Foreign key columns excluded from precision/recall computation")
    if use_thr:
        print(f"Threshold {args.threshold}")
        print(f"Raw     : P={np.mean(thr_raw_p):.3f}  R={np.mean(thr_raw_r):.3f}")
        print(f"Steiner : P={np.mean(thr_st_p):.3f}  R={np.mean(thr_st_r):.3f}")
    else:
        print("\nTop-k results:")
        for k in all_ks:
            print(f"@{k:>6} P={np.mean(stats[k]['cp']):.3f} R={np.mean(stats[k]['cr']):.3f} | "
                  f"Steiner P={np.mean(stats_st[k]['cp']):.3f} R={np.mean(stats_st[k]['cr']):.3f}")

    # ── save JSON log ───────────────────────────────────────────────
    args.log_dir.mkdir(parents=True, exist_ok=True)
    # Update filename to indicate if foreign keys are excluded
    suffix = "_no_fk" if args.exclude_foreign_keys else ""
    out_json = args.log_dir / f"on_the_fly_eval_{args.dataset}_{args.split}_{args.evaluation_mode}{suffix}.json"
    payload: Dict[str, Any] = {
        "evaluation_mode": args.evaluation_mode,
        "exclude_foreign_keys": args.exclude_foreign_keys,
        "per_db": {
            db: {
                "n_tables":   d["actual_tables"],
                "n_columns":  d["actual_cols"],
                "avg_embed_s": float(np.mean(d["embed_times"])),
                "avg_infer_s": float(np.mean(d["infer_times"])),
            } for db, d in per_db.items()
        },
        "overall_avg_embed_s": float(overall_e),
        "overall_avg_infer_s": float(overall_i),
        "roc_auc_micro": auc_roc,
        "pr_auc_micro":  auc_pr,
    }
    if use_thr:
        payload["threshold"] = args.threshold
        payload["threshold_stats"] = {
            "Raw_P":      float(np.mean(thr_raw_p)),
            "Raw_R":      float(np.mean(thr_raw_r)),
            "Steiner_P":  float(np.mean(thr_st_p)),
            "Steiner_R":  float(np.mean(thr_st_r)),
        }
    else:
        payload["topk_stats"] = {
            k: {
                "P":          float(np.mean(stats[k]["cp"])),
                "R":          float(np.mean(stats[k]["cr"])),
                "Steiner_P":  float(np.mean(stats_st[k]["cp"])),
                "Steiner_R":  float(np.mean(stats_st[k]["cr"])),
            } for k in all_ks
        }
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n✓ Log saved to {out_json.resolve()}")
    


    # ── Foreign key statistics ─────────────────────────────────────
    if args.exclude_foreign_keys:
        total_fk_cols = 0
        total_fk_gold = 0
        total_cols = 0
        for sample_data_item in sample_data.values():
            names = sample_data_item['names']
            gold_cols = sample_data_item['gold_cols']
            graph = sample_data_item['graph']
            
            fk_cols = [n for n in names if graph.nodes[n].get("is_in_foreign_key", False)]
            fk_gold = [n for n in fk_cols if n in gold_cols]
            
            total_fk_cols += len(fk_cols)
            total_fk_gold += len(fk_gold)
            total_cols += len(names)
        
        print(f"\nForeign Key Statistics:")
        print(f"Total columns: {total_cols}")
        print(f"Foreign key columns: {total_fk_cols} ({total_fk_cols/total_cols*100:.1f}%)")
        print(f"Foreign key columns in gold: {total_fk_gold} ({total_fk_gold/total_fk_cols*100:.1f}% of FK columns)")
        print(f"Non-foreign key columns: {total_cols - total_fk_cols} ({(total_cols - total_fk_cols)/total_cols*100:.1f}%)")

    # ── optional timing histogram ───────────────────────────────────
    plt.figure(figsize=(7,4))
    sns.histplot([t["embed_s"] for t in timings], label="embed", bins=40, alpha=0.6)
    sns.histplot([t["infer_s"] for t in timings], label="infer", bins=40, alpha=0.6)
    plt.xlabel("seconds"); plt.ylabel("count"); plt.legend(); plt.tight_layout()
    plt.savefig(args.log_dir / "timing_hist.png")
    plt.close()

    # ───────── threshold analysis & plots ─────────
    if use_thr:
        print("\nGenerating threshold curve analysis...")
        # Extract data in the same order as triples for proper alignment
        all_scores = [sample_data[sample_id]['scores'] for _, _, _, sample_id in triples]
        all_labels = [sample_data[sample_id]['labels'] for _, _, _, sample_id in triples]
        all_graphs = [sample_data[sample_id]['graph'] for _, _, _, sample_id in triples]
        all_names = [sample_data[sample_id]['names'] for _, _, _, sample_id in triples]
        
        # Debug: Verify alignment
        print(f"Data alignment check:")
        print(f"  Number of samples: {len(triples)}")
        print(f"  all_scores length: {len(all_scores)}")
        print(f"  all_labels length: {len(all_labels)}")
        print(f"  all_graphs length: {len(all_graphs)}")
        print(f"  all_names length: {len(all_names)}")
        
        # Check first sample alignment
        if len(all_scores) > 0:
            first_scores = all_scores[0]
            first_labels = all_labels[0]
            first_names = all_names[0]
            print(f"  First sample - scores shape: {first_scores.shape}, labels shape: {first_labels.shape}, names len: {len(first_names)}")
            print(f"  First sample - scores sum: {first_scores.sum()}, labels sum: {first_labels.sum()}")
        
        plot_and_save_threshold_curves(
            all_scores, all_labels, all_graphs, all_names, 
            gold_scores, non_gold_scores, args.log_dir, args.exclude_foreign_keys
        )


# ───────── threshold analysis functions ─────────
def _macro_metrics_at_threshold(
    sample_scores: List[np.ndarray],
    sample_labels: List[np.ndarray],
    sample_graphs: List[nx.Graph],
    sample_names: List[List[str]],
    thr: float,
    exclude_foreign_keys: bool = False,
):
    """Compute macro-averaged metrics at a threshold, both raw and Steiner."""
    pr_raw, re_raw = [], []
    pr_st , re_st  = [], []

    for scores, labels, G, names in zip(
        sample_scores, sample_labels, sample_graphs, sample_names
    ):
        mask = scores >= thr
        
        if exclude_foreign_keys:
            # Filter out foreign key columns for evaluation
            fk_mask = np.array([not G.nodes[name].get("is_in_foreign_key", False) for name in names])
            mask = mask & fk_mask
            labels = labels & fk_mask
        
        tp = int(((mask == 1) & (labels == 1)).sum())
        fp = int(((mask == 1) & (labels == 0)).sum())
        fn = int(((mask == 0) & (labels == 1)).sum())

        p_raw = tp / (tp + fp) if tp + fp else 0.0
        r_raw = tp / (tp + fn) if tp + fn else 0.0
        pr_raw.append(p_raw); re_raw.append(r_raw)

        preds_names = [n for n, m in zip(names, mask) if m]
        truths = {n for n, lab in zip(names, labels) if lab == 1}
        
        if exclude_foreign_keys:
            # Filter out foreign key columns for Steiner tree evaluation
            preds_names = filter_foreign_key_columns(G, preds_names)
            truths = set(filter_foreign_key_columns(G, list(truths)))
        
        if len(preds_names) > 0:
            st_nodes = set(get_steiner_subgraph(G, preds_names).nodes())
            if exclude_foreign_keys:
                st_nodes = set(filter_foreign_key_columns(G, list(st_nodes)))
            tp_s = len(st_nodes & truths)
            p_st = tp_s / len(st_nodes) if st_nodes else 0.0
            r_st = tp_s / len(truths) if truths   else 0.0
        else:
            p_st = r_st = 0.0
        pr_st.append(p_st); re_st.append(r_st)

    mean = lambda arr: float(np.mean(arr))
    return (
        mean(pr_raw), mean(re_raw),
        mean(pr_st),  mean(re_st),
    )


def plot_and_save_threshold_curves(
    sample_scores, sample_labels, sample_graphs, sample_names,
    gold_scores, non_gold_scores, out_dir: Path, exclude_foreign_keys: bool = False,
):
    """Create plots and write CSV with both raw and Steiner macro curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t_min = min(float(s.min()) for s in sample_scores)
    t_max = max(float(s.max()) for s in sample_scores)
    thresholds = np.linspace(t_min, t_max, 50)
    print(thresholds)

    raw_p, raw_r = [], []
    st_p , st_r  = [], []
    for t in thresholds:
        rp, rr, sp, sr = _macro_metrics_at_threshold(
            sample_scores, sample_labels, sample_graphs, sample_names, t, exclude_foreign_keys
        )
        # print(f"Threshold: {t}, Raw Precision: {rp}, Raw Recall: {rr}, Steiner Precision: {sp}, Steiner Recall: {sr}")
        raw_p.append(rp); raw_r.append(rr)
        st_p.append(sp);  st_r.append(sr)

    # Update filename to indicate if foreign keys are excluded
    suffix = "_no_fk" if exclude_foreign_keys else ""
    csv_path = out_dir / f"macro_threshold_curve{suffix}.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("threshold,prec_raw,rec_raw,prec_st,rec_st\n")
        for row in zip(thresholds, raw_p, raw_r, st_p, st_r):
            fh.write(",".join(f"{v:.6f}" for v in row) + "\n")
    print(f"Wrote threshold curve CSV to {csv_path.resolve()}")
    
    # Save comprehensive threshold analysis data
    threshold_data_path = out_dir / f"threshold_analysis_data{suffix}.json"
    threshold_data = {
        "metadata": {
            "exclude_foreign_keys": exclude_foreign_keys,
            "num_thresholds": len(thresholds),
            "threshold_range": {
                "min": float(thresholds.min()),
                "max": float(thresholds.max()),
                "step": float(thresholds[1] - thresholds[0]) if len(thresholds) > 1 else 0.0
            }
        },
        "threshold_curves": {
            "thresholds": thresholds.tolist(),
            "raw_precision": raw_p,
            "raw_recall": raw_r,
            "steiner_precision": st_p,
            "steiner_recall": st_r
        },
        "score_distributions": {
            "gold_scores": gold_scores,
            "non_gold_scores": non_gold_scores,
            "gold_score_stats": {
                "mean": float(np.mean(gold_scores)),
                "std": float(np.std(gold_scores)),
                "min": float(np.min(gold_scores)),
                "max": float(np.max(gold_scores)),
                "median": float(np.median(gold_scores)),
                "count": len(gold_scores)
            },
            "non_gold_score_stats": {
                "mean": float(np.mean(non_gold_scores)),
                "std": float(np.std(non_gold_scores)),
                "min": float(np.min(non_gold_scores)),
                "max": float(np.max(non_gold_scores)),
                "median": float(np.median(non_gold_scores)),
                "count": len(non_gold_scores)
            }
        }
    }
    
    with threshold_data_path.open("w", encoding="utf-8") as fh:
        json.dump(threshold_data, fh, indent=2)
    print(f"✓ Threshold analysis data saved → {threshold_data_path.resolve()}")

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, raw_p, label="Raw Precision")
    plt.plot(thresholds, raw_r, label="Raw Recall")
    plt.plot(thresholds, st_p, '--', label="Steiner Precision")
    plt.plot(thresholds, st_r, '--', label="Steiner Recall")
    plt.xlabel("Threshold"); plt.ylabel("Score"); plt.legend()
    title = "Macro Precision / Recall vs Threshold"
    if exclude_foreign_keys:
        title += " (Foreign Keys Excluded)"
    plt.title(title)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / f"macro_precision_recall_vs_threshold{suffix}.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.histplot(gold_scores, kde=True, stat='density', bins=50, label="Gold", alpha=0.6)
    sns.histplot(non_gold_scores, kde=True, stat='density', bins=50, label="Non‑Gold", alpha=0.6)
    plt.xlabel("Score"); plt.ylabel("Density"); plt.legend(); plt.tight_layout()
    title = "Score distributions"
    if exclude_foreign_keys:
        title += " (Foreign Keys Excluded)"
    plt.title(title)
    plt.savefig(out_dir / f"score_distributions{suffix}.png")
    plt.close()


if __name__ == "__main__":
    main()
