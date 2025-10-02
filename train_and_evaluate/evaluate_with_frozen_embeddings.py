#!/usr/bin/env python3
"""
evaluate_with_frozen_embeddings.py (macro-PR version)
─────────────────────────────────────────────────────
Evaluate a trained GraphColumnRetrieverFrozen on a dev set using pre‑initialized
embeddings. This revision FIXES the threshold analysis: instead of pooling all
column scores across ALL samples (which produced a *micro* curve dominated by
large queries), we now compute Precision/Recall per sample at each threshold and
plot the *macro* average across samples. This yields a query‑balanced view of
threshold behaviour.

Additional notes:
- We still compute pooled (micro) ROC/PR AUC numbers for reference because some
  downstream comparisons used them; these are clearly labelled as "micro".
- Gold / non‑gold score distributions are still pooled (useful for sanity checks).
- Backwards‑compatible top‑k evaluation is unchanged.

Usage example:
python evaluate_with_frozen_embeddings_macroPR.py \
  --dev_embeddings_dir data/embeddings/dev \
  --dataset bird --reranker_type standard \
  --checkpoint ckpts/model.pt --k 5 10 20
"""

import argparse, json, os
from pathlib import Path
from typing import Dict, Any, List, Sequence, Tuple

import torch
from torch_geometric.loader import DataLoader
from torch.cuda.amp.autocast_mode import autocast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
import matplotlib.pyplot as plt
import seaborn as sns

# Import model and utils from train_with_frozen_embeddings.py
from train_with_frozen_embeddings import (
    load_embeddings_and_metadata,
    create_dataset_from_embeddings,  # unused but kept for API compatibility
    graph_to_data_with_embeddings,
    GraphColumnRetrieverFrozen,
    DEVICE,
)
from init_embeddings import make_desc

# ---------------------------------------------------------------------------
# CLI ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate GraphColumnRetrieverFrozen with frozen embeddings")
    p.add_argument("--dev_embeddings_dir", type=Path, required=True,
                   help="Directory containing pre-initialized dev embeddings")
    p.add_argument("--dataset", choices=["bird", "spider", "spider2"], required=True,
                   help="Dataset name (bird, spider, or spider2)")
    p.add_argument("--reranker_type", choices=["standard", "layerwise", "qwen"], required=True,
                   help="Reranker type used for embeddings (standard, layerwise, or qwen)")
    p.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    p.add_argument("--hid_dim", type=int, default=1024, help="GNN hidden dimension")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model checkpoint (*.pt)")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation (default: 1)")
    p.add_argument("--k", nargs="+", type=int, default=[10, 20], help="Top-k values for evaluation (e.g. --k 5 10 20)")
    p.add_argument("--log-file", type=Path, default=Path("logs/eval_frozen_encoder.json"),
                   help="Path to save evaluation results as JSON")
    p.add_argument("--log-low-recall", action="store_true",
                   help="If set, write a JSON of samples with recall<1.0 at max-k, including missing columns.")
    p.add_argument("--log-higher-than-missing", action="store_true",
                   help="If set with --log-low-recall, also log columns ranked higher than the lowest-ranked missing column.")
    # Optional MongoDB enrichment for low-recall logs (matches Qwen reranker script)
    p.add_argument("--mongo-uri", default="mongodb://192.168.1.108:27017",
                   help="MongoDB connection URI (used only when --log-low-recall)")
    p.add_argument("--mongo-db", default="mats",
                   help="MongoDB database name (used only when --log-low-recall)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Graph utils ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def get_steiner_subgraph(G: "nx.Graph", terminals: Sequence[str]) -> "nx.Graph":
    """Return Steiner forest of the subgraph induced by *terminals*.

    Handles disconnected graphs by computing a Steiner tree within each
    connected component that contains >=1 terminal, then composing them into a
    forest. For singleton terminals, just include the node.
    """
    G_u = G.to_undirected()
    terms = set(terminals)
    forest = nx.Graph()
    for comp_nodes in nx.connected_components(G_u):
        comp_terms = terms & comp_nodes
        if not comp_terms:
            continue
        subG = G_u.subgraph(comp_nodes).copy()
        if len(comp_terms) == 1:
            node = next(iter(comp_terms))
            forest.add_node(node, **G.nodes[node])
        else:
            forest = nx.compose(forest, steiner_tree(subG, comp_terms))
    return forest

# ---------------------------------------------------------------------------
# Description helpers (for logging) -----------------------------------------
# ---------------------------------------------------------------------------

MEANING_DICT: Dict[str, str] = {}

def _get_table_column(col_name: str) -> Tuple[str, str]:
    elms = col_name.split(".")
    table = elms[0]
    col = ".".join(elms[1:])
    return table, col

# def make_desc(node: Dict[str, Any]) -> str:
#     """Create a human-readable description for a column node (like Qwen script)."""
#     col = node.get("node_name", "")
#     col_type = node.get("type", "")
#     meaning = node.get("meaning", "")
#     if col:
#         if col in MEANING_DICT:
#             meaning = MEANING_DICT[col]
#         elif node.get("generated_column_meaning"):
#             meaning = node["generated_column_meaning"]
#     vals = " , ".join(map(str, node.get("similar_values", [])[:2]))
#     has_null = node.get("has_null", False)
#     val_desc = node.get("value_desc", "")
#     table, column = _get_table_column(col) if col else ("", "")
#     parts = [f"{table} . {column}", meaning, f"type {col_type}", f"has values {vals}", f"has_null = {has_null}"]
#     if isinstance(val_desc, str) and val_desc.strip():
#         parts.append(f"Value description: {val_desc.strip()}")
#     return " ; ".join(parts)

def make_desc_from_graph(G: nx.Graph, col_name: str) -> str:
    node_data = G.nodes[col_name] if col_name in G.nodes else {}
    return make_desc(node_data)

# ---------------------------------------------------------------------------
# Eval core -----------------------------------------------------------------
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_topks_and_collect_scores(
    loader: DataLoader,
    model: torch.nn.Module,
    ks: List[int],
    graphs: List[nx.Graph],
):
    """
    Evaluate and gather results.

    Returns
    -------
    stats_lists       : dict[top‑k] -> dict(metric -> list per‑sample)
    stats_st_lists    : dict[top‑k] -> dict(metric -> list per‑sample, Steiner)
    sample_scores     : list[np.ndarray]  – raw score vector per sample
    sample_labels     : list[np.ndarray]  – binary label vector per sample
    sample_names      : list[list[str]]   – column names (order matches score vector)
    gold_scores       : flat list[float]  – scores of gold columns
    non_gold_scores   : flat list[float]  – scores of non‑gold columns
    """
    model.eval()
    ks = sorted(set(ks))

    stats_lists = {k: {m: [] for m in ("prec", "rec_col", "rec_tab")} for k in ks}
    stats_st_lists = {k: {m: [] for m in ("cp", "cr")} for k in ks}

    sample_scores: List[np.ndarray] = []
    sample_labels: List[np.ndarray] = []
    sample_names:  List[List[str]]  = []

    gold_scores, non_gold_scores = [], []

    for i, data in enumerate(loader):  # bs = 1
        data = data.to(DEVICE)
        graph = graphs[i]

        with autocast():
            logits = model(data)  # (N,)
        names = data.orig_names[0]
        truths = [names[j] for j, y in enumerate(data.y) if y == 1]

        preds_idx = logits.argsort(descending=True)
        preds_all = [names[j] for j in preds_idx]

        scores_np = logits.detach().cpu().numpy()
        labels_np = data.y.detach().cpu().numpy()

        sample_scores.append(scores_np)
        sample_labels.append(labels_np)
        sample_names.append(names)

        for j, col in enumerate(names):
            if col in truths:
                gold_scores.append(float(scores_np[j]))
            else:
                non_gold_scores.append(float(scores_np[j]))

        for k in ks:
            k_eff = min(k, len(preds_all))
            preds_k = preds_all[:k_eff]

            tp = len(set(preds_k) & set(truths))
            stats_lists[k]["prec"].append(tp / k_eff if k_eff else 0.0)
            stats_lists[k]["rec_col"].append(tp / len(truths) if truths else 0.0)
            stats_lists[k]["rec_tab"].append(
                len({p.split('.')[0] for p in preds_k}
                    & {t.split('.')[0] for t in truths})
                / len({t.split('.')[0] for t in truths}) if truths else 0.0
            )

            st_nodes = list(get_steiner_subgraph(graph, preds_k).nodes())
            tp_st = len(set(st_nodes) & set(truths))
            stats_st_lists[k]["cp"].append(tp_st / len(st_nodes) if st_nodes else 0.0)
            stats_st_lists[k]["cr"].append(tp_st / len(truths) if truths else 1.0)

    return (
        stats_lists,
        stats_st_lists,
        sample_scores,
        sample_labels,
        sample_names,
        gold_scores,
        non_gold_scores,
    )

# ---------------------------------------------------------------------------
# Threshold analysis (MACRO) ------------------------------------------------
# ---------------------------------------------------------------------------

def _macro_metrics_at_threshold(
    sample_scores: List[np.ndarray],
    sample_labels: List[np.ndarray],
    sample_graphs: List[nx.Graph],
    sample_names: List[List[str]],
    thr: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute macro‑averaged metrics at a threshold, both raw and Steiner.

    Returns
    -------
    prec_raw, rec_raw, f1_raw,
    prec_st , rec_st , f1_st   : floats (macro averages)
    """
    precs_raw, recs_raw, f1s_raw = [], [], []
    precs_st , recs_st , f1s_st  = [], [], []

    for scores, labels, G, names in zip(
        sample_scores, sample_labels, sample_graphs, sample_names
    ):
        preds_mask = scores >= thr

        tp = int(((preds_mask == 1) & (labels == 1)).sum())
        fp = int(((preds_mask == 1) & (labels == 0)).sum())
        fn = int(((preds_mask == 0) & (labels == 1)).sum())

        p_raw = tp / (tp + fp) if tp + fp else 0.0
        r_raw = tp / (tp + fn) if tp + fn else 0.0
        f1_raw = 2 * p_raw * r_raw / (p_raw + r_raw) if p_raw + r_raw else 0.0
        precs_raw.append(p_raw)
        recs_raw.append(r_raw)
        f1s_raw.append(f1_raw)

        preds_names = [n for n, m in zip(names, preds_mask) if m]
        truths_set  = {n for n, lab in zip(names, labels) if lab == 1}

        if preds_names:
            st_nodes = set(get_steiner_subgraph(G, preds_names).nodes())
            tp_st = len(st_nodes & truths_set)
            p_st = tp_st / len(st_nodes) if st_nodes else 0.0
            r_st = tp_st / len(truths_set) if truths_set else 0.0
        else:
            p_st = r_st = 0.0
        f1_st = 2 * p_st * r_st / (p_st + r_st) if p_st + r_st else 0.0

        precs_st.append(p_st)
        recs_st.append(r_st)
        f1s_st.append(f1_st)

    macro = lambda arr: float(np.mean(arr))
    return (
        macro(precs_raw), macro(recs_raw), macro(f1s_raw),
        macro(precs_st),  macro(recs_st),  macro(f1s_st),
    )




def plot_threshold_analysis(
    sample_scores: List[np.ndarray],
    sample_labels: List[np.ndarray],
    sample_graphs: List[nx.Graph],
    sample_names: List[List[str]],
    gold_scores: Sequence[float],
    non_gold_scores: Sequence[float],
    log_dir: str,
):
    """
    Create plots and write CSV with both raw and Steiner macro curves.
    """
    os.makedirs(log_dir, exist_ok=True)

    all_min = min(float(s.min()) for s in sample_scores)
    all_max = max(float(s.max()) for s in sample_scores)
    thresholds = np.linspace(all_min, all_max, 200)

    prec_raw, rec_raw, f1_raw = [], [], []
    prec_st , rec_st , f1_st  = [], [], []

    for t in thresholds:
        pr, rr, fr, ps, rs, fs = _macro_metrics_at_threshold(
            sample_scores, sample_labels, sample_graphs, sample_names, t
        )
        prec_raw.append(pr);  rec_raw.append(rr);  f1_raw.append(fr)
        prec_st.append(ps);   rec_st.append(rs);   f1_st.append(fs)

    csv_path = os.path.join(log_dir, "macro_threshold_curve.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("threshold,prec_raw,rec_raw,f1_raw,prec_st,rec_st,f1_st\n")
        for row in zip(thresholds, prec_raw, rec_raw, f1_raw, prec_st, rec_st, f1_st):
            fh.write(",".join(f"{v:.6f}" for v in row) + "\n")
    print(f"Wrote threshold‑curve CSV to {csv_path}")

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, prec_raw, label='Raw Precision')
    plt.plot(thresholds, rec_raw,  label='Raw Recall')
    plt.plot(thresholds, f1_raw,   label='Raw F1')
    plt.plot(thresholds, prec_st, '--', label='Steiner Precision')
    plt.plot(thresholds, rec_st,  '--', label='Steiner Recall')
    plt.plot(thresholds, f1_st,   '--', label='Steiner F1')
    plt.xlabel('Threshold');  plt.ylabel('Score')
    plt.title('MACRO Precision / Recall / F1 vs Threshold')
    plt.legend();  plt.grid(True, alpha=0.3);  plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'macro_precision_recall_f1_vs_threshold.png'))
    plt.close()

    # ------------------------------------------------------------------
    # MICRO (pooled) curves --------------------------------------------
    # ------------------------------------------------------------------
    all_scores = np.concatenate(sample_scores, axis=0)
    all_labels = np.concatenate(sample_labels, axis=0)

    # ROC ---------------------------------------------------------------
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'Micro ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MICRO ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'micro_roc_curve.png'))
    plt.close()

    # Precision-Recall --------------------------------------------------
    precision_mi, recall_mi, _ = precision_recall_curve(all_labels, all_scores)
    pr_auc_mi = auc(recall_mi, precision_mi)
    plt.figure(figsize=(6, 6))
    plt.plot(recall_mi, precision_mi, label=f'Micro PR (AUC = {pr_auc_mi:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('MICRO Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'micro_precision_recall_curve.png'))
    plt.close()

    # ------------------------------------------------------------------
    # Score distributions -----------------------------------------------
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.histplot(gold_scores, label='Gold', kde=True, stat='density', bins=50, alpha=0.6)
    sns.histplot(non_gold_scores, label='Non-Gold', kde=True, stat='density', bins=50, alpha=0.6)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.title('Score Distributions: Gold vs Non-Gold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'score_distributions.png'))
    plt.close()

    print(f"Saved plots (macro & micro) to {log_dir}")

# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():  # noqa: D401
    args = cli()
    # Display effective reranker
    print(f"Evaluating with frozen embeddings for dataset {args.dataset}, reranker_type {args.reranker_type}")
    print(f"Dev embeddings dir: {args.dev_embeddings_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Top-k values: {args.k}")
    print(f"Log file: {args.log_file}")

    # Load dev embeddings and metadata ---------------------------------
    dev_embeddings, dev_metadata = load_embeddings_and_metadata(
        args.dev_embeddings_dir, args.dataset, args.reranker_type, split="dev"
    )

    # Create dev dataset and collect graphs -----------------------------
    dev_set = []
    dev_graphs = []
    dev_meta = []  # keep per-sample info aligned with dev_set order
    for emb in dev_embeddings.values():
        q = emb['query']
        names = emb['node_names']  # noqa: F841 (kept for clarity)
        embeddings = emb['embeddings']
        positives = emb['positives']
        G = emb['G']
        dev_set.append(graph_to_data_with_embeddings(G, q, positives, embeddings))
        dev_graphs.append(G)
        dev_meta.append({
            "query": q,
            "sample_id": emb.get("sample_id")
        })
    print(f"Created {len(dev_set)} data samples")

    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)

    # Create model ------------------------------------------------------
    model = GraphColumnRetrieverFrozen(
        embed_dim=dev_metadata['embed_dim'],
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
    ).to(DEVICE)

    # Load checkpoint ---------------------------------------------------
    state = torch.load(args.checkpoint, map_location=DEVICE)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    print(f"✓ Loaded checkpoint {args.checkpoint}")

    # Evaluate and collect scores --------------------------------------
    (stats_lists, stats_st_lists,
        sample_scores, sample_labels, sample_names,
        gold_scores, non_gold_scores) = evaluate_topks_and_collect_scores(
            dev_loader, model, args.k, dev_graphs
        )

    # Optionally log low-recall samples --------------------------------
    if args.log_low_recall:
        # Optional MongoDB connection for enrichment
        mongo_client = None
        mongo_collection = None
        try:
            from pymongo import MongoClient  # type: ignore
            mongo_client = MongoClient(args.mongo_uri)
            mongo_db = mongo_client[args.mongo_db]
            if args.dataset.lower() == "bird":
                collection_name = "dev_samples"
            elif args.dataset.lower() == "spider":
                collection_name = "spider_dev_samples"
            elif args.dataset.lower() == "spider2":
                collection_name = "spider2_lite_samples"
            else:
                collection_name = "dev_samples"
            mongo_collection = mongo_db[collection_name]
            print(f"Using MongoDB collection: {collection_name}")
        except Exception as e:
            print(f"[WARN] MongoDB enrichment disabled ({e})")
            mongo_client = None
            mongo_collection = None

        k_max = max(args.k)
        low_recall_logs: List[Dict[str, Any]] = []
        for i, (scores_np, labels_np, names, G) in enumerate(zip(sample_scores, sample_labels, sample_names, dev_graphs)):
            # Build predictions and golds
            order = np.argsort(scores_np)[::-1]
            preds = [names[j] for j in order]
            golds = [names[j] for j, lab in enumerate(labels_np) if lab == 1]
            # Steiner nodes for top-k
            st_nodes = list(get_steiner_subgraph(G, preds[:k_max]).nodes())
            missing = list(set(golds) - set(st_nodes))
            if missing:
                # prepare score lookup
                col2score = {n: float(scores_np[j]) for j, n in enumerate(names)}
                # Optional mongo enrichment
                sql_query = "unknown"; db_id = "unknown"; db_type = "unknown"
                if mongo_collection is not None:
                    try:
                        sid = dev_meta[i].get("sample_id")
                        if sid is not None:
                            mongo_doc = mongo_collection.find_one({"_id": sid})
                            if mongo_doc:
                                sql_query = mongo_doc.get("SQL", sql_query)
                                db_id = mongo_doc.get("db_id", db_id)
                                db_type = mongo_doc.get("db_type", db_type)
                    except Exception as _e:
                        pass
                # log entry aligned with Qwen reranker script
                entry: Dict[str, Any] = {
                    "question": dev_meta[i]["query"],
                    "sample_id": dev_meta[i].get("sample_id"),
                    "sql_query": sql_query,
                    "db_id": db_id,
                    "db_type": db_type,
                    "schema_length": len(names),
                    "missing_cols": [
                        {
                            "column_desc": make_desc_from_graph(G, m),
                            "score": col2score.get(m),
                            "ranking": (preds.index(m) + 1) if m in preds else len(preds) + 1,
                        }
                        for m in missing
                    ],
                }
                if args.log_higher_than_missing:
                    present_miss = [m for m in missing if m in preds]
                    min_missing_rank = min((preds.index(m) + 1) for m in present_miss) if present_miss else None
                    if min_missing_rank is not None:
                        higher_ranked_cols = []
                        for rank_idx, col_name in enumerate(preds, start=1):
                            if rank_idx < min_missing_rank:
                                higher_ranked_cols.append({
                                    "column_desc": make_desc_from_graph(G, col_name),
                                    "score": col2score.get(col_name),
                                    "ranking": rank_idx,
                                })
                        if higher_ranked_cols:
                            entry["higher_ranked_cols"] = higher_ranked_cols
                low_recall_logs.append(entry)
        out_path = args.log_file.parent / f"low_recall_k{max(args.k)}.json"
        os.makedirs(out_path.parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(low_recall_logs, f, indent=2, ensure_ascii=False)
        print(f"Wrote low-recall logs to {out_path.resolve()} ({len(low_recall_logs)} samples)")
        if mongo_client is not None:
            try:
                mongo_client.close()
            except Exception:
                pass

    # Calculate average top‑k stats ------------------------------------
    stats_avg = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in stats_lists.items()}
    stats_st_avg = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in stats_st_lists.items()}

    print("\nEvaluation results:")
    for k in sorted(stats_avg.keys()):
        sp, sr = stats_st_avg[k]["cp"], stats_st_avg[k]["cr"]
        print(f"@{k}: Precision={stats_avg[k]['prec']:.4f}  "
              f"Recall_col={stats_avg[k]['rec_col']:.4f}  "
              f"Recall_tab={stats_avg[k]['rec_tab']:.4f} | "
              f"Steiner P={sp:.3f} R={sr:.3f}")

    # ------------------------------------------------------------------
    # MICRO (pooled) AUC numbers --------------------------------------
    # ------------------------------------------------------------------
    all_scores = np.concatenate(sample_scores, axis=0)
    all_labels = np.concatenate(sample_labels, axis=0)
    try:
        auc_roc = roc_auc_score(all_labels, all_scores)
        precision_mi, recall_mi, _ = precision_recall_curve(all_labels, all_scores)
        auc_pr = auc(recall_mi, precision_mi)
        print(f"\nColumn-level ROC AUC (micro): {auc_roc:.4f}")
        print(f"Column-level PR AUC  (micro): {auc_pr:.4f}")
    except Exception as e:  # pragma: no cover - defensive
        print(f"Could not calculate pooled AUC: {e}")
        auc_roc = None
        auc_pr = None

    # Summary stats for score dists ------------------------------------
    gold_mean = float(np.mean(gold_scores)) if gold_scores else 0.0
    gold_std = float(np.std(gold_scores)) if gold_scores else 0.0
    non_gold_mean = float(np.mean(non_gold_scores)) if non_gold_scores else 0.0
    non_gold_std = float(np.std(non_gold_scores)) if non_gold_scores else 0.0
    score_gap = gold_mean - non_gold_mean
    print(f"\nGold mean:     {gold_mean:.4f} ± {gold_std:.4f}  (n={len(gold_scores)})")
    print(f"Non-gold mean: {non_gold_mean:.4f} ± {non_gold_std:.4f}  (n={len(non_gold_scores)})")
    print(f"Score gap:     {score_gap:.4f}")

    # Save to log file --------------------------------------------------
    os.makedirs(args.log_file.parent, exist_ok=True)
    results = {
        "topk_stats": stats_avg,
        "steiner_stats": stats_st_avg,
        "auc_roc_micro": auc_roc,
        "auc_pr_micro": auc_pr,
        "gold_mean": gold_mean,
        "gold_std": gold_std,
        "non_gold_mean": non_gold_mean,
        "non_gold_std": non_gold_std,
        "score_gap": score_gap,
        # NOTE: sample_scores/labels omitted from JSON by default (very large); add if needed.
    }
    with open(args.log_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Evaluation results saved to {args.log_file.resolve()}")

    # import sys; sys.exit()

    # Draw figures & threshold analysis --------------------------------
    plot_threshold_analysis(
        sample_scores, sample_labels, dev_graphs, sample_names,
        gold_scores, non_gold_scores, str(args.log_file.parent)
    )


if __name__ == "__main__":
    main()
