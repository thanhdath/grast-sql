#!/usr/bin/env python3
"""
evaluate_end2end.py - Evaluate end2end trained models
─────────────────────────────────────────────────────
Evaluate a trained End2EndGraphColumnRetriever on a dev set. This script loads
the model checkpoint and evaluates it on the original dataset format (not frozen
embeddings), providing comprehensive metrics including top-k evaluation, steiner
tree analysis, and threshold analysis.

Usage example:
python evaluate_end2end.py \
  --dataset bird --reranker_type standard \
  --checkpoint output/end2end_training/best_roc_auc_epoch_10.pt \
  --k 5 10 20
"""

import argparse
import json
import os
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
import pickle

# Import from train_end2end.py
from train_end2end import (
    End2EndGraphColumnRetriever,
    load_dataset,
    graph_to_data,
    DATASET_PKL,
    DEVICE,
    EDGE_TYPE_MAP,
    TOP_K_VALUES
)
from init_embeddings import make_desc

# ---------------------------------------------------------------------------
# CLI ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate End2EndGraphColumnRetriever")
    p.add_argument("--dataset", choices=["bird", "spider"], required=True,
                   help="Dataset name (bird or spider)")
    p.add_argument("--reranker_type", choices=["standard", "layerwise"], required=True,
                   help="Reranker type used for training (standard or layerwise)")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to the pretrained encoder model")
    p.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    p.add_argument("--hid_dim", type=int, default=1024, help="GNN hidden dimension")
    p.add_argument("--cut_layer", type=int, default=39,
                   help="Which hidden layer to use for layerwise reranker (-1 => last)")
    p.add_argument("--frozen_encoder", action="store_true",
                   help="Whether encoder was frozen during training")
    p.add_argument("--checkpoint", type=Path, required=True, 
                   help="Path to trained model checkpoint (*.pt)")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation (default: 1)")
    p.add_argument("--encoding_batch_size", type=int, default=8, help="Batch size for encoding")
    p.add_argument("--k", nargs="+", type=int, default=[10, 20, 30, 40], 
                   help="Top-k values for evaluation (default: 10 20 30 40)")
    p.add_argument("--log-file", type=Path, default=Path("logs/eval_end2end.json"),
                   help="Path to save evaluation results as JSON")
    p.add_argument("--disable_amp", action="store_true", help="Disable automatic mixed precision")
    p.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    p.add_argument("--max_samples", type=int, default=99999, help="Maximum number of samples to evaluate (default: 20 for debugging)")
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
# Data loading --------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_dev_data_and_graphs(dataset: str, max_samples: int = 20) -> Tuple[List, List[nx.DiGraph], List[Dict]]:
    """Load dev data and corresponding graphs with original metadata."""
    dev_pkl = DATASET_PKL[dataset]["dev"]
    triples = pickle.load(open(dev_pkl, "rb"))
    
    dev_data = []
    dev_graphs = []
    dev_metadata = []
    
    # Add progress bar for data loading
    pbar = tqdm(triples, desc=f"Loading dev data (max {max_samples})")
    
    for i, (q, G, positives, *rest) in enumerate(pbar):
        if max_samples is not None and i >= max_samples:
            break
        if not positives:
            continue
        
        # Extract additional metadata if available
        sql = rest[0] if len(rest) > 0 else ""
        metadata = {
            "question": q,
            "ground_truth_columns": positives,
            "ground_truth_sql": sql,
            "sample_index": i
        }
        
        dev_data.append(graph_to_data(G, q, positives))
        dev_graphs.append(G)
        dev_metadata.append(metadata)
        
        # Update progress bar
        pbar.set_postfix({
            'Loaded': len(dev_data),
            'Max': max_samples,
            'Skipped': i + 1 - len(dev_data)
        })
    
    pbar.close()
    print(f"Loaded {len(dev_data)} dev samples (max requested: {max_samples})")
    return dev_data, dev_graphs, dev_metadata

# ---------------------------------------------------------------------------
# Eval core -----------------------------------------------------------------
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_topks_and_collect_scores(
    loader: DataLoader,
    model: torch.nn.Module,
    ks: List[int],
    graphs: List[nx.DiGraph],
    metadata: List[Dict],
    encoding_batch_size: int = 8,
    disable_amp: bool = False,
    fp16: bool = False,
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
    failed_samples    : list[dict]        – samples with recall < 1.0
    """
    model.eval()
    ks = sorted(set(ks))

    stats_lists = {k: {m: [] for m in ("prec", "rec_col", "rec_tab")} for k in ks}
    stats_st_lists = {k: {m: [] for m in ("cp", "cr")} for k in ks}

    sample_scores: List[np.ndarray] = []
    sample_labels: List[np.ndarray] = []
    sample_names: List[List[str]] = []

    gold_scores, non_gold_scores = [], []
    failed_samples = []

    # Add progress bar for evaluation
    pbar = tqdm(loader, desc="Evaluating samples", total=len(loader))
    
    for i, data in enumerate(pbar):  # bs = 1
        data = data.to(DEVICE)
        graph = graphs[i]
        sample_metadata = metadata[i]

        if disable_amp and not fp16:
            logits = model(data, encoding_batch_size)
        else:
            with autocast():
                logits = model(data, encoding_batch_size)
        
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

        # Check for failed samples (recall < 1.0) at k=10, 20, 30, 40
        k_checks = [10, 20, 30, 40]
        for k_check in k_checks:
            k_eff = min(k_check, len(preds_all))
            preds_k = preds_all[:k_eff]
            tp = len(set(preds_k) & set(truths))
            recall = tp / len(truths) if truths else 1.0
            
            if recall < 1.0:
                failed_sample = {
                    "sample_index": sample_metadata["sample_index"],
                    "question": sample_metadata["question"],
                    "ground_truth_columns": sample_metadata["ground_truth_columns"],
                    "ground_truth_sql": sample_metadata["ground_truth_sql"],
                    "predicted_columns": preds_k,
                    "k_value": k_check,
                    "recall": recall,
                    "precision": tp / k_eff if k_eff else 0.0,
                    "missing_columns": list(set(truths) - set(preds_k)),
                    "extra_columns": list(set(preds_k) - set(truths))
                }
                failed_samples.append(failed_sample)

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
        
        # Update progress bar with current metrics
        if i < len(ks):
            current_k = sorted(ks)[i % len(ks)]
            current_prec = stats_lists[current_k]["prec"][-1] if stats_lists[current_k]["prec"] else 0.0
            current_rec = stats_lists[current_k]["rec_col"][-1] if stats_lists[current_k]["rec_col"] else 0.0
            pbar.set_postfix({
                f'P@{current_k}': f'{current_prec:.3f}',
                f'R@{current_k}': f'{current_rec:.3f}',
                'Samples': f'{i+1}/{len(loader)}',
                'Failed': len(failed_samples)
            })

    pbar.close()

    return (
        stats_lists,
        stats_st_lists,
        sample_scores,
        sample_labels,
        sample_names,
        gold_scores,
        non_gold_scores,
        failed_samples,
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

    # Add progress bar for threshold analysis
    pbar = tqdm(thresholds, desc="Computing threshold analysis")
    
    for t in pbar:
        pr, rr, fr, ps, rs, fs = _macro_metrics_at_threshold(
            sample_scores, sample_labels, sample_graphs, sample_names, t
        )
        prec_raw.append(pr);  rec_raw.append(rr);  f1_raw.append(fr)
        prec_st.append(ps);   rec_st.append(rs);   f1_st.append(fs)
        
        # Update progress bar with current threshold
        pbar.set_postfix({'Threshold': f'{t:.3f}'})
    
    pbar.close()

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
    print(f"Evaluating end2end model for dataset {args.dataset}, reranker_type {args.reranker_type}")
    print(f"Model path: {args.model_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Top-k values: {args.k}")
    print(f"Max samples: {args.max_samples} (debug mode)" if args.max_samples == 20 else f"Max samples: {args.max_samples}")
    print(f"Log file: {args.log_file}")
    print("="*80)

    # Load dev data and graphs ------------------------------------------
    print("Loading dev data and graphs...")
    dev_data, dev_graphs, dev_metadata = load_dev_data_and_graphs(args.dataset, args.max_samples)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
    print(f"Created DataLoader with {len(dev_data)} samples")

    # Create model ------------------------------------------------------
    print("Creating model...")
    model = End2EndGraphColumnRetriever(
        encoder_name=args.model_path,
        reranker_type=args.reranker_type,
        cut_layer=args.cut_layer,
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
        frozen_encoder=args.frozen_encoder
    ).to(DEVICE)

    # Load checkpoint ---------------------------------------------------
    print("Loading checkpoint...")
    state = torch.load(args.checkpoint, map_location=DEVICE)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    print(f"✓ Loaded checkpoint {args.checkpoint}")

    # Evaluate and collect scores --------------------------------------
    print("Starting evaluation...")
    (stats_lists, stats_st_lists,
        sample_scores, sample_labels, sample_names,
        gold_scores, non_gold_scores, failed_samples) = evaluate_topks_and_collect_scores(
            dev_loader, model, args.k, dev_graphs, dev_metadata,
            args.encoding_batch_size, args.disable_amp, args.fp16
        )

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
    
    # Print failed samples summary
    if failed_samples:
        # Group failed samples by k value
        failed_by_k = {}
        for failed in failed_samples:
            k_val = failed['k_value']
            if k_val not in failed_by_k:
                failed_by_k[k_val] = []
            failed_by_k[k_val].append(failed)
        
        print(f"\n🔍 Failed samples (recall < 1.0): {len(failed_samples)} total")
        for k_val in sorted(failed_by_k.keys()):
            k_failed = failed_by_k[k_val]
            print(f"  k={k_val}: {len(k_failed)} failed samples")
        
        print("\nSample details (first 3 per k value):")
        for k_val in sorted(failed_by_k.keys()):
            k_failed = failed_by_k[k_val]
            print(f"\n  k={k_val} failed samples:")
            for i, failed in enumerate(k_failed[:3]):  # Show first 3 failed samples per k
                print(f"    {i+1}. Sample {failed['sample_index']}:")
                print(f"       Question: {failed['question'][:80]}{'...' if len(failed['question']) > 80 else ''}")
                print(f"       GT Columns: {failed['ground_truth_columns']}")
                print(f"       Pred Columns: {failed['predicted_columns'][:3]}{'...' if len(failed['predicted_columns']) > 3 else ''}")
                print(f"       Missing: {failed['missing_columns']}")
                print(f"       Recall: {failed['recall']:.3f}, Precision: {failed['precision']:.3f}")
            if len(k_failed) > 3:
                print(f"    ... and {len(k_failed) - 3} more failed samples for k={k_val}")
    else:
        print(f"\n✅ All samples achieved perfect recall at all k values!")

    # Save to log file --------------------------------------------------
    os.makedirs(args.log_file.parent, exist_ok=True)
    
    # Group failed samples by k value for JSON output
    failed_by_k = {}
    for failed in failed_samples:
        k_val = failed['k_value']
        if k_val not in failed_by_k:
            failed_by_k[k_val] = []
        failed_by_k[k_val].append(failed)
    
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
        "failed_samples": failed_samples,
        "failed_samples_by_k": failed_by_k,
        "failed_samples_count": len(failed_samples),
        "failed_samples_count_by_k": {str(k): len(samples) for k, samples in failed_by_k.items()},
        "total_samples": len(sample_scores),
        "failure_rate": len(failed_samples) / len(sample_scores) if sample_scores else 0.0,
        "failure_rate_by_k": {str(k): len(samples) / len(sample_scores) if sample_scores else 0.0 for k, samples in failed_by_k.items()},
        "model_config": {
            "dataset": args.dataset,
            "reranker_type": args.reranker_type,
            "model_path": args.model_path,
            "num_layers": args.num_layers,
            "hid_dim": args.hid_dim,
            "frozen_encoder": args.frozen_encoder,
            "cut_layer": args.cut_layer
        }
    }
    with open(args.log_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Evaluation results saved to {args.log_file.resolve()}")
    print(f"📊 Failed samples: {len(failed_samples)}/{len(sample_scores)} ({len(failed_samples)/len(sample_scores)*100:.1f}%)")
    for k_val in sorted(failed_by_k.keys()):
        k_failed = failed_by_k[k_val]
        print(f"   k={k_val}: {len(k_failed)} failed ({len(k_failed)/len(sample_scores)*100:.1f}%)")

    # Draw figures & threshold analysis --------------------------------
    plot_threshold_analysis(
        sample_scores, sample_labels, dev_graphs, sample_names,
        gold_scores, non_gold_scores, str(args.log_file.parent)
    )


if __name__ == "__main__":
    main() 