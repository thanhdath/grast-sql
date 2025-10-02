#!/usr/bin/env python3
"""
train_with_frozen_embeddings.py  (AUC‑Enhanced + Best ROC/PR Tracking)
──────────────────────────────────────────────────────────────────────
Training script that loads pre‑initialized embeddings and runs graph transformers.
"""

import random
import argparse
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Import Graph Reranker components
from modules.graph_reranker.data import (
    load_embeddings_and_metadata,
    create_dataset_from_embeddings,
)
from modules.graph_reranker.model import GraphColumnRetrieverFrozen

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_K_VALUES = (10, 20)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(loader: DataLoader, model: nn.Module, device: torch.device, top_k: Sequence[int]) -> Tuple[Dict[int, Dict[str, float]], List[float], List[float]]:
    """Evaluate model performance on a loader.

    Returns
    -------
    stats_avg : dict
        {k: {prec, rec_col, rec_tab}} averaged across samples.
    all_scores : list of float
        Concatenated logits for ALL nodes across ALL samples (micro pool).
    all_labels : list of float
        Corresponding gold labels (0/1) for each score.
    """
    model.eval()
    stats: Dict[int, Dict[str, List[float]]] = {k: {m: [] for m in ("prec", "rec_col", "rec_tab")} for k in top_k}
    all_scores: List[float] = []
    all_labels: List[float] = []

    for data in loader:  # dev loader uses bs=1
        data = data.to(device)
        with autocast():
            logits = model(data)  # (N,)
        names = data.orig_names[0]
        truths = [names[i] for i, y in enumerate(data.y) if y == 1]
        preds_idx = logits.argsort(descending=True)
        L = logits.size(0)

        # Collect scores/labels for micro AUC
        all_scores.extend(logits.detach().cpu().tolist())
        all_labels.extend(data.y.detach().cpu().tolist())

        for k in top_k:
            k_eff = min(k, L)
            preds = [names[i] for i in preds_idx[:k_eff]]
            tp = len(set(preds) & set(truths))
            stats[k]["prec"].append(tp / k_eff if k_eff else 0)
            stats[k]["rec_col"].append(tp / len(truths) if truths else 0)
            stats[k]["rec_tab"].append(
                len({p.split('.')[0] for p in preds} &
                    {t.split('.')[0] for t in truths}) /
                len({t.split('.')[0] for t in truths}) if truths else 0)

    # Average
    stats_avg: Dict[int, Dict[str, float]] = {k: {m: (sum(v) / len(v) if v else 0.0) for m, v in d.items()} for k, d in stats.items()}
    return stats_avg, all_scores, all_labels


@torch.no_grad()
def evaluate_average_loss(loader: DataLoader, model: nn.Module, criterion: nn.Module, device: torch.device) -> float:
    """Compute average loss over a data loader using the provided criterion."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for data in loader:
        data = data.to(device)
        with autocast():
            logits = model(data)
            if logits.numel() == 0 or logits.size(0) != data.y.size(0):
                continue
            loss = criterion(logits, data.y.float())
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(1, num_batches) 

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train Graph Column Retriever with Frozen Embeddings")
    p.add_argument("--embeddings_dir", type=Path, required=True,
                   help="Directory containing pre-initialized train embeddings")
    p.add_argument("--dev_embeddings_dir", type=Path, required=False, default=None,
                   help="Directory containing pre-initialized dev embeddings (optional). If not provided, dev evaluation will be skipped and save the last checkpoint only.")
    p.add_argument("--dataset", choices=["bird", "spider", "spider2"], required=True,
                   help="Dataset name (bird, spider, or spider2)")
    p.add_argument("--reranker_type", choices=["standard", "layerwise", "qwen"], required=True,
                   help="Reranker type used for embeddings (standard, layerwise, or qwen)")
    p.add_argument("--num_layers", type=int, default=3,
                   help="Number of GNN layers")
    p.add_argument("--hid_dim", type=int, default=2048,
                   help="GNN hidden dimension")
    p.add_argument("--num_epochs", type=int, default=40,
                   help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Training batch size")
    p.add_argument("--learning_rate", type=float, default=5e-5,
                   help="Learning rate")
    p.add_argument("--output_dir", type=Path,
                   default=Path("output/frozen_encoder_training"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training with frozen embeddings for dataset {args.dataset}, reranker_type {args.reranker_type}")
    print(f"Output directory: {args.output_dir}")

    # Load train embeddings and metadata
    train_embeddings, train_metadata = load_embeddings_and_metadata(
        args.embeddings_dir, args.dataset, args.reranker_type, split="train"
    )
    # Load dev embeddings and metadata if provided
    dev_embeddings: Optional[Dict[str, Any]] = None
    dev_metadata: Optional[Dict[str, Any]] = None
    if args.dev_embeddings_dir:
        dev_embeddings, dev_metadata = load_embeddings_and_metadata(
            args.dev_embeddings_dir, args.dataset, args.reranker_type, split="dev"
        )
        # Create datasets
        assert dev_embeddings is not None
        dev_set = create_dataset_from_embeddings(dev_embeddings)
        dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False)
    else:
        print("[INFO] No dev embeddings directory provided. Skipping dev evaluation.")
        dev_loader = None

    # Create datasets
    train_set = create_dataset_from_embeddings(train_embeddings)

    # Create model
    model = GraphColumnRetrieverFrozen(
        embed_dim=train_metadata['embed_dim'],
        hid_dim=args.hid_dim,
        num_layers=args.num_layers
    ).to(DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()
    writer = SummaryWriter(f"runs/frozen_encoder_training_{args.dataset}_{args.reranker_type}")

    best_rec10 = 0.0
    best_pr_auc = float('-inf')
    best_pr_auc_epoch = 0
    best_pr_checkpoint_path: Optional[Path] = None

    best_roc_auc = float('-inf')
    best_roc_auc_epoch = 0
    best_roc_checkpoint_path: Optional[Path] = None

    def save_checkpoint(tag: str, current_epoch: int) -> Path:
        """Helper to save a checkpoint and return its path."""
        ckpt_path = args.output_dir / f"best_{tag}_epoch_{current_epoch:02d}.pt"
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_rec10': best_rec10,
            'best_pr_auc': best_pr_auc,
            'best_pr_auc_epoch': best_pr_auc_epoch,
            'best_roc_auc': best_roc_auc,
            'best_roc_auc_epoch': best_roc_auc_epoch,
            'args': vars(args)
        }, ckpt_path)
        print(f"[INFO] Saved new best {tag.upper()} checkpoint to {ckpt_path}")
        return ckpt_path

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        train_all_scores: List[float] = []
        train_all_labels: List[float] = []

        for step, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            data = data.to(DEVICE)
            with autocast():
                logits = model(data)
                if logits.numel() == 0 or logits.size(0) != data.y.size(0):
                    print(f"Skipping batch {step}: logits {logits.size()}, target {data.y.size()}")
                    continue
                loss = criterion(logits, data.y.float())

            # Collect train scores/labels for train ROC AUC
            train_all_scores.extend(logits.detach().cpu().tolist())
            train_all_labels.extend(data.y.detach().cpu().tolist())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()

        # ------------------------------------------------------------------
        # Dev evaluation (top‑K + AUC metrics)
        # ------------------------------------------------------------------
        if dev_loader:
            dev_stats, all_scores, all_labels = evaluate(dev_loader, model, DEVICE, TOP_K_VALUES)
            train_avg_loss = total_loss / max(1, len(train_loader))

            # Compute dev loss
            dev_avg_loss = evaluate_average_loss(dev_loader, model, criterion, DEVICE)

            # Micro ROC / PR AUC across all columns (Dev)
            try:
                dev_roc_auc = roc_auc_score(all_labels, all_scores)
            except Exception as e:  # e.g., only one class present
                print(f"[WARN] ROC AUC undefined this epoch: {e}")
                dev_roc_auc = float('nan')

            try:
                precision, recall, _ = precision_recall_curve(all_labels, all_scores)
                dev_pr_auc = auc(recall, precision)
            except Exception as e:
                print(f"[WARN] PR AUC undefined this epoch: {e}")
                dev_pr_auc = float('nan')

            # Micro ROC AUC across all columns (Train)
            try:
                train_roc_auc = roc_auc_score(train_all_labels, train_all_scores)
            except Exception as e:
                print(f"[WARN] Train ROC AUC undefined this epoch: {e}")
                train_roc_auc = float('nan')

            # Track best PR & ROC AUC
            new_best_pr = not math.isnan(dev_pr_auc) and dev_pr_auc > best_pr_auc
            new_best_roc = not math.isnan(dev_roc_auc) and dev_roc_auc > best_roc_auc

            if new_best_pr:
                best_pr_auc = dev_pr_auc
                best_pr_auc_epoch = epoch
                # Remove old checkpoint if exists
                if best_pr_checkpoint_path and best_pr_checkpoint_path.exists():
                    best_pr_checkpoint_path.unlink()
                best_pr_checkpoint_path = save_checkpoint("pr_auc", epoch)

            if new_best_roc:
                best_roc_auc = dev_roc_auc
                best_roc_auc_epoch = epoch
                # Remove old checkpoint if exists
                if best_roc_checkpoint_path and best_roc_checkpoint_path.exists():
                    best_roc_checkpoint_path.unlink()
                best_roc_checkpoint_path = save_checkpoint("roc_auc", epoch)

            # Track best Recall@10 for backwards compatibility
            best_rec10 = max(best_rec10, dev_stats[10]["rec_col"])

            # ------------------- TensorBoard logging -------------------------
            writer.add_scalar("Loss/train", train_avg_loss, epoch)
            writer.add_scalar("Loss/dev", dev_avg_loss, epoch)
            for k in TOP_K_VALUES:
                writer.add_scalar(f"Dev/Precision@{k}", dev_stats[k]["prec"], epoch)
                writer.add_scalar(f"Dev/Recall_col@{k}", dev_stats[k]["rec_col"], epoch)
                writer.add_scalar(f"Dev/Recall_tab@{k}", dev_stats[k]["rec_tab"], epoch)
            writer.add_scalar("Dev/ROC_AUC", dev_roc_auc, epoch)
            writer.add_scalar("Dev/PR_AUC", dev_pr_auc, epoch)
            writer.add_scalar("Train/ROC_AUC", train_roc_auc, epoch)

            # --------------------------- Console -----------------------------
            print(f"\nEpoch {epoch:02d} | train_loss {train_avg_loss:.4f} | dev_loss {dev_avg_loss:.4f}")
            for k in TOP_K_VALUES:
                print(f"@{k}: Prec={dev_stats[k]['prec']:.4f}  "
                      f"Rec-col={dev_stats[k]['rec_col']:.4f}  "
                      f"Rec-tab={dev_stats[k]['rec_tab']:.4f}")
            if math.isnan(train_roc_auc):
                print("Train ROC AUC: N/A")
            else:
                print(f"Train ROC AUC: {train_roc_auc:.4f}")
            if math.isnan(dev_roc_auc):
                print("Dev ROC AUC: N/A")
            else:
                print(f"Dev ROC AUC: {dev_roc_auc:.4f}")
            if math.isnan(dev_pr_auc):
                print("Dev PR  AUC: N/A")
            else:
                print(f"Dev PR  AUC: {dev_pr_auc:.4f}")
        else:
            # If no dev set, just log train loss and save a final checkpoint
            train_avg_loss = total_loss / max(1, len(train_loader))
            writer.add_scalar("Loss/train", train_avg_loss, epoch)
            print(f"\nEpoch {epoch:02d} | train_loss {train_avg_loss:.4f}")

    # ------------------------------------------------------------------
    # Training done — final summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Training complete after {args.num_epochs} epochs.")
    print(f"Best column recall @10 = {best_rec10:.4f}")
    if dev_loader:
        if math.isinf(best_roc_auc) or math.isnan(best_roc_auc):
            print("Best Dev ROC AUC: N/A (metric undefined across epochs)")
        else:
            print(f"Best Dev ROC AUC = {best_roc_auc:.4f} (achieved at epoch {best_roc_auc_epoch})")
        if math.isinf(best_pr_auc) or math.isnan(best_pr_auc):
            print("Best Dev PR AUC: N/A (metric undefined across epochs)")
        else:
            print(f"Best Dev PR AUC = {best_pr_auc:.4f} (achieved at epoch {best_pr_auc_epoch})")
    else:
        # Save final checkpoint when no dev set is used
        last_ckpt = args.output_dir / f"last_epoch_{args.num_epochs:02d}.pt"
        torch.save({
            'epoch': args.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args)
        }, last_ckpt)
        print(f"[INFO] Saved last checkpoint to {last_ckpt}")
    print("=" * 60)

    writer.close()


if __name__ == "__main__":
    main()
