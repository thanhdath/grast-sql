#!/bin/bash

# Qwen Reranker End2End Evaluation Only
# - Datasets: BIRD, SPIDER
# - Mode: end2end only (no encoder_only)
# - Reranker: qwen (num_layers=3, hidden_dim=2048)

set -euo pipefail

echo "=========================================="
echo "Qwen Reranker End2End Evaluation"
echo "=========================================="

mkdir -p logs

# Discover latest best checkpoints
BIRD_QWEN_ENCODER="/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-0.6b-bird/merged_model"
BIRD_QWEN_CKPT_DIR="/home/datht/graph-schema/output/bird/grast_bird_reranker-qwen3-0.6B-finetuned-with-table-meaning/layer-3-hidden-2048"
BEST_BIRD_QWEN_CKPT=$(ls -t "$BIRD_QWEN_CKPT_DIR" | grep best_roc_auc_epoch_ | head -n 1 || true)

SPIDER_QWEN_ENCODER="/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-0.6B-spider/merged_model"
SPIDER_QWEN_CKPT_DIR="/home/datht/graph-schema/output/spider/grast_spider_reranker-qwen3-0.6B-finetuned-with-table-meaning/layer-3-hidden-2048"
BEST_SPIDER_QWEN_CKPT=$(ls -t "$SPIDER_QWEN_CKPT_DIR" | grep best_roc_auc_epoch_ | head -n 1 || true)

if [[ -z "${BEST_BIRD_QWEN_CKPT:-}" ]]; then
  echo "[WARN] No best checkpoint found for BIRD at $BIRD_QWEN_CKPT_DIR"
else
  echo "[INFO] BIRD best checkpoint: $BEST_BIRD_QWEN_CKPT"
fi

if [[ -z "${BEST_SPIDER_QWEN_CKPT:-}" ]]; then
  echo "[WARN] No best checkpoint found for SPIDER at $SPIDER_QWEN_CKPT_DIR"
else
  echo "[INFO] SPIDER best checkpoint: $BEST_SPIDER_QWEN_CKPT"
fi

echo ""
echo "🐦 BIRD (Qwen 0.6B)"
echo "------------------------------------------"
python evaluate_on_the_fly_embeddings.py \
  --dataset bird \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 2048 \
  --num_layers 3 \
  --encoder_path "$BIRD_QWEN_ENCODER" \
  --checkpoint "$BIRD_QWEN_CKPT_DIR/$BEST_BIRD_QWEN_CKPT" \
  --pkl_path data/bird_dev_samples_graph.pkl \
  --batch_size 64 \
  --threshold 0 \
  --log_dir logs/bird_dev_qwen_end2end

echo ""
echo "🕷️ SPIDER (Qwen 0.6B)"
echo "------------------------------------------"
python evaluate_on_the_fly_embeddings.py \
  --dataset spider \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 2048 \
  --num_layers 3 \
  --encoder_path "$SPIDER_QWEN_ENCODER" \
  --checkpoint "$SPIDER_QWEN_CKPT_DIR/$BEST_SPIDER_QWEN_CKPT" \
  --pkl_path data/spider_dev_samples_graph.pkl \
  --batch_size 64 \
  --threshold 0 \
  --log_dir logs/spider_dev_qwen_end2end

echo ""
echo "=========================================="
echo "✅ Qwen end2end evaluations completed"
echo "📁 Results saved to logs/ directory"
echo "==========================================" 