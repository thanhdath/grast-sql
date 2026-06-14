#!/usr/bin/env bash
# Reproduce GRAST-SL schema linking on Spider / BIRD / Spider 2.0-Snow.
# Pipeline: (coarse retrieval) -> vLLM reranker node features -> deep-attention GNN -> table-aware Steiner.
#
#   ./reproduce.sh bird        # or: spider | spider2
#
# Override paths via env: DATA_DIR, RERANKER, CKPT, RETRIEVER, PORT, GPU.
set -euo pipefail

DATASET="${1:-bird}"
DATA_DIR="${DATA_DIR:-/home/datht/grast-sql/data}"
RERANKER="${RERANKER:-/home/datht/grast-sql/train_and_evaluate/output/finetuned-reranker-qwen3-0.6B-full-v2/merged_model}"
CKPT="${CKPT:-/home/datht/grast-sql/output/gnn_ckpts_e2e/hybrid_attnpost2_warm_l3_h2048/best_pr_auc_epoch_02.pt}"
RETRIEVER="${RETRIEVER:-/home/datht/grast-sql/gte-modernbert-base/checkpoints/gte-modernbert-stage1}"
PORT="${PORT:-8000}"; GPU="${GPU:-0}"
PY="${PY:-python}"

case "$DATASET" in
  bird)    PKL="$DATA_DIR/bird_dev_samples_graph.pkl";              TARGET_R=0.96; EXTRA="" ;;
  spider)  PKL="$DATA_DIR/spider_dev_samples_graph.pkl";            TARGET_R=0.98; EXTRA="" ;;
  spider2) PKL="$DATA_DIR/spider2_snow_256_eval.pkl";              TARGET_R=0.90; EXTRA="" ;;  # Stage-I pool already built; P@R0.90 ~= paper 0.251
  *) echo "unknown dataset '$DATASET' (use bird|spider|spider2)"; exit 1 ;;
esac
LOG_DIR="logs/reproduce_${DATASET}"

# 1) vLLM embedding server (reranker node features) -------------------------------
if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[reproduce] starting vLLM on GPU ${GPU} (port ${PORT}) ..."
  CUDA_VISIBLE_DEVICES="${GPU}" vllm serve "$RERANKER" --port "$PORT" --task embed \
    --max-model-len 8192 --gpu-memory-utilization 0.85 > /tmp/vllm_grast.log 2>&1 &
  for _ in $(seq 1 60); do curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && break; sleep 5; done
fi
curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 || { echo "vLLM failed to start (see /tmp/vllm_grast.log)"; exit 1; }
echo "[reproduce] vLLM up; running ${DATASET} ..."

# 2) end-to-end eval (GNN on CPU; embeddings from vLLM) ---------------------------
CUDA_VISIBLE_DEVICES="" "$PY" -u evaluate_on_the_fly.py \
  --dataset "$DATASET" --split dev --evaluation_mode end2end \
  --reranker_type qwen --encoder_path "$RERANKER" --checkpoint "$CKPT" --pkl_path "$PKL" \
  --hidden_dim 2048 --num_layers 3 --k 10 20 30 --batch_size 128 $EXTRA \
  --log_dir "$LOG_DIR" --pred_collection "grast_reproduce_${DATASET}"

# 3) operating-point precision at matched recall (GNN + table-aware Steiner) ------
"$PY" - "$LOG_DIR" "$PKL" "$TARGET_R" <<'PY'
import json, pickle, sys
import numpy as np
from modules.steiner_tree_spanner.steiner import get_steiner_subgraph
log_dir, pkl, target = sys.argv[1], sys.argv[2], float(sys.argv[3])
sd = json.load(open(f"{log_dir}/score_distribution_data_" +
                    [f.split("score_distribution_data_")[1] for f in __import__("os").listdir(log_dir)
                     if f.startswith("score_distribution_data_")][0]))
ps = sd["per_sample_scores"]
sid2G = {str(s): G for q, G, g, s in pickle.load(open(pkl, "rb"))}
items = [(sid2G[str(k)], r["names"], np.asarray(r["scores"], float), set(r["gold_columns"]))
         for k, r in ps.items() if str(k) in sid2G and r["gold_columns"]]
def macro(tau):
    P = R = 0.0
    for G, names, sc, gold in items:
        base = set(np.array(names, dtype=object)[1/(1+np.exp(-sc)) >= tau].tolist())
        sel = (set(get_steiner_subgraph(G, list(base)).nodes()) | base) if base else base
        tp = len(sel & gold); P += tp/max(len(sel),1); R += tp/max(len(gold),1)
    n = len(items); return R/n, P/n
allp = np.concatenate([1/(1+np.exp(-np.asarray(r["scores"]))) for r in ps.values() if r["gold_columns"]])
best = None
for t in np.unique(np.quantile(allp, np.linspace(0,1,400)))[::-1]:
    R, P = macro(t)
    if best is None or abs(R-target) < abs(best[0]-target): best = (R, P, t)
    if R > target+0.03: break
R, P, t = best
roc = json.load(open(f"{log_dir}/" + [f for f in __import__('os').listdir(log_dir) if f.startswith('on_the_fly_eval_')][0]))["roc_auc_micro"]
print(f"\n=== {len(items)} samples | micro ROC={roc:.3f} | GNN+Steiner @ R={R:.3f}: P={P:.3f} (tau={t:.4f}) ===")
PY
echo "[reproduce] done -> $LOG_DIR"
