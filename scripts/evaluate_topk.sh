# ==============================
# Qwen reranker (vLLM) runs
# ==============================

# Ensure logs directory exists
mkdir -p logs
mkdir -p logs/end2end

# BIRD DEV - Top-K 20 with Qwen reranker
echo ""
echo "🐦 BIRD DEV - Top-K 20 (Qwen)"
echo "=============================="

QWEN_BIRD_MODEL=/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-0.6b-bird/merged_model
QWEN_BIRD_CKPT_DIR=output/bird/grast_bird_reranker-qwen3-0.6B-finetuned-with-table-meaning
BEST_QWEN_BIRD_CHECKPOINT=$(ls -t "$QWEN_BIRD_CKPT_DIR" 2>/dev/null | grep best_roc_auc_epoch_ | head -n 1)

if [ -z "$BEST_QWEN_BIRD_CHECKPOINT" ]; then
  echo "WARNING: Could not find best_roc_auc_epoch_* in $QWEN_BIRD_CKPT_DIR"
  echo "Please train GNN with qwen embeddings and set QWEN_BIRD_CKPT_DIR accordingly."
fi

python -u evaluate_on_the_fly_embeddings.py \
  --dataset bird \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 3 \
  --encoder_path "$QWEN_BIRD_MODEL" \
  --checkpoint "$QWEN_BIRD_CKPT_DIR/$BEST_QWEN_BIRD_CHECKPOINT" \
  --pkl_path data/bird_dev_samples_graph.pkl \
  --k 30 \
  --batch_size 128 \
  --log_dir logs/bird_dev_topk20_qwen \
  --pred_collection grast_bird_dev_qwen_finetuned_0.6b > logs/end2end/running_bird_qwen_0.6B.log

# SPIDER DEV - Top-K 20 with Qwen reranker
echo ""
echo "🕷️  SPIDER DEV - Top-K 20 (Qwen)"
echo "================================"

# If you have a fine-tuned Qwen reranker for Spider, set the path here; fallback to hub model
QWEN_SPIDER_MODEL=/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-0.6B-spider/merged_model
QWEN_SPIDER_CKPT_DIR=output/spider/grast_spider_reranker-qwen3-0.6B-finetuned-with-table-meaning
BEST_QWEN_SPIDER_CHECKPOINT=$(ls -t "$QWEN_SPIDER_CKPT_DIR" 2>/dev/null | grep best_pr_auc_epoch_ | head -n 1)

if [ -z "$BEST_QWEN_SPIDER_CHECKPOINT" ]; then
  echo "WARNING: Could not find best_* checkpoint in $QWEN_SPIDER_CKPT_DIR"
  echo "Please train GNN with qwen embeddings for Spider and set QWEN_SPIDER_CKPT_DIR accordingly."
fi

python -u evaluate_on_the_fly_embeddings.py \
  --dataset spider \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 3 \
  --encoder_path "$QWEN_SPIDER_MODEL" \
  --checkpoint "$QWEN_SPIDER_CKPT_DIR/$BEST_QWEN_SPIDER_CHECKPOINT" \
  --pkl_path data/spider_dev_samples_graph.pkl \
  --k 20 \
  --batch_size 128 \
  --log_dir logs/spider_dev_topk20_qwen \
  --pred_collection grast_qwen_spider > logs/end2end/running_spider_qwen_0.6B.log

# BIRD DEV - Top-K 20 (Qwen 4B)
echo ""
echo "🐦 BIRD DEV - Top-K 20 (Qwen 4B)"
echo "================================"

QWEN_BIRD_MODEL_4B=/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-4B-bird/merged_model
QWEN_BIRD_CKPT_DIR_4B=output/bird/grast_bird_reranker-qwen3-4B-finetuned-with-table-meaning
BEST_QWEN_BIRD_CHECKPOINT_4B=$(ls -t "$QWEN_BIRD_CKPT_DIR_4B" 2>/dev/null | grep best_roc_auc_epoch_ | head -n 1)

if [ -z "$BEST_QWEN_BIRD_CHECKPOINT_4B" ]; then
  echo "WARNING: Could not find best_roc_auc_epoch_* in $QWEN_BIRD_CKPT_DIR_4B"
  echo "Please train GNN with qwen 4B embeddings and set QWEN_BIRD_CKPT_DIR_4B accordingly."
fi

python -u evaluate_on_the_fly_embeddings.py \
  --dataset bird \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 3 \
  --encoder_path "$QWEN_BIRD_MODEL_4B" \
  --checkpoint "$QWEN_BIRD_CKPT_DIR_4B/$BEST_QWEN_BIRD_CHECKPOINT_4B" \
  --pkl_path data/bird_dev_samples_graph.pkl \
  --k 30 \
  --batch_size 128 \
  --log_dir logs/bird_dev_topk20_qwen_4B \
  --pred_collection grast_qwen_4B_bird > logs/end2end/running_bird_qwen_4B.log

# BIRD DEV - Top-K 20 (Qwen 8B)
echo ""
echo "🐦 BIRD DEV - Top-K 20 (Qwen 8B)"
echo "================================"

QWEN_BIRD_MODEL_8B=/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-8B-bird/merged_model
QWEN_BIRD_CKPT_DIR_8B=output/bird/grast_bird_reranker-qwen3-8B-finetuned-with-table-meaning
BEST_QWEN_BIRD_CHECKPOINT_8B=$(ls -t "$QWEN_BIRD_CKPT_DIR_8B" 2>/dev/null | grep best_roc_auc_epoch_ | head -n 1)

if [ -z "$BEST_QWEN_BIRD_CHECKPOINT_8B" ]; then
  echo "WARNING: Could not find best_roc_auc_epoch_* in $QWEN_BIRD_CKPT_DIR_8B"
  echo "Please train GNN with qwen 8B embeddings and set QWEN_BIRD_CKPT_DIR_8B accordingly."
fi

python -u evaluate_on_the_fly_embeddings.py \
  --dataset bird \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 3 \
  --encoder_path "$QWEN_BIRD_MODEL_8B" \
  --checkpoint "$QWEN_BIRD_CKPT_DIR_8B/$BEST_QWEN_BIRD_CHECKPOINT_8B" \
  --pkl_path data/bird_dev_samples_graph.pkl \
  --k 30 \
  --batch_size 128 \
  --log_dir logs/bird_dev_topk20_qwen_8B \
  --pred_collection grast_qwen_8B_bird > logs/end2end/running_bird_qwen_8B.log

# SPIDER DEV - Top-K 20 (Qwen 4B)
echo ""
echo "🕷️  SPIDER DEV - Top-K 20 (Qwen 4B)"
echo "==================================="

QWEN_SPIDER_MODEL_4B=/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-4B-spider/merged_model
QWEN_SPIDER_CKPT_DIR_4B=output/spider/grast_spider_reranker-qwen3-4B-finetuned-with-table-meaning
BEST_QWEN_SPIDER_CHECKPOINT_4B=$(ls -t "$QWEN_SPIDER_CKPT_DIR_4B" 2>/dev/null | grep best_pr_auc_epoch_ | head -n 1)

if [ -z "$BEST_QWEN_SPIDER_CHECKPOINT_4B" ]; then
  echo "WARNING: Could not find best_* checkpoint in $QWEN_SPIDER_CKPT_DIR_4B"
  echo "Please train GNN with qwen 4B embeddings for Spider and set QWEN_SPIDER_CKPT_DIR_4B accordingly."
fi

python -u evaluate_on_the_fly_embeddings.py \
  --dataset spider \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 3 \
  --encoder_path "$QWEN_SPIDER_MODEL_4B" \
  --checkpoint "$QWEN_SPIDER_CKPT_DIR_4B/$BEST_QWEN_SPIDER_CHECKPOINT_4B" \
  --pkl_path data/spider_dev_samples_graph.pkl \
  --k 20 \
  --batch_size 128 \
  --log_dir logs/spider_dev_topk20_qwen_4B \
  --pred_collection grast_qwen_4B_spider > logs/end2end/running_spider_qwen_4B.log

# SPIDER DEV - Top-K 20 (Qwen 8B)
echo ""
echo "🕷️  SPIDER DEV - Top-K 20 (Qwen 8B)"
echo "==================================="

QWEN_SPIDER_MODEL_8B=/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-8B-spider/merged_model
QWEN_SPIDER_CKPT_DIR_8B=output/spider/grast_spider_reranker-qwen3-8B-finetuned-with-table-meaning
BEST_QWEN_SPIDER_CHECKPOINT_8B=$(ls -t "$QWEN_SPIDER_CKPT_DIR_8B" 2>/dev/null | grep best_pr_auc_epoch_ | head -n 1)

if [ -z "$BEST_QWEN_SPIDER_CHECKPOINT_8B" ]; then
  echo "WARNING: Could not find best_* checkpoint in $QWEN_SPIDER_CKPT_DIR_8B"
  echo "Please train GNN with qwen 8B embeddings for Spider and set QWEN_SPIDER_CKPT_DIR_8B accordingly."
fi

python -u evaluate_on_the_fly_embeddings.py \
  --dataset spider \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 3 \
  --encoder_path "$QWEN_SPIDER_MODEL_8B" \
  --checkpoint "$QWEN_SPIDER_CKPT_DIR_8B/$BEST_QWEN_SPIDER_CHECKPOINT_8B" \
  --pkl_path data/spider_dev_samples_graph.pkl \
  --k 20 \
  --batch_size 128 \
  --log_dir logs/spider_dev_topk20_qwen_8B \
  --pred_collection grast_qwen_8B_spider > logs/end2end/running_spider_qwen_8B.log

# BIRD TRAIN - Top-K 30 (Qwen)
echo ""
echo "🐦 BIRD TRAIN - Top-K 30 (Qwen)"
echo "==============================="

python -u evaluate_on_the_fly_embeddings.py \
  --dataset bird \
  --split train \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 3 \
  --encoder_path "$QWEN_BIRD_MODEL" \
  --checkpoint "$QWEN_BIRD_CKPT_DIR/$BEST_QWEN_BIRD_CHECKPOINT" \
  --pkl_path data/bird_train_samples_graph.pkl \
  --k 30 \
  --batch_size 128 \
  --log_dir logs/bird_train_topk20_qwen \
  --pred_collection grast_qwen_train > logs/end2end/running_bird_train_qwen_0.6B.log

# BIRD TRAIN - Top-K 30 (Qwen 4B)
echo ""
echo "🐦 BIRD TRAIN - Top-K 20 (Qwen 4B)"
echo "=================================="

python -u evaluate_on_the_fly_embeddings.py \
  --dataset bird \
  --split train \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 3 \
  --encoder_path "$QWEN_BIRD_MODEL_4B" \
  --checkpoint "$QWEN_BIRD_CKPT_DIR_4B/$BEST_QWEN_BIRD_CHECKPOINT_4B" \
  --pkl_path data/bird_train_samples_graph.pkl \
  --k 30 \
  --batch_size 128 \
  --log_dir logs/bird_train_topk20_qwen_4B \
  --pred_collection grast_qwen_4B_bird_train > logs/end2end/running_bird_train_qwen_4B.log

# BIRD TRAIN - Top-K 20 (Qwen 8B)
echo ""
echo "🐦 BIRD TRAIN - Top-K 30 (Qwen 8B)"
echo "=================================="

python -u evaluate_on_the_fly_embeddings.py \
  --dataset bird \
  --split train \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 3 \
  --encoder_path "$QWEN_BIRD_MODEL_8B" \
  --checkpoint "$QWEN_BIRD_CKPT_DIR_8B/$BEST_QWEN_BIRD_CHECKPOINT_8B" \
  --pkl_path data/bird_train_samples_graph.pkl \
  --k 30 \
  --batch_size 128 \
  --log_dir logs/bird_train_topk20_qwen_8B \
  --pred_collection grast_qwen_8B_bird_train > logs/end2end/running_bird_train_qwen_8B.log

# SPIDER TRAIN - Top-K 20 (Qwen)
echo ""
echo "🕷️  SPIDER TRAIN - Top-K 20 (Qwen)"
echo "=================================="

python -u evaluate_on_the_fly_embeddings.py \
  --dataset spider \
  --split train \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 3 \
  --encoder_path "$QWEN_SPIDER_MODEL" \
  --checkpoint "$QWEN_SPIDER_CKPT_DIR/$BEST_QWEN_SPIDER_CHECKPOINT" \
  --pkl_path data/spider_train_samples_graph.pkl \
  --k 20 \
  --batch_size 128 \
  --log_dir logs/spider_train_topk20_qwen \
  --pred_collection grast_qwen_spider_train > logs/end2end/running_spider_train_qwen_0.6B.log

# SPIDER TRAIN - Top-K 20 (Qwen 4B)
echo ""
echo "🕷️  SPIDER TRAIN - Top-K 20 (Qwen 4B)"
echo "====================================="

python -u evaluate_on_the_fly_embeddings.py \
  --dataset spider \
  --split train \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 3 \
  --encoder_path "$QWEN_SPIDER_MODEL_4B" \
  --checkpoint "$QWEN_SPIDER_CKPT_DIR_4B/$BEST_QWEN_SPIDER_CHECKPOINT_4B" \
  --pkl_path data/spider_train_samples_graph.pkl \
  --k 20 \
  --batch_size 128 \
  --log_dir logs/spider_train_topk20_qwen_4B \
  --pred_collection grast_qwen_4B_spider_train > logs/end2end/running_spider_train_qwen_4B.log

# SPIDER TRAIN - Top-K 20 (Qwen 8B)
echo ""
echo "🕷️  SPIDER TRAIN - Top-K 20 (Qwen 8B)"
echo "====================================="

python -u evaluate_on_the_fly_embeddings.py \
  --dataset spider \
  --split train \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 3 \
  --encoder_path "$QWEN_SPIDER_MODEL_8B" \
  --checkpoint "$QWEN_SPIDER_CKPT_DIR_8B/$BEST_QWEN_SPIDER_CHECKPOINT_8B" \
  --pkl_path data/spider_train_samples_graph.pkl \
  --k 20 \
  --batch_size 128 \
  --log_dir logs/spider_train_topk20_qwen_8B \
  --pred_collection grast_qwen_8B_spider_train > logs/end2end/running_spider_train_qwen_8B.log

# SPIDER 2.0 LITE - Top-K 20 (Qwen 0.6B/4B/8B)
echo ""
echo "🕷️ ‍ SPIDER 2.0 LITE - Top-K 20 (Qwen)"
echo "======================================="

# 0.6B
QWEN_SPIDER2_CKPT_DIR_0p6B=output/spider2/grast_bird_reranker-qwen3-0.6B-finetuned-with-table-meaning
BEST_QWEN_SPIDER2_CHECKPOINT_0p6B=$(ls -t "$QWEN_SPIDER2_CKPT_DIR_0p6B" 2>/dev/null | grep best_roc_auc_epoch_ | head -n 1)
if [ -z "$BEST_QWEN_SPIDER2_CHECKPOINT_0p6B" ]; then
  echo "WARNING: Could not find best_* checkpoint in $QWEN_SPIDER2_CKPT_DIR_0p6B"
  echo "Please train GNN with Qwen 0.6B embeddings for Spider2 and set QWEN_SPIDER2_CKPT_DIR_0p6B accordingly."
fi
python -u evaluate_on_the_fly_embeddings.py \
  --dataset spider2 \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 2 \
  --encoder_path "$QWEN_BIRD_MODEL" \
  --checkpoint "$QWEN_SPIDER2_CKPT_DIR_0p6B/$BEST_QWEN_SPIDER2_CHECKPOINT_0p6B" \
  --pkl_path data/spider2_dev_samples_graph_no_evidence.pkl \
  --k 20 \
  --batch_size 128 \
  --log_dir logs/spider2_dev_topk20_qwen_0.6B \
  --pred_collection grast_qwen_spider2_0p6B > logs/end2end/running_spider2_qwen_0.6B.log

# 4B
QWEN_SPIDER2_CKPT_DIR_4B=output/spider2/grast_bird_reranker-qwen3-4B-finetuned-with-table-meaning
BEST_QWEN_SPIDER2_CHECKPOINT_4B=$(ls -t "$QWEN_SPIDER2_CKPT_DIR_4B" 2>/dev/null | grep best_roc_auc_epoch_ | head -n 1)
python -u evaluate_on_the_fly_embeddings.py \
  --dataset spider2 \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 2 \
  --encoder_path "$QWEN_BIRD_MODEL_4B" \
  --checkpoint "$QWEN_SPIDER2_CKPT_DIR_4B/$BEST_QWEN_SPIDER2_CHECKPOINT_4B" \
  --pkl_path data/spider2_dev_samples_graph_no_evidence.pkl \
  --k 20 \
  --batch_size 128 \
  --log_dir logs/spider2_dev_topk20_qwen_4B \
  --pred_collection grast_qwen_spider2_4B > logs/end2end/running_spider2_qwen_4B.log

# 8B
QWEN_SPIDER2_CKPT_DIR_8B=output/spider2/grast_bird_reranker-qwen3-8B-finetuned-with-table-meaning
BEST_QWEN_SPIDER2_CHECKPOINT_8B=$(ls -t "$QWEN_SPIDER2_CKPT_DIR_8B" 2>/dev/null | grep best_roc_auc_epoch_ | head -n 1)
python -u evaluate_on_the_fly_embeddings.py \
  --dataset spider2 \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 1024 \
  --num_layers 2 \
  --encoder_path "$QWEN_BIRD_MODEL_8B" \
  --checkpoint "$QWEN_SPIDER2_CKPT_DIR_8B/$BEST_QWEN_SPIDER2_CHECKPOINT_8B" \
  --pkl_path data/spider2_dev_samples_graph_no_evidence.pkl \
  --k 20 \
  --batch_size 128 \
  --log_dir logs/spider2_dev_topk20_qwen_8B \
  --pred_collection grast_qwen_spider2_8B > logs/end2end/running_spider2_qwen_8B.log








# SPIDER 2.0-LITE DEV - Top-K 30 with Qwen reranker 8B (with evidence)
echo ""
echo "🕷️  SPIDER 2.0-LITE DEV - Top-K 30 (Qwen 8B, with evidence)"
echo "============================================================"

# Using pre-trained model from HuggingFace
# export VLLM_TP_SIZE=2
CUDA_VISIBLE_DEVICES=0,1 vllm serve griffith-bigdata/GRAST-SQL-4B-BIRD-Reranker \
  --port 8000 \
  --max-model-len 32000 \
  --tensor-parallel-size 2 \
  --task embedding \
  --gpu-memory-utilization 0.8

python -u evaluate_on_the_fly.py   \
  --dataset spider2 \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 2048 \
  --num_layers 3 \
  --max_length 16000 \
  --encoder_path griffith-bigdata/GRAST-SQL-4B-BIRD-Reranker \
  --checkpoint griffith-bigdata/GRAST-SQL-4B-BIRD-Reranker/best-bird-dev-roc-auc-layer-3-hidden-2048.pt \
  --pkl_path data/spider2_dev_samples_graph_no_evidence.pkl \
  --k_percent 15 \
  --batch_size 128 \
  --log_dir logs/spider2_dev_topk30_qwen_4b_no_evidence \
  --pred_collection grast_qwen_4b_spider2_lite


CUDA_VISIBLE_DEVICES=0,1 vllm serve griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker \
  --port 8000 \
  --max-model-len 8192 \
  --tensor-parallel-size 2 \
  --task embedding \
  --gpu-memory-utilization 0.8

python -u evaluate_on_the_fly.py \
  --dataset bird \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 2048 \
  --num_layers 3 \
  --encoder_path griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker \
  --checkpoint griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker/best-bird-dev-roc-auc-layer-3-hidden-2048.pt \
  --pkl_path data/bird_dev_samples_graph.pkl \
  --k 30 \
  --batch_size 128 \
  --log_dir logs/bird_dev_topk30_qwen_0.6b \
  --pred_collection grast_qwen_0.6b_bird_dev