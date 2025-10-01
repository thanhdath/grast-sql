#!/bin/bash

# Training and evaluation for qwen3-0.6B-finetuned model
source activate bge
python3 frozen_encoder_trainer/train_with_frozen_embeddings.py \
  --embeddings_dir output/embeddings/bird-train-reranker-qwen3-0.6B-finetuned-with-table-meaning \
  --dev_embeddings_dir output/embeddings/bird-dev-reranker-qwen3-0.6B-finetuned-with-table-meaning \
  --dataset bird \
  --reranker_type qwen \
  --num_layers 3 \
  --hid_dim 2048 \
  --num_epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --output_dir output/bird/grast_bird_reranker-qwen3-0.6B-finetuned-with-table-meaning \
  --seed 42

BEST_CHECKPOINT=$(ls -t output/bird/grast_bird_reranker-qwen3-0.6B-finetuned-with-table-meaning/ | grep best_roc_auc_epoch_ | head -n 1)
echo "BEST_CHECKPOINT: $BEST_CHECKPOINT"

source activate bge
python -u frozen_encoder_trainer/evaluate_with_frozen_embeddings.py \
    --dev_embeddings_dir output/embeddings/bird-dev-reranker-qwen3-0.6B-finetuned-with-table-meaning \
    --dataset bird \
    --reranker_type qwen \
    --num_layers 3 \
    --hid_dim 2048 \
    --checkpoint output/bird/grast_bird_reranker-qwen3-0.6B-finetuned-with-table-meaning/$BEST_CHECKPOINT \
    --k 2 3 5 6 8 10 20 30 40 \
    --log-low-recall \
    --log-file logs/bird/qwen-qwen3-0.6B-finetuned-bird/qwen3-0.6B-finetuned-with-table-meaning_dev.json > logs/bird/running_qwen-qwen3-0.6B-finetuned-bird.txt

# Evaluation for qwen3-4B-finetuned model
BEST_CHECKPOINT=$(ls -t output/bird/grast_bird_reranker-qwen3-4B-finetuned-with-table-meaning/ | grep best_roc_auc_epoch_ | head -n 1)
echo "BEST_CHECKPOINT: $BEST_CHECKPOINT"

source activate bge
python -u frozen_encoder_trainer/evaluate_with_frozen_embeddings.py \
    --dev_embeddings_dir output/embeddings/bird-dev-reranker-qwen3-4B-finetuned-with-table-meaning \
    --dataset bird \
    --reranker_type qwen \
    --num_layers 3 \
    --hid_dim 2048 \
    --checkpoint output/bird/grast_bird_reranker-qwen3-4B-finetuned-with-table-meaning/$BEST_CHECKPOINT \
    --k 2 3 5 6 8 10 20 30 40 \
    --log-low-recall \
    --log-file logs/bird/qwen-qwen3-4B-finetuned-bird/qwen3-4B-finetuned-with-table-meaning_dev.json > logs/bird/running_qwen-qwen3-4B-finetuned-bird.txt

# Evaluation for qwen3-8B-finetuned model
BEST_CHECKPOINT=$(ls -t output/bird/grast_bird_reranker-qwen3-8B-finetuned-with-table-meaning/ | grep best_roc_auc_epoch_ | head -n 1)
echo "BEST_CHECKPOINT: $BEST_CHECKPOINT"

source activate bge
python -u frozen_encoder_trainer/evaluate_with_frozen_embeddings.py \
    --dev_embeddings_dir output/embeddings/bird-dev-reranker-qwen3-8B-finetuned-with-table-meaning \
    --dataset bird \
    --reranker_type qwen \
    --num_layers 3 \
    --hid_dim 2048 \
    --checkpoint output/bird/grast_bird_reranker-qwen3-8B-finetuned-with-table-meaning/$BEST_CHECKPOINT \
    --k 2 3 5 6 8 10 20 30 40 \
    --log-low-recall \
    --log-file logs/bird/qwen-qwen3-8B-finetuned-bird/qwen3-8B-finetuned-with-table-meaning_dev.json > logs/bird/running_qwen-qwen3-8B-finetuned-bird.txt
