#!/bin/bash

# Evaluate original Qwen3 Reranker models (0.6B, 4B, 8B) on BIRD dev samples (original)
cd embedder/

echo "Starting evaluation for Qwen3-Reranker-0.6B (original) on BIRD..."
mkdir -p ../logs/bird/original_qwen3_reranker_dev/0_6b
python -u evaluate_qwen_reranker.py \
    --dev-file /home/datht/graph-schema/data/bird_dev_samples_graph.pkl \
    --model-path /home/datht/huggingface/Qwen/Qwen3-Reranker-0.6B \
    --split dev \
    --dataset bird \
    --k 100 \
    --batch-size 16 \
    --query-max-length 4096 \
    --passage-max-length 4096 \
    --device cuda:0 \
    --use-vllm \
    --log-file ../logs/bird/original_qwen3_reranker_dev/0_6b/dev.json > ../logs/bird/running_qwen3_reranker_0_6b_bird.log

echo "Starting evaluation for Qwen3-Reranker-4B (original) on BIRD..."
mkdir -p ../logs/bird/original_qwen3_reranker_dev/4b
python -u evaluate_qwen_reranker.py \
    --dev-file /home/datht/graph-schema/data/bird_dev_samples_graph.pkl \
    --model-path /home/datht/huggingface/Qwen/Qwen3-Reranker-4B \
    --split dev \
    --dataset bird \
    --k 100 \
    --batch-size 16 \
    --query-max-length 4096 \
    --passage-max-length 4096 \
    --device cuda:0 \
    --use-vllm \
    --log-file ../logs/bird/original_qwen3_reranker_dev/4b/dev.json > ../logs/bird/running_qwen3_reranker_4b_bird.log

echo "Starting evaluation for Qwen3-Reranker-8B (original) on BIRD..."
mkdir -p ../logs/bird/original_qwen3_reranker_dev/8b
python -u evaluate_qwen_reranker.py \
    --dev-file /home/datht/graph-schema/data/bird_dev_samples_graph.pkl \
    --model-path /home/datht/huggingface/Qwen/Qwen3-Reranker-8B \
    --split dev \
    --dataset bird \
    --k 100 \
    --batch-size 16 \
    --query-max-length 4096 \
    --passage-max-length 4096 \
    --device cuda:0 \
    --use-vllm \
    --log-file ../logs/bird/original_qwen3_reranker_dev/8b/dev.json > ../logs/bird/running_qwen3_reranker_8b_bird.log

echo "All BIRD evaluations completed! Logs saved under ../logs/bird/original_qwen3_reranker_dev/" 

echo "Starting evaluation for Qwen3-Reranker-4B on BIRD TRAIN (filtered db_ids)..."
mkdir -p ../logs/bird/original_qwen3_reranker_train/4b_filtered
python -u evaluate_qwen_reranker.py \
    --dev-file /home/datht/graph-schema/data/bird_train_samples_graph.pkl \
    --model-path /home/datht/huggingface/Qwen/Qwen3-Reranker-4B \
    --split train \
    --dataset bird \
    --k 100 \
    --batch-size 16 \
    --query-max-length 4096 \
    --passage-max-length 4096 \
    --device cuda:0 \
    --use-vllm \
    --db-ids works_cycles professional_basketball hockey citeseer \
    --log-file ../logs/bird/original_qwen3_reranker_train/4b_filtered/train_filtered.json > ../logs/bird/running_qwen3_reranker_4b_filtered_bird.log