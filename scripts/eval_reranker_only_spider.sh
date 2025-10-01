#!/bin/bash

# Evaluate original Qwen3 Reranker models (0.6B, 4B, 8B) on SPIDER dev samples (original)
cd embedder/

echo "Starting evaluation for Qwen3-Reranker-0.6B (original) on SPIDER..."
mkdir -p ../logs/spider/original_qwen3_reranker_dev/0_6b
python -u evaluate_qwen_reranker.py \
    --dev-file /home/datht/graph-schema/data/spider_dev_samples_graph.pkl \
    --model-path /home/datht/huggingface/Qwen/Qwen3-Reranker-0.6B \
    --split dev \
    --dataset spider \
    --k 100 \
    --batch-size 16 \
    --query-max-length 4096 \
    --passage-max-length 4096 \
    --device cuda:0 \
    --use-vllm \
    --log-file ../logs/spider/original_qwen3_reranker_dev/0_6b/dev.json > ../logs/spider/running_qwen3_reranker_0_6b_spider.log

echo "Starting evaluation for Qwen3-Reranker-4B (original) on SPIDER..."
mkdir -p ../logs/spider/original_qwen3_reranker_dev/4b
python -u evaluate_qwen_reranker.py \
    --dev-file /home/datht/graph-schema/data/spider_dev_samples_graph.pkl \
    --model-path /home/datht/huggingface/Qwen/Qwen3-Reranker-4B \
    --split dev \
    --dataset spider \
    --k 100 \
    --batch-size 16 \
    --query-max-length 4096 \
    --passage-max-length 4096 \
    --device cuda:0 \
    --use-vllm \
    --log-file ../logs/spider/original_qwen3_reranker_dev/4b/dev.json > ../logs/spider/running_qwen3_reranker_4b_spider.log

echo "Starting evaluation for Qwen3-Reranker-8B (original) on SPIDER..."
mkdir -p ../logs/spider/original_qwen3_reranker_dev/8b
python -u evaluate_qwen_reranker.py \
    --dev-file /home/datht/graph-schema/data/spider_dev_samples_graph.pkl \
    --model-path /home/datht/huggingface/Qwen/Qwen3-Reranker-8B \
    --split dev \
    --dataset spider \
    --k 100 \
    --batch-size 16 \
    --query-max-length 4096 \
    --passage-max-length 4096 \
    --device cuda:0 \
    --use-vllm \
    --log-file ../logs/spider/original_qwen3_reranker_dev/8b/dev.json > ../logs/spider/running_qwen3_reranker_8b_spider.log

echo "All SPIDER evaluations completed! Logs saved under ../logs/spider/original_qwen3_reranker_dev/"
