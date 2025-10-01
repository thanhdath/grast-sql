#!/bin/bash

set -e

cd /home/datht/graph-schema/embedder

mkdir -p /home/datht/graph-schema/logs/spider/embedding_qwen3_dev/0_6b
# python -u wip_evaluate_qwen_embedding.py \
#   --dev-file /home/datht/graph-schema/data/spider_dev_samples_graph.pkl \
#   --model-path /home/datht/huggingface/Qwen/Qwen3-Embedding-0.6B \
#   --split dev \
#   --dataset spider \
#   --k 100 \
#   --embedding-batch-size 128 \
#   --max-length 8192 \
#   --truncate-side right \
#   --log-file /home/datht/graph-schema/logs/spider/embedding_qwen3_dev/0_6b/dev.json > /home/datht/graph-schema/logs/spider/running_qwen3_embedding_0_6b_spider.log

# mkdir -p /home/datht/graph-schema/logs/bird/embedding_qwen3_dev/0_6b
# python -u wip_evaluate_qwen_embedding.py \
#   --dev-file /home/datht/graph-schema/data/bird_dev_samples_graph.pkl \
#   --model-path /home/datht/huggingface/Qwen/Qwen3-Embedding-0.6B \
#   --split dev \
#   --dataset bird \
#   --k 100 \
#   --embedding-batch-size 128 \
#   --max-length 8192 \
#   --truncate-side right \
#   --log-file /home/datht/graph-schema/logs/bird/embedding_qwen3_dev/0_6b/dev.json > /home/datht/graph-schema/logs/bird/running_qwen3_embedding_0_6b_bird.log

mkdir -p /home/datht/graph-schema/logs/spider2/embedding_qwen3_dev/0_6b
python -u wip_evaluate_qwen_embedding.py \
  --dev-file /home/datht/graph-schema/data/spider2_dev_samples_graph_no_evidence.pkl \
  --model-path /home/datht/huggingface/Qwen/Qwen3-Embedding-0.6B \
  --split dev \
  --dataset spider2 \
  --k 100 \
  --embedding-batch-size 2 \
  --max-length 8192 \
  --truncate-side right \
  --log-file /home/datht/graph-schema/logs/spider2/embedding_qwen3_dev/0_6b/dev.json
  
  #  > /home/datht/graph-schema/logs/spider2/running_qwen3_embedding_0_6b_spider2.log 