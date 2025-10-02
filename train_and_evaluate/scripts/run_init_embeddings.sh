#!/usr/bin/env bash
# This script runs init_embeddings.py (Qwen vLLM) for:
# - 0.6B finetuned: SPIDER train/dev, BIRD train/dev, Spider2.0-lite with/without evidence
# - 4B finetuned: SPIDER train/dev, BIRD train/dev, Spider2.0-lite with/without evidence
# - 8B finetuned: SPIDER train/dev, BIRD train/dev, Spider2.0-lite with/without evidence

# ========== 0.6B (finetuned) ==========
echo "LLM Reranker 0.6B (SPIDER train)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/spider_train_samples_graph.pkl \
  --model_path griffith-bigdata/GRAST-SQL-0.6B-SPIDER-Reranker \
  --batch_size 16 \
  --max_length 8192 \
  --output_dir output/embeddings/spider-train-reranker-qwen3-0.6B

echo "LLM Reranker 0.6B (SPIDER dev)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/spider_dev_samples_graph.pkl \
  --model_path griffith-bigdata/GRAST-SQL-0.6B-SPIDER-Reranker \
  --batch_size 16 \
  --max_length 8192 \
  --output_dir output/embeddings/spider-dev-reranker-qwen3-0.6B

echo "LLM Reranker 0.6B (BIRD train)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/bird_train_samples_graph.pkl \
  --model_path griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker \
  --batch_size 16 \
  --max_length 8192 \
  --output_dir output/embeddings/bird-train-reranker-qwen3-0.6B

echo "LLM Reranker 0.6B (BIRD dev)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/bird_dev_samples_graph.pkl \
  --model_path griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker \
  --batch_size 16 \
  --max_length 8192 \
  --output_dir output/embeddings/bird-dev-reranker-qwen3-0.6B

echo "LLM Reranker 0.6B (Spider2.0-lite, no external knowledge)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/spider2_dev_samples_graph_no_external_knowledge.pkl \
  --model_path griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker \
  --batch_size 8 \
  --max_length 8192 \
  --output_dir output/embeddings/spider2-dev-no-external-knowledge-reranker-qwen3-0.6B

echo "LLM Reranker 0.6B (Spider2.0-lite, with external knowledge)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/spider2_dev_samples_graph_with_external_knowledge.pkl \
  --model_path griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker \
  --batch_size 8 \
  --max_length 8192 \
  --output_dir output/embeddings/spider2-dev-with-external-knowledge-reranker-qwen3-0.6B

# ========== 4B (finetuned) ==========
echo "LLM Reranker 4B (SPIDER train)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/spider_train_samples_graph.pkl \
  --model_path griffith-bigdata/GRAST-SQL-4B-SPIDER-Reranker \
  --batch_size 8 \
  --max_length 8192 \
  --output_dir output/embeddings/spider-train-reranker-qwen3-4B

echo "LLM Reranker 4B (SPIDER dev)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/spider_dev_samples_graph.pkl \
  --model_path griffith-bigdata/GRAST-SQL-4B-SPIDER-Reranker \
  --batch_size 8 \
  --max_length 8192 \
  --output_dir output/embeddings/spider-dev-reranker-qwen3-4B

echo "LLM Reranker 4B (BIRD train)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/bird_train_samples_graph.pkl \
  --model_path griffith-bigdata/GRAST-SQL-4B-BIRD-Reranker \
  --batch_size 8 \
  --max_length 8192 \
  --output_dir output/embeddings/bird-train-reranker-qwen3-4B

echo "LLM Reranker 4B (BIRD dev)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/bird_dev_samples_graph.pkl \
  --model_path griffith-bigdata/GRAST-SQL-4B-BIRD-Reranker \
  --batch_size 8 \
  --max_length 8192 \
  --output_dir output/embeddings/bird-dev-reranker-qwen3-4B

echo "LLM Reranker 4B (Spider2.0-lite, no external knowledge)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/spider2_dev_samples_graph_no_external_knowledge.pkl \
  --model_path griffith-bigdata/GRAST-SQL-4B-BIRD-Reranker \
  --batch_size 4 \
  --max_length 8192 \
  --output_dir output/embeddings/spider2-dev-no-external-knowledge-reranker-qwen3-4B


echo "LLM Reranker 4B (Spider2.0-lite, with external knowledge)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/spider2_dev_samples_graph_with_external_knowledge.pkl \
  --model_path griffith-bigdata/GRAST-SQL-4B-BIRD-Reranker \
  --batch_size 4 \
  --max_length 8192 \
  --output_dir output/embeddings/spider2-dev-with-external-knowledge-reranker-qwen3-4B

# ========== 8B (finetuned) ==========
echo "LLM Reranker 8B (SPIDER train)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/spider_train_samples_graph.pkl \
  --model_path griffith-bigdata/GRAST-SQL-8B-SPIDER-Reranker \
  --batch_size 4 \
  --max_length 8192 \
  --output_dir output/embeddings/spider-train-reranker-qwen3-8B

echo "LLM Reranker 8B (SPIDER dev)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/spider_dev_samples_graph.pkl \
  --model_path griffith-bigdata/GRAST-SQL-8B-SPIDER-Reranker \
  --batch_size 4 \
  --max_length 8192 \
  --output_dir output/embeddings/spider-dev-reranker-qwen3-8B

echo "LLM Reranker 8B (BIRD train)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/bird_train_samples_graph.pkl \
  --model_path griffith-bigdata/GRAST-SQL-8B-BIRD-Reranker \
  --batch_size 4 \
  --max_length 8192 \
  --output_dir output/embeddings/bird-train-reranker-qwen3-8B

echo "LLM Reranker 8B (BIRD dev)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/bird_dev_samples_graph.pkl \
  --model_path griffith-bigdata/GRAST-SQL-8B-BIRD-Reranker \
  --batch_size 4 \
  --max_length 8192 \
  --output_dir output/embeddings/bird-dev-reranker-qwen3-8B

echo "LLM Reranker 8B (Spider2.0-lite, no external knowledge)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/spider2_dev_samples_graph_no_external_knowledge.pkl \
  --model_path griffith-bigdata/GRAST-SQL-8B-BIRD-Reranker \
  --batch_size 2 \
  --max_length 8192 \
  --output_dir output/embeddings/spider2-dev-with-external-knowledge-reranker-qwen3-8B


echo "LLM Reranker 8B (Spider2.0-lite, with external knowledge)"
python modules/column_encoder/init_embeddings.py \
  /home/datht/graph-schema/data/spider2_dev_samples_graph_with_external_knowledge.pkl \
  --model_path griffith-bigdata/GRAST-SQL-8B-BIRD-Reranker \
  --batch_size 2 \
  --max_length 8192 \
  --output_dir output/embeddings/spider2-dev-no-external-knowledge-reranker-qwen3-8B
