# QWEN - SPIDER (train)
python frozen_encoder_trainer/init_embeddings.py \
    --reranker_type qwen \
    --model_path /home/datht/huggingface/Qwen/Qwen3-Reranker-0.6B \
    --dataset spider \
    --split train \
    --batch_size 16 \
    --max_length 8192 \
    --output_dir output/embeddings/spider-train-reranker-qwen3-0.6B

# QWEN - SPIDER (dev)
python frozen_encoder_trainer/init_embeddings.py \
    --reranker_type qwen \
    --model_path /home/datht/huggingface/Qwen/Qwen3-Reranker-0.6B \
    --dataset spider \
    --split dev \
    --batch_size 16 \
    --max_length 8192 \
    --output_dir output/embeddings/spider-dev-reranker-qwen3-0.6B

# QWEN - BIRD (train)
python frozen_encoder_trainer/init_embeddings.py \
    --reranker_type qwen \
    --model_path /home/datht/huggingface/Qwen/Qwen3-Reranker-0.6B \
    --dataset bird \
    --split train \
    --batch_size 16 \
    --max_length 8192 \
    --output_dir output/embeddings/bird-train-reranker-qwen3-0.6B

# QWEN - BIRD (dev)
python frozen_encoder_trainer/init_embeddings.py \
    --reranker_type qwen \
    --model_path /home/datht/huggingface/Qwen/Qwen3-Reranker-0.6B \
    --dataset bird \
    --split dev \
    --batch_size 16 \
    --max_length 8192 \
    --output_dir output/embeddings/bird-dev-reranker-qwen3-0.6B

# QWEN - SPIDER 2.0 - NO EVIDENCE
echo "Initializing Spider 2.0 embeddings with Qwen reranker (no evidence)..."
python frozen_encoder_trainer/init_embeddings.py \
    --reranker_type qwen \
    --model_path /home/datht/huggingface/Qwen/Qwen3-Reranker-0.6B \
    --dataset spider2 \
    --split dev \
    --evidence_version no_evidence \
    --batch_size 4 \
    --max_length 8192 \
    --output_dir output/embeddings/spider2-dev-reranker-qwen3-0.6B-no-evidence

# QWEN 4B - SPIDER 2.0 - NO EVIDENCE
echo "Initializing Spider 2.0 embeddings with Qwen reranker (no evidence)..."
python frozen_encoder_trainer/init_embeddings.py \
    --reranker_type qwen \
    --model_path /home/datht/huggingface/Qwen/Qwen3-Reranker-4B \
    --dataset spider2 \
    --split dev \
    --evidence_version no_evidence \
    --batch_size 2 \
    --max_length 8192 \
    --output_dir output/embeddings/spider2-dev-reranker-qwen3-4B-no-evidence

python frozen_encoder_trainer/init_embeddings.py \
    --reranker_type qwen \
    --model_path /home/datht/huggingface/Qwen/Qwen3-Reranker-8B  \
    --dataset spider2 \
    --split dev \
    --evidence_version no_evidence \
    --batch_size 2 \
    --max_length 8192 \
    --output_dir output/embeddings/spider2-dev-reranker-qwen3-8B-no-evidence



# QWEN - SPIDER 2.0 - WITH EVIDENCE
echo "Initializing Spider 2.0 embeddings with Qwen reranker (with evidence)..."
python frozen_encoder_trainer/init_embeddings.py \
    --reranker_type qwen \
    --model_path /home/datht/huggingface/Qwen/Qwen3-Reranker-0.6B \
    --dataset spider2 \
    --split dev \
    --evidence_version with_evidence \
    --batch_size 4 \
    --max_length 8192 \
    --output_dir output/embeddings/spider2-dev-reranker-qwen3-0.6B-with-evidence



# QWEN FINETUNED - Spider 2.0 - NO EVIDENCE - finetuned-reranker-qwen3-0.6b-bird/checkpoint-584
python frozen_encoder_trainer/init_embeddings.py \
    --reranker_type qwen \
    --model_path embedder/output/finetuned-reranker-qwen3-0.6b-bird/checkpoint-584 \
    --dataset spider2 \
    --split dev \
    --evidence_version no_evidence \
    --batch_size 4 \
    --max_length 8192 \
    --output_dir output/embeddings/spider2-dev-reranker-qwen3-0.6b-bird-checkpoint-584-no-evidence

# QWEN FINETUNED - BIRD train 
python frozen_encoder_trainer/init_embeddings.py \
    --reranker_type qwen \
    --model_path embedder/output/finetuned-reranker-qwen3-0.6b-bird/checkpoint-584 \
    --dataset bird \
    --split train \
    --batch_size 16 \
    --max_length 8192 \
    --output_dir output/embeddings/bird-train-reranker-qwen3-0.6b-bird-checkpoint-584

