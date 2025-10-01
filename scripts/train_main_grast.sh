mkdir -p logs/
mkdir -p logs/main

echo "GRAST-SQL 0.6B on Spider, train set = SPIDER train, validation set = None, layer = 3, hidden = 2048"
python3 -u frozen_encoder_trainer/train_with_frozen_embeddings.py \
    --embeddings_dir output/embeddings/spider-train-reranker-qwen3-0.6B-finetuned-with-table-meaning \
    --dataset spider \
    --reranker_type qwen \
    --num_layers 3 \
    --hid_dim 2048 \
    --num_epochs 40 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --output_dir output/spider/grast_spider_reranker-qwen3-0.6B-finetuned-with-table-meaning/layer-3-hidden-2048 \
    --seed 42 > logs/main/spider-qwen-layer-3-hidden-2048.log


echo "GRAST-SQL 0.6B on BIRD, train set = BIRD train, validation set = None, layer = 3, hidden = 2048"
python3 -u frozen_encoder_trainer/train_with_frozen_embeddings.py \
    --embeddings_dir output/embeddings/bird-train-reranker-qwen3-0.6B-finetuned-with-table-meaning \
    --dataset bird \
    --reranker_type qwen \
    --num_layers 3 \
    --hid_dim 2048 \
    --num_epochs 40 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --output_dir output/bird/grast_bird_reranker-qwen3-0.6B-finetuned-with-table-meaning/layer-3-hidden-2048 \
    --seed 42 > logs/main/bird-qwen-layer-3-hidden-2048.log


echo "GRAST-SQL 0.6B on SPIDER 2.0-lite, train set = BIRD train, validation set = BIRD dev, layer = 3, hidden = 2048"
python3 -u frozen_encoder_trainer/train_with_frozen_embeddings.py \
    --embeddings_dir output/embeddings/bird-train-reranker-qwen3-0.6B-finetuned-with-table-meaning \
    --dev_embeddings_dir output/embeddings/bird-dev-reranker-qwen3-0.6B-finetuned-with-table-meaning \
    --dataset bird \
    --reranker_type qwen \
    --num_layers 3 \
    --hid_dim 2048 \
    --num_epochs 40 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --output_dir output/spider2/grast_bird_reranker-qwen3-0.6B-finetuned-with-table-meaning/layer-3-hidden-2048 \
    --seed 42 > logs/main/spider2-qwen-layer-3-hidden-2048.log


# 4B 
echo "GRAST-SQL 4B on Spider, train set = SPIDER train, validation set = None, layer = 3, hidden = 2048"
python3 -u frozen_encoder_trainer/train_with_frozen_embeddings.py \
    --embeddings_dir output/embeddings/spider-train-reranker-qwen3-4B-finetuned-with-table-meaning \
    --dataset spider \
    --reranker_type qwen \
    --num_layers 3 \
    --hid_dim 2048 \
    --num_epochs 40 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --output_dir output/spider/grast_spider_reranker-qwen3-4B-finetuned-with-table-meaning/layer-3-hidden-2048 \
    --seed 42 > logs/main/spider-qwen-layer-3-hidden-2048.log


echo "GRAST-SQL 4B on BIRD, train set = BIRD train, validation set = None, layer = 3, hidden = 2048"
python3 -u frozen_encoder_trainer/train_with_frozen_embeddings.py \
    --embeddings_dir output/embeddings/bird-train-reranker-qwen3-4B-finetuned-with-table-meaning \
    --dataset bird \
    --reranker_type qwen \
    --num_layers 3 \
    --hid_dim 2048 \
    --num_epochs 40 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --output_dir output/bird/grast_bird_reranker-qwen3-4B-finetuned-with-table-meaning/layer-3-hidden-2048 \
    --seed 42 > logs/main/bird-qwen-layer-3-hidden-2048.log


echo "GRAST-SQL 4B on SPIDER 2.0-lite, train set = BIRD train, validation set = BIRD dev, layer = 3, hidden = 2048"
python3 -u frozen_encoder_trainer/train_with_frozen_embeddings.py \
    --embeddings_dir output/embeddings/bird-train-reranker-qwen3-4B-finetuned-with-table-meaning \
    --dev_embeddings_dir output/embeddings/bird-dev-reranker-qwen3-4B-finetuned-with-table-meaning \
    --dataset bird \
    --reranker_type qwen \
    --num_layers 3 \
    --hid_dim 2048 \
    --num_epochs 40 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --output_dir output/spider2/grast_bird_reranker-qwen3-4B-finetuned-with-table-meaning/layer-3-hidden-2048 \
    --seed 42 > logs/main/spider2-qwen-layer-3-hidden-2048.log


# 8B
echo "GRAST-SQL 8B on Spider, train set = SPIDER train, validation set = None, layer = 3, hidden = 2048"
python3 -u frozen_encoder_trainer/train_with_frozen_embeddings.py \
    --embeddings_dir output/embeddings/spider-train-reranker-qwen3-8B-finetuned-with-table-meaning \
    --dataset spider \
    --reranker_type qwen \
    --num_layers 3 \
    --hid_dim 2048 \
    --num_epochs 40 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --output_dir output/spider/grast_spider_reranker-qwen3-8B-finetuned-with-table-meaning/layer-3-hidden-2048 \
    --seed 42 > logs/main/spider-qwen-layer-3-hidden-2048.log


echo "GRAST-SQL 8B on BIRD, train set = BIRD train, validation set = None, layer = 3, hidden = 2048"
python3 -u frozen_encoder_trainer/train_with_frozen_embeddings.py \
    --embeddings_dir output/embeddings/bird-train-reranker-qwen3-8B-finetuned-with-table-meaning \
    --dataset bird \
    --reranker_type qwen \
    --num_layers 3 \
    --hid_dim 2048 \
    --num_epochs 40 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --output_dir output/bird/grast_bird_reranker-qwen3-8B-finetuned-with-table-meaning/layer-3-hidden-2048 \
    --seed 42 > logs/main/bird-qwen-layer-3-hidden-2048.log


echo "GRAST-SQL 8B on SPIDER 2.0-lite, train set = BIRD train, validation set = BIRD dev, layer = 3, hidden = 2048"
python3 -u frozen_encoder_trainer/train_with_frozen_embeddings.py \
    --embeddings_dir output/embeddings/bird-train-reranker-qwen3-8B-finetuned-with-table-meaning \
    --dev_embeddings_dir output/embeddings/bird-dev-reranker-qwen3-8B-finetuned-with-table-meaning \
    --dataset bird \
    --reranker_type qwen \
    --num_layers 3 \
    --hid_dim 2048 \
    --num_epochs 40 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --output_dir output/spider2/grast_bird_reranker-qwen3-8B-finetuned-with-table-meaning/layer-3-hidden-2048 \
    --seed 42 > logs/main/spider2-qwen-layer-3-hidden-2048.log

