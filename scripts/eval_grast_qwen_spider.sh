model_path=/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-0.6B-spider/merged_model
model_name=qwen3-0.6B-finetuned

# source activate gschema
# python frozen_encoder_trainer/init_embeddings.py \
#     --reranker_type qwen \
#     --model_path $model_path \
#     --dataset spider \
#     --split train \
#     --batch_size 64 \
#     --max_length 8192 \
#     --output_dir output/embeddings/spider-train-reranker-$model_name-with-table-meaning

# source activate gschema
# python frozen_encoder_trainer/init_embeddings.py \
#     --reranker_type qwen \
#     --model_path $model_path \
#     --dataset spider \
#     --split dev \
#     --batch_size 64 \
#     --max_length 8192 \
#     --output_dir output/embeddings/spider-dev-reranker-$model_name-with-table-meaning

# source activate bge
# # TRAIN AND EVAL 0.6B MODEL
# python3 frozen_encoder_trainer/train_with_frozen_embeddings.py \
#   --embeddings_dir output/embeddings/spider-train-reranker-$model_name-with-table-meaning \
#   --dev_embeddings_dir output/embeddings/spider-dev-reranker-$model_name-with-table-meaning \
#   --dataset spider \
#   --reranker_type qwen \
#   --num_layers 3 \
#   --hid_dim 1024 \
#   --num_epochs 30 \
#   --batch_size 32 \
#   --learning_rate 1e-4 \
#   --output_dir output/spider/grast_spider_reranker-$model_name-with-table-meaning \
#   --seed 42

# ls and get best checkpoint, assign to BEST_CHECKPOINT,  best_roc_auc_epoch_XX.pt
BEST_CHECKPOINT=$(ls -t output/spider/grast_spider_reranker-$model_name-with-table-meaning/ | grep best_roc_auc_epoch_ | head -n 1)
echo "BEST_CHECKPOINT: $BEST_CHECKPOINT"

source activate bge
python -u frozen_encoder_trainer/evaluate_with_frozen_embeddings.py   \
    --dev_embeddings_dir output/embeddings/spider-dev-reranker-$model_name-with-table-meaning   \
    --dataset spider   \
    --reranker_type qwen \
    --num_layers 3   \
    --hid_dim 1024   \
    --checkpoint output/spider/grast_spider_reranker-$model_name-with-table-meaning/$BEST_CHECKPOINT   \
    --k 2 3 5 6 8 10 20 30 40 \
    --log-low-recall \
    --log-file logs/spider/qwen-$model_name-spider/$model_name-with-table-meaning_dev.json > logs/spider/running_qwen-$model_name-spider.txt







model_path=/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-4B-spider/merged_model
model_name=qwen3-4B-finetuned

# source activate gschema
# python frozen_encoder_trainer/init_embeddings.py \
#     --reranker_type qwen \
#     --model_path $model_path \
#     --dataset spider \
#     --split train \
#     --batch_size 64 \
#     --max_length 8192 \
#     --output_dir output/embeddings/spider-train-reranker-$model_name-with-table-meaning

# source activate gschema
# python frozen_encoder_trainer/init_embeddings.py \
#     --reranker_type qwen \
#     --model_path $model_path \
#     --dataset spider \
#     --split dev \
#     --batch_size 64 \
#     --max_length 8192 \
#     --output_dir output/embeddings/spider-dev-reranker-$model_name-with-table-meaning

# source activate bge
# # TRAIN AND EVAL 0.6B MODEL
# python3 frozen_encoder_trainer/train_with_frozen_embeddings.py \
#   --embeddings_dir output/embeddings/spider-train-reranker-$model_name-with-table-meaning \
#   --dev_embeddings_dir output/embeddings/spider-dev-reranker-$model_name-with-table-meaning \
#   --dataset spider \
#   --reranker_type qwen \
#   --num_layers 3 \
#   --hid_dim 1024 \
#   --num_epochs 30 \
#   --batch_size 32 \
#   --learning_rate 1e-4 \
#   --output_dir output/spider/grast_spider_reranker-$model_name-with-table-meaning \
#   --seed 42

# ls and get best checkpoint, assign to BEST_CHECKPOINT,  best_roc_auc_epoch_XX.pt
BEST_CHECKPOINT=$(ls -t output/spider/grast_spider_reranker-$model_name-with-table-meaning/ | grep best_roc_auc_epoch_ | head -n 1)
echo "BEST_CHECKPOINT: $BEST_CHECKPOINT"

source activate bge
python -u frozen_encoder_trainer/evaluate_with_frozen_embeddings.py   \
    --dev_embeddings_dir output/embeddings/spider-dev-reranker-$model_name-with-table-meaning   \
    --dataset spider   \
    --reranker_type qwen \
    --num_layers 3   \
    --hid_dim 1024   \
    --checkpoint output/spider/grast_spider_reranker-$model_name-with-table-meaning/$BEST_CHECKPOINT   \
    --k 2 3 5 6 8 10 20 30 40 \
    --log-low-recall \
    --log-file logs/spider/qwen-$model_name-spider/$model_name-with-table-meaning_dev.json > logs/spider/running_qwen-$model_name-spider.txt


model_path=/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-8B-spider/merged_model
model_name=qwen3-8B-finetuned

BEST_CHECKPOINT=$(ls -t output/spider/grast_spider_reranker-$model_name-with-table-meaning/ | grep best_roc_auc_epoch_ | head -n 1)
echo "BEST_CHECKPOINT: $BEST_CHECKPOINT"

source activate bge
python -u frozen_encoder_trainer/evaluate_with_frozen_embeddings.py   \
    --dev_embeddings_dir output/embeddings/spider-dev-reranker-$model_name-with-table-meaning   \
    --dataset spider   \
    --reranker_type qwen \
    --num_layers 3   \
    --hid_dim 1024   \
    --checkpoint output/spider/grast_spider_reranker-$model_name-with-table-meaning/$BEST_CHECKPOINT   \
    --k 2 3 5 6 8 10 20 30 40 \
    --log-low-recall \
    --log-file logs/spider/qwen-$model_name-spider/$model_name-with-table-meaning_dev.json > logs/spider/running_qwen-$model_name-spider.txt
    