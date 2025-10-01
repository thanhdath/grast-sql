# model_path=/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-0.6b-bird/merged_model
# model_name=qwen3-0.6B-finetuned

# model_path=/home/datht/huggingface/Qwen/Qwen3-Reranker-4B
# model_name=qwen3-4B

model_path=/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-0.6b-bird/merged_model/
model_name=qwen3-0.6B-finetuned

# source activate gschema
# python frozen_encoder_trainer/init_embeddings.py \
#     --reranker_type qwen \
#     --model_path $model_path \
#     --dataset bird \
#     --split train \
#     --batch_size 64 \
#     --max_length 8192 \
#     --output_dir output/embeddings/bird-train-reranker-$model_name-with-table-meaning

# source activate gschema
# python frozen_encoder_trainer/init_embeddings.py \
#     --reranker_type qwen \
#     --model_path $model_path \
#     --dataset spider2 \
#     --split dev \
#     --evidence_version no_evidence \
#     --batch_size 64 \
#     --max_length 8192 \
#     --output_dir output/embeddings/spider2-dev-reranker-$model_name-with-table-meaning


source activate bge
# TRAIN AND EVAL 0.6B MODEL
python3 frozen_encoder_trainer/train_with_frozen_embeddings.py \
  --embeddings_dir output/embeddings/bird-train-reranker-$model_name-with-table-meaning \
  --dev_embeddings_dir output/embeddings/spider2-dev-reranker-$model_name-with-table-meaning \
  --dataset bird \
  --reranker_type qwen \
  --num_layers 2 \
  --hid_dim 1024 \
  --num_epochs 30 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --output_dir output/spider2/grast_bird_reranker-$model_name-with-table-meaning \
  --seed 42

# ls and get best checkpoint, assign to BEST_CHECKPOINT,  best_roc_auc_epoch_XX.pt
BEST_CHECKPOINT=$(ls -t output/spider2/grast_bird_reranker-$model_name-with-table-meaning/ | grep best_roc_auc_epoch_ | head -n 1)
echo "BEST_CHECKPOINT: $BEST_CHECKPOINT"

source activate bge
mkdir -p logs/spider2
python -u frozen_encoder_trainer/evaluate_with_frozen_embeddings.py   \
    --dev_embeddings_dir output/embeddings/spider2-dev-reranker-$model_name-with-table-meaning   \
    --dataset spider2   \
    --reranker_type qwen   \
    --num_layers 2   \
    --hid_dim 1024   \
    --checkpoint output/spider2/grast_bird_reranker-$model_name-with-table-meaning/$BEST_CHECKPOINT   \
    --k 30 40 100 200 300 \
    --log-low-recall \
    --log-file logs/spider2/qwen-$model_name/grast_bird_reranker_$model_name-with-table-meaning_dev.json > logs/spider2/running_qwen-$model_name-spider2.txt



model_path=/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-4B-bird/merged_model/
model_name=qwen3-4B-finetuned

python3 frozen_encoder_trainer/train_with_frozen_embeddings.py \
  --embeddings_dir output/embeddings/bird-train-reranker-$model_name-with-table-meaning \
  --dev_embeddings_dir output/embeddings/spider2-dev-reranker-$model_name-with-table-meaning \
  --dataset bird \
  --reranker_type qwen \
  --num_layers 2 \
  --hid_dim 1024 \
  --num_epochs 30 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --output_dir output/spider2/grast_bird_reranker-$model_name-with-table-meaning \
  --seed 42

# ls and get best checkpoint, assign to BEST_CHECKPOINT,  best_roc_auc_epoch_XX.pt
BEST_CHECKPOINT=$(ls -t output/spider2/grast_bird_reranker-$model_name-with-table-meaning/ | grep best_roc_auc_epoch_ | head -n 1)
echo "BEST_CHECKPOINT: $BEST_CHECKPOINT"

source activate bge
python -u frozen_encoder_trainer/evaluate_with_frozen_embeddings.py   \
    --dev_embeddings_dir output/embeddings/spider2-dev-reranker-$model_name-with-table-meaning   \
    --dataset spider2   \
    --reranker_type qwen   \
    --num_layers 2   \
    --hid_dim 1024   \
    --checkpoint output/spider2/grast_bird_reranker-$model_name-with-table-meaning/$BEST_CHECKPOINT   \
    --k 30 40 100 200 300   \
    --log-low-recall \
    --log-file logs/spider2/qwen-$model_name/grast_bird_reranker_$model_name-with-table-meaning_dev.json > logs/spider2/running_qwen-$model_name-spider2.txt







model_path=/home/datht/graph-schema/embedder/output/finetuned-reranker-qwen3-8B-bird/merged_model/
model_name=qwen3-8B-finetuned

python3 frozen_encoder_trainer/train_with_frozen_embeddings.py \
  --embeddings_dir output/embeddings/bird-train-reranker-$model_name-with-table-meaning \
  --dev_embeddings_dir output/embeddings/spider2-dev-reranker-$model_name-with-table-meaning \
  --dataset bird \
  --reranker_type qwen \
  --num_layers 2 \
  --hid_dim 1024 \
  --num_epochs 30 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --output_dir output/spider2/grast_bird_reranker-$model_name-with-table-meaning \
  --seed 42

# ls and get best checkpoint, assign to BEST_CHECKPOINT,  best_roc_auc_epoch_XX.pt
BEST_CHECKPOINT=$(ls -t output/spider2/grast_bird_reranker-$model_name-with-table-meaning/ | grep best_roc_auc_epoch_ | head -n 1)
echo "BEST_CHECKPOINT: $BEST_CHECKPOINT"

source activate bge
python -u frozen_encoder_trainer/evaluate_with_frozen_embeddings.py   \
    --dev_embeddings_dir output/embeddings/spider2-dev-reranker-$model_name-with-table-meaning   \
    --dataset spider2   \
    --reranker_type qwen   \
    --num_layers 2   \
    --hid_dim 1024   \
    --checkpoint output/spider2/grast_bird_reranker-$model_name-with-table-meaning/$BEST_CHECKPOINT   \
    --k 30 40 100 200 300   \
    --log-low-recall \
    --log-file logs/spider2/qwen-$model_name/grast_bird_reranker_$model_name-with-table-meaning_dev.json > logs/spider2/running_qwen-$model_name-spider2.txt


