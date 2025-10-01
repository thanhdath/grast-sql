mkdir -p logs/param-sensitivity

model_name=qwen3-0.6B-finetuned
# SPIDER dataset, train set = SPIDER train, validation set = None
for layer in 0 1 2 3 4
do
    for hid in 256 512 1024 2048
    do
        embeddings_dir_train=output/embeddings/spider-train-reranker-$model_name-with-table-meaning
        output_dir=output/spider/grast_spider_reranker-$model_name-with-table-meaning/layer-$layer-hidden-$hid
        log_file=logs/param-sensitivity/spider-qwen-layer-$layer-hidden-$hid.log

        python3 -u frozen_encoder_trainer/train_with_frozen_embeddings.py \
            --embeddings_dir $embeddings_dir_train \
            --dataset spider \
            --reranker_type qwen \
            --num_layers $layer \
            --hid_dim $hid \
            --num_epochs 40 \
            --batch_size 32 \
            --learning_rate 5e-5 \
            --output_dir $output_dir \
            --seed 42 > $log_file

        echo "=== EVALUATION ===" >> $log_file
        # evaluation (best model = last epoch)
        python3 -u frozen_encoder_trainer/evaluate_with_frozen_embeddings.py \
            --dev_embeddings_dir output/embeddings/spider-dev-reranker-$model_name-with-table-meaning \
            --dataset spider \
            --reranker_type qwen \
            --num_layers $layer \
            --hid_dim $hid \
            --checkpoint $output_dir/last_epoch_40.pt >> $log_file
    done

done



# BIRD dataset, train set = BIRD train, validation set = None
for layer in 0 1 2 3 4
do
    for hid in 256 512 1024 2048
    do
        embeddings_dir_train=output/embeddings/bird-train-reranker-$model_name-with-table-meaning
        output_dir=output/bird/grast_bird_reranker-$model_name-with-table-meaning/layer-$layer-hidden-$hid
        log_file=logs/param-sensitivity/bird-qwen-layer-$layer-hidden-$hid.log

        python3 -u frozen_encoder_trainer/train_with_frozen_embeddings.py \
        --embeddings_dir $embeddings_dir_train \
        --dataset bird \
        --reranker_type qwen \
        --num_layers $layer \
        --hid_dim $hid \
        --num_epochs 40 \
        --batch_size 32 \
        --learning_rate 5e-5 \
        --output_dir $output_dir \
        --seed 42 > $log_file

        echo "=== EVALUATION ===" >> $log_file
        # evaluation (best model = last epoch)
        python3 -u frozen_encoder_trainer/evaluate_with_frozen_embeddings.py \
        --dev_embeddings_dir output/embeddings/bird-dev-reranker-$model_name-with-table-meaning \
        --dataset bird \
        --reranker_type qwen \
        --num_layers $layer \
        --hid_dim $hid \
        --checkpoint $output_dir/last_epoch_40.pt >> $log_file
    done

done


# Spider 2.0-lite dataset, train set = BIRD train, validation set = BIRD dev
for layer in 0 1 2 3 4
do
    for hid in 256 512 1024 2048
    do
        embeddings_dir_train=output/embeddings/bird-train-reranker-$model_name-with-table-meaning
        embeddings_dir_dev=output/embeddings/bird-dev-reranker-$model_name-with-table-meaning
        output_dir=output/spider2/grast_spider2_reranker-$model_name-with-table-meaning/layer-$layer-hidden-$hid
        log_file=logs/param-sensitivity/spider2-qwen-layer-$layer-hidden-$hid.log

        python3 -u frozen_encoder_trainer/train_with_frozen_embeddings.py \
        --embeddings_dir $embeddings_dir_train \
        --dev_embeddings_dir $embeddings_dir_dev \
        --dataset spider2 \
        --reranker_type qwen \
        --num_layers $layer \
        --hid_dim $hid \
        --num_epochs 40 \
        --batch_size 32 \
        --learning_rate 5e-5 \
        --output_dir $output_dir \
        --seed 42 > $log_file

        echo "=== EVALUATION ===" >> $log_file
        # evaluation (best model = best ROC model)
        BEST_CHECKPOINT=$(ls -t $output_dir/ | grep best_roc_auc_epoch_ | head -n 1)
        echo "BEST_CHECKPOINT: $BEST_CHECKPOINT"
        python3 -u frozen_encoder_trainer/evaluate_with_frozen_embeddings.py \
        --dev_embeddings_dir $embeddings_dir_dev \
        --dataset spider2 \
        --reranker_type qwen \
        --num_layers $layer \
        --hid_dim $hid \
        --checkpoint $output_dir/$BEST_CHECKPOINT >> $log_file
    done

done
