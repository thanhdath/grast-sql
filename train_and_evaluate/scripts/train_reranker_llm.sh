torchrun --nproc_per_node 1 \
	-m FlagEmbedding.finetune.reranker.decoder_only.base \
	--model_name_or_path /home/datht/huggingface/Qwen/Qwen3-Reranker-4B \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --use_flash_attn True \
    --target_modules q_proj k_proj v_proj o_proj \
    --save_merged_lora_model True \
    --model_type decoder \
    --trust_remote_code True \
    --cache_dir ./cache/model \
    --train_data ./data/bird_train_sts_all_cols.jsonl \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 4096 \
    --passage_max_len 4096 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --query_instruction_for_rerank 'Given a natural-language database question (Query) and a Column description (Document), decide if the column may be necessary to answer the question.' \
    --query_instruction_format '<|im_start|>system\nJudge whether the column (Document) is necessary to use when writing the SQL query, based on the provided Query and Instruct. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n<Instruct>: {}\n<Query>: {}' \
    --passage_instruction_for_rerank '\n<Document>: ' \
    --passage_instruction_format '{}{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' \
    --output_dir ./output/finetuned-reranker-qwen3-4B-bird \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --fp16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed ./ds_stage1.json \
    --logging_steps 20 \
    --save_strategy epoch \
    --report_to tensorboard \
    --save_total_limit 2


# torchrun --nproc_per_node 1 \
# 	-m FlagEmbedding.finetune.reranker.decoder_only.base \
# 	--model_name_or_path /home/datht/huggingface/Qwen/Qwen3-Reranker-4B \
#     --use_lora True \
#     --lora_rank 32 \
#     --lora_alpha 64 \
#     --use_flash_attn True \
#     --target_modules q_proj k_proj v_proj o_proj \
#     --save_merged_lora_model True \
#     --model_type decoder \
#     --trust_remote_code True \
#     --cache_dir ./cache/model \
#     --train_data ./data/bird_train_sts_all_cols.jsonl \
#     --cache_path ./cache/data \
#     --train_group_size 8 \
#     --query_max_len 4096 \
#     --passage_max_len 4096 \
#     --pad_to_multiple_of 8 \
#     --knowledge_distillation False \
#     --query_instruction_for_rerank 'Given a natural-language database question (Query) and a Column description (Document), decide if the column may be necessary to answer the question.' \
#     --query_instruction_format '<|im_start|>system\nJudge whether the column (Document) is necessary to use when writing the SQL query, based on the provided Query and Instruct. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n<Instruct>: {}\n<Query>: {}' \
#     --passage_instruction_for_rerank '\n<Document>: ' \
#     --passage_instruction_format '{}{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' \
#     --output_dir ./output/finetuned-reranker-qwen3-4b-bird \
#     --overwrite_output_dir \
#     --learning_rate 2e-4 \
#     --fp16 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --dataloader_drop_last True \
#     --warmup_ratio 0.1 \
#     --gradient_checkpointing \
#     --weight_decay 0.01 \
#     --deepspeed ./ds_stage1.json \
#     --logging_steps 20 \
#     --save_strategy epoch \
#     --report_to tensorboard \
#     --save_total_limit 2
