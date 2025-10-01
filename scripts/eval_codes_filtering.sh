python -u text2sql_zero_shot.py     \
    --llm_path /home/datht/huggingface/seeklhy/codes-1b-spider \
    --sic_path /home/datht/codes/sic_ckpts/sic_spider \
    --table_num 6 \
    --column_num 10 \
    --dataset_path ../codes/data/sft_spider_dev_text2sql_fixed.json \
    --max_tokens 4096 \
    --max_new_tokens 256 \
    --mode eval

python -u text2sql_zero_shot.py     \
    --llm_path /home/datht/huggingface/seeklhy/codes-1b-bird-with-evidence \
    --sic_path /home/datht/codes/sic_ckpts/sic_bird_with_evidence \
    --table_num 6 \
    --column_num 10 \
    --dataset_path ../codes/data/sft_bird_with_evidence_dev_text2sql_fixed.json \
    --max_tokens 4096 \
    --max_new_tokens 256 \
    --mode eval