## GRAST-SL: Scaling Text-to-SQL via LLM-efficient Schema Filtering System with Functional Dependency Graph Rerankers
![visitors](https://visitor-badge.laobi.icu/badge?page_id=thanhdath.grast-sql)
<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt="Contrib"/>

GRAST-SL is a **lightweight schema-filtering framework** that scales Text-to-SQL to very wide schemas by removing irrelevant columns while preserving join connectivity. It's a **scoring-based** method, able to set threshold to keep high recall, can scale to large databases like Spider 2.0 Snow. This schema linking system aims to **reduce tables, columns -> reduce prompt tokens** to prompt Text-to-SQL LLM in large databases.

**[14-06-2026] V2**: Coming.

- Better P under high recall for Spider 2.0 Snow, BIRD and Spider.
- Lower latency compared to V1 by adding a coarse embedding retriever.
- Enhance GNN structural with global attention over MxM column embeddings. 
- Enhance Steiner Tree Spanner.
- 1 model (total parameters 0.8B) works for all three dataset Spider, BIRD, Spider 2.0 Snow.

**V1 Pre-print**: [![arXiv](https://img.shields.io/badge/arXiv-2512.16083-b31b1b.svg)](https://arxiv.org/abs/2512.16083)

---

# ⬇️ V1

_Everything below — models, datasets, and reproduction — is V1._

### Models & datasets

All trained models (0.6B / 4B / 8B) and the Spider, BIRD, and Spider 2.0-lite datasets are in the [griffith-bigdata Hugging Face collection](https://huggingface.co/collections/griffith-bigdata/grast-sql). Models trained on BIRD also apply to Spider 2.0-lite (Spider 2.0 has no training set).

<!-- ![GRAST-SL main flow](figures/main-flow.png) -->

### Repository structure

- `modules/` — core components: `embedding` (Stage-I bi-encoder coarse retrieval → top-M pool), `column_encoder` (question-aware column encoder / GNN node features), `graph_reranker` (deep-attention GNN reranker), `schema_enricher` (FD graph + metadata/keys/values), `steiner_tree_spanner` (join-connectivity closure), `db_content_retrieval` (BM25 value search).
- `train_and_evaluate/` — 2-stage training: Stage-1 encoder (FlagEmbedding reranker), Stage-2 GNN (`train_with_frozen_embeddings.py`).
- `data_processing/` — Spider/BIRD/Spider 2.0 preprocessing; `scripts/` — train/eval helpers; `visualization/` — analysis assets.
- `init_schema.py` (build FD graph + metadata), `filter_columns.py` (top-K columns for a question), `evaluate_on_the_fly.py` (end-to-end P/R eval), `environment.yaml` (conda env).

## Setup

```bash
conda env create -n grast-sql --file environment.yaml && conda activate grast-sql
```

## Evaluation For V1

Start the vLLM embedding server, then run the end-to-end eval (BIRD dev, 0.6B; set `--dataset` to `spider`/`bird`/`spider2`):

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker \
  --port 8000 --max-model-len 8192 --tensor-parallel-size 2 --task embedding --gpu-memory-utilization 0.8

python -u evaluate_on_the_fly.py --dataset bird --split dev --evaluation_mode end2end \
  --reranker_type qwen --hidden_dim 2048 --num_layers 3 \
  --encoder_path griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker \
  --checkpoint griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker/layer-3-hidden-2048.pt \
  --pkl_path data/bird_dev_samples_graph.pkl --k 30 --batch_size 128 \
  --log_dir logs/bird_dev_topk30_qwen_0.6b --pred_collection grast_qwen_0.6b_bird_dev
```

## Apply to your own database

```bash
# 1) one-time per DB: build FD graph + metadata (needs OPENAI_API_KEY in .env)
python init_schema.py --db-path your_db.sqlite --output schema.pkl --model gpt-4.1-mini

# 2) filter the top-K columns for a question
python filter_columns.py --graph schema.pkl --top-k 5 \
  --question "Show name, country, age for all singers ordered by age from the oldest to the youngest."
```

`filter_columns.py` defaults `--checkpoint`/`--encoder-path` to `griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker` (`--hidden-dim 2048 --num-layers 3`).

## Citation:
```
@article{hoang2025scaling,
  title={Scaling Text2SQL via LLM-efficient Schema Filtering with Functional Dependency Graph Rerankers},
  author={Hoang, Thanh Dat and Nguyen, Thanh Tam and Huynh, Thanh Trung and Yin, Hongzhi and Nguyen, Quoc Viet Hung},
  journal={arXiv preprint arXiv:2512.16083},
  year={2025}
}
```

-----------
**Backup Statistics**

![Visitors](https://margherita-gustatory-zane.ngrok-free.dev/badge/thanhdath%2Fgrast-sql.svg?ngrok-skip-browser-warning=true)
