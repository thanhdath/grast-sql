## GRAST-SQL: Scaling Text-to-SQL via LLM-efficient Schema Filtering with Functional Dependency Graph Rerankers

![visitors](https://visitor-badge.laobi.icu/badge?page_id=thanhdath.grast-sql)

GRAST-SQL is a lightweight, open-source schema-filtering framework that scales Text-to-SQL to real-world, very wide schemas by compacting prompts without sacrificing accuracy. It ranks columns with a query-aware LLM encoder enriched by values/metadata, reranks them via a graph transformer over a functional-dependency (FD) graph to capture inter-column structure, and then guarantees joinability with a Steiner-tree spanner to produce a small, connected sub-schema. Across Spider, BIRD, and Spider-2.0-lite, GRAST-SQL delivers near-perfect recall with substantially higher precision than CodeS, SchemaExP, Qwen rerankers, and embedding retrievers, maintains sub-second median latency on typical databases, scales to 23K+ columns, and cuts prompt tokens by up to 50% in end-to-end systems—often with slight accuracy gains—all while using compact models. This repository provides the trained models, code, and datasets to reproduce results and apply GRAST-SQL to your own databases.

### Datasets
- **Spider**: [Spider Evaluation Dataset](https://huggingface.co/datasets/griffith-bigdata/GRAST-SQL-Spider)
- **BIRD**: [BIRD Training/Evaluation Dataset](https://huggingface.co/datasets/griffith-bigdata/GRAST-SQL-BIRD)
- **Spider-2.0-lite**: [Spider 2.0-lite Eval Dataset](https://huggingface.co/datasets/griffith-bigdata/GRAST-SQL-Spider2.0-lite)

### Models
- **GRAST-SQL 0.6B**: [GRAST-SQL 0.6B BIRD](https://huggingface.co/griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker)
- **GRAST-SQL 4B**: [GRAST-SQL 4B BIRD](https://huggingface.co/griffith-bigdata/GRAST-SQL-4B-BIRD-Reranker)
- **GRAST-SQL 8B**: [GRAST-SQL 8B BIRD](https://huggingface.co/griffith-bigdata/GRAST-SQL-8B-BIRD-Reranker)

More models can be found in [Huggingface collection](https://huggingface.co/collections/griffith-bigdata/grast-sql)

> Note: Models trained on BIRD also apply to Spider-2.0-lite, since Spider-2.0 does not include a training set.

### System flow
![GRAST-SQL main flow](figures/main-flow.png)

### Repository structure

```
grast-sql/
├── modules/                    # Core model and algorithmic components
│   ├── column_encoder/        # Query-aware column embedding generation using LLM encoders
│   ├── graph_reranker/       # Graph transformer model for column reranking
│   ├── schema_enricher/       # Schema metadata enrichment (meanings, keys, types, values)
│   ├── steiner_tree_spanner/  # Steiner tree algorithm for connectivity guarantee
│   └── db_content_retrieval/ # Database content retrieval utilities, using BM25 algorithm
│
├── train_and_evaluate/        # 2-stage training algorithm
│                               # - create_llm_reranker_data.py: Data preparation
|                               # - 1st stage was trained using FlagEmbedding Reranker repo, see (/home/datht/grast-sql/train_and_evaluate/scripts)
│                               # - train_with_frozen_embeddings.py: GNN training
│
├── data_processing/           # Dataset preprocessing and preparation
│   ├── spider-bird/          # Spider and BIRD dataset processing scripts
│   └── spider2.0/            # Spider 2.0 dataset processing scripts
│
├── scripts/                   # Helper shell scripts for training and evaluation which were used for conducting this research
│                               # - evaluate_topk.sh: Top-k evaluation
│                               # - train_main_grast.sh: Main training script
│                               # - eval_*.sh: Various evaluation scripts
│
├── figures/                   # Documentation figures and diagrams
│
├── visualization/             # Visualization utilities and analysis assets
│
├── data/                      # Processed data files (pickle files, graphs)
├── logs/                      # Evaluation logs and results
├── output/                    # Model checkpoints and training outputs
│
├── init_schema.py             # Initialize schema: Extract schema from database, generate 
│                               #   metadata (meanings, keys), and build functional dependency graph -> save graph to .pkl file for further uses
│
├── filter_columns.py          # Filter top-K columns for a given natural language query
│                               #   using GRAST-SQL model (GNN + embeddings + Steiner tree)
│
├── evaluate_on_the_fly.py     # End-to-end evaluation script: Embed query-column pairs on-the-fly,
│                               #   run GNN reranker, evaluate precision/recall on datasets
│                               #   (Spider, BIRD, Spider-2.0-lite)
│
├── environment.yaml           # Conda environment specification for reproducible setup
│
└── README.md                  
```

## Setup


Create a conda environment using the provided environment file:

```bash
conda env create -n grast-sql --file environment.yaml
conda activate grast-sql
```

## Evaluation

To reproduce the evaluation results on BIRD dev set with the 0.6B model:

### Step 1: Start the vLLM Server

Start the vLLM server for GRAST-SQL, keep this server running in a separate terminal or background process:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker \
  --port 8000 \
  --max-model-len 8192 \
  --tensor-parallel-size 2 \
  --task embedding \
  --gpu-memory-utilization 0.8
```

### Step 2: Run Evaluation

Run the end-to-end evaluation script:

```bash
python -u evaluate_on_the_fly.py \
  --dataset bird \
  --split dev \
  --evaluation_mode end2end \
  --reranker_type qwen \
  --hidden_dim 2048 \
  --num_layers 3 \
  --encoder_path griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker \
  --checkpoint griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker/layer-3-hidden-2048.pt \
  --pkl_path data/bird_dev_samples_graph.pkl \
  --k 30 \
  --batch_size 128 \
  --log_dir logs/bird_dev_topk30_qwen_0.6b \
  --pred_collection grast_qwen_0.6b_bird_dev
```

**Arguments:**
- `--dataset`: Dataset name (e.g., `bird`, `spider`, `spider2`)
- `--split`: Data split (e.g., `dev`, `train`)
- `--evaluation_mode`: Evaluation mode (e.g., `end2end`)
- `--reranker_type`: Type of reranker (e.g., `qwen`)
- `--hidden_dim`: Hidden dimension for GNN (default: 2048)
- `--num_layers`: Number of GNN layers (default: 3)
- `--encoder_path`: Path to encoder model from HuggingFace
- `--checkpoint`: Path to GNN checkpoint from HuggingFace
- `--pkl_path`: Path to the graph pickle file
- `--k`: Top-K columns to retrieve
- `--batch_size`: Batch size for evaluation
- `--log_dir`: Directory to save evaluation logs
- `--pred_collection`: Collection name for predictions


## Plugging GRAST-SQL on Custom Database

To apply GRAST-SQL to your own database, follow these two simple steps:

### Step 1: Initialize - Functional Dependency Graph Construction & Metadata Completion

Extract schema information, generate table/column meanings, predict missing keys, and build the functional dependency graph:

```bash
python init_schema.py \
    --db-path /home/datht/mats/data/spider/database/concert_singer/concert_singer.sqlite \
    --output concert_singer.pkl \
    --model gpt-4.1-mini
```

**Arguments:**
- `--db-path`: Path to your SQLite database file (required)
- `--output`: Output path for the graph pickle file (default: `schema_graph.pkl`)
- `--model`: OpenAI model to use for meaning generation and key prediction (default: `gpt-4.1-mini`)

**Note:** Make sure your OpenAI API key is set in `.env`.

### Step 2: Filter Top-K Columns

Use the GRAST-SQL model to filter the most relevant columns for a given question:

```bash
python filter_columns.py \
    --graph concert_singer.pkl \
    --question "Show name, country, age for all singers ordered by age from the oldest to the youngest." \
    --top-k 5
```

**Arguments:**
- `--graph`: Path to the graph pickle file from Step 1 (required)
- `--question`: Natural language question about the database (required)
- `--top-k`: Number of top columns to retrieve (default: 10)
- `--checkpoint`: Path to GNN checkpoint (default: `griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker/layer-3-hidden-2048.pt`)
- `--encoder-path`: Path to encoder model (default: `griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker`)
- `--max-length`: Maximum sequence length (default: 4096)
- `--batch-size`: Batch size for embedding generation (default: 32)
- `--hidden-dim`: Hidden dimension for GNN (default: 2048)
- `--num-layers`: Number of GNN layers (default: 3)


## Citation:
```
@misc{hoang2025scalingtext2sqlllmefficientschema,
      title={Scaling Text2SQL via LLM-efficient Schema Filtering with Functional Dependency Graph Rerankers}, 
      author={Thanh Dat Hoang and Thanh Tam Nguyen and Thanh Trung Huynh and Hongzhi Yin and Quoc Viet Hung Nguyen},
      year={2025},
      eprint={2512.16083},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2512.16083}, 
}
```

-----------
**Backup Statistics**

![Visitors](https://margherita-gustatory-zane.ngrok-free.dev/badge/thanhdath%2Fgrast-sql.svg?ngrok-skip-browser-warning=true)

