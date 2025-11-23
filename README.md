## GRAST-SQL: Implementation of "Scaling Text-to-SQL via LLM-efficient Schema Filtering with Functional Dependency Graph Rerankers"

GRAST-SQL is a lightweight, open-source schema-filtering framework that scales Text-to-SQL to real-world, very wide schemas by compacting prompts without sacrificing accuracy. It ranks columns with a query-aware LLM encoder enriched by values/metadata, reranks them via a graph transformer over a functional-dependency (FD) graph to capture inter-column structure, and then guarantees joinability with a Steiner-tree spanner to produce a small, connected sub-schema. Across Spider, BIRD, and Spider-2.0-lite, GRAST-SQL delivers near-perfect recall with substantially higher precision than CodeS, SchemaExP, Qwen rerankers, and embedding retrievers, maintains sub-second median latency on typical databases, scales to 23K+ columns, and cuts prompt tokens by up to 50% in end-to-end systems—often with slight accuracy gains—all while using compact models. This repository provides the trained models, code, and datasets to reproduce results and apply GRAST-SQL to your own databases.

### Datasets
- **Spider**: [Hugging Face collection](https://huggingface.co/collections/griffith-bigdata/grast-sql-68d3a19c2947e2d59f63cf4c)
- **BIRD**: [Hugging Face collection](https://huggingface.co/collections/griffith-bigdata/grast-sql-68d3a19c2947e2d59f63cf4c)
- **Spider-2.0-lite**: [Hugging Face collection](https://huggingface.co/collections/griffith-bigdata/grast-sql-68d3a19c2947e2d59f63cf4c)

### Models
- **GRAST-SQL 0.6B**: [Hugging Face collection](https://huggingface.co/collections/griffith-bigdata/grast-sql-68d3a19c2947e2d59f63cf4c)
- **GRAST-SQL 4B**: [Hugging Face collection](https://huggingface.co/collections/griffith-bigdata/grast-sql-68d3a19c2947e2d59f63cf4c)
- **GRAST-SQL 8B**: [Hugging Face collection](https://huggingface.co/collections/griffith-bigdata/grast-sql-68d3a19c2947e2d59f63cf4c)

> Note: Models trained on BIRD also apply to Spider-2.0-lite, since Spider-2.0 does not include a training set.

### System flow
![GRAST-SQL main flow](figures/main-flow.png)

### Repository structure
- `figures/`: Figures and diagrams for documentation (e.g., `main-flow.png`)
- `models/`: Trained model artifacts and configuration
- `scripts/`: Helper scripts and runners
- `visualization/`: Visualization utilities and analysis assets
- `data_processing/`: Data preparation pipelines (e.g., building FD inputs)
- `training/`: Training recipes and pipelines for encoders/rerankers
- `modules/`: Core model and algorithmic components
- `build_graph.py`: Tools for constructing the functional-dependency (FD) graph
- `evaluate_on_the_fly.py`: End-to-end evaluation and inference entry point
- `environment.yaml`: Reproducible environment specification

## Setup

### Environment Setup

Create a conda environment using the provided environment file:

```bash
conda env create -n grast-sql --file environment.yaml
conda activate grast-sql
```

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
