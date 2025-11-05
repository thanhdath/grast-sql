#!/usr/bin/env python3
"""
Column filtering: Get top-k columns for a given question using GRAST-SQL model.

This script:
1. Loads a pre-built functional dependency graph
2. Uses GRAST-SQL model (GNN + embeddings) to score columns
3. Returns top-k columns based on model scores
4. Displays results with column metadata

Usage:
    python filter_columns.py \
        --graph schema_graph.pkl \
        --question "List all products with their prices" \
        --top-k 10 \
        --checkpoint /path/to/checkpoint.pt \
        --embeddings-dir /path/to/embeddings/
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple
import torch
from torch.cuda.amp.autocast_mode import autocast
import networkx as nx

from modules.column_encoder.init_embeddings import make_desc, EmbeddingInitializer
from modules.graph_reranker.data import graph_to_data_with_embeddings
from modules.graph_reranker.model import GraphColumnRetrieverFrozen

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_top_k_columns_with_model(
    question: str,
    graph: nx.DiGraph,
    k: int,
    gnn: GraphColumnRetrieverFrozen,
    emb_init: EmbeddingInitializer,
    batch_size: int,
    device_str: str,
) -> List[Tuple[str, float]]:
    """Filter top-k columns using GRAST-SQL model."""
    names = sorted(graph.nodes())
    descs = [make_desc(graph.nodes[n]) for n in names]
    
    # Generate embeddings on-the-fly
    embeds_list = []
    for i in range(0, len(descs), batch_size):
        batch_descs = descs[i:i + batch_size]
        batch_embeds = emb_init.encode_pairs([question] * len(batch_descs), [batch_descs])
        embeds_list.append(batch_embeds)
    embeds = torch.cat(embeds_list, dim=0)
    
    # Convert graph to PyTorch Geometric format
    pyg = graph_to_data_with_embeddings(graph, question, [], embeds).to(device_str)
    
    # Run inference
    with torch.no_grad(), autocast():
        logits = gnn(pyg)
    
    # Get scores and sort
    scores = logits.cpu().numpy()
    column_scores = [(name, float(score)) for name, score in zip(names, scores)]
    column_scores.sort(key=lambda x: x[1], reverse=True)
    return column_scores[:k]


def main():
    parser = argparse.ArgumentParser(description="Filter top-k columns using GRAST-SQL model")
    parser.add_argument("--graph", type=Path, required=True,
                       help="Path to the graph pickle file (from init_schema.py)")
    parser.add_argument("--question", type=str, required=True,
                       help="Natural language question about the database")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of top columns to retrieve (default: 10)")
    parser.add_argument("--checkpoint", type=Path,
                       default=Path("griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker/layer-3-hidden-2048.pt"),
                       help="Path to GNN checkpoint file")
    parser.add_argument("--encoder-path", type=str,
                       default="griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker",
                       help="Path to encoder model (default: griffith-bigdata/GRAST-SQL-0.6B-BIRD-Reranker)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for embedding generation (default: 32)")
    parser.add_argument("--max-length", type=int, default=4096,
                       help="Maximum sequence length (default: 4096)")
    parser.add_argument("--hidden-dim", type=int, default=2048,
                       help="Hidden dimension for GNN (default: 2048)")
    parser.add_argument("--num-layers", type=int, default=3,
                       help="Number of GNN layers (default: 3)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("GRAST-SQL Column Filtering")
    print("=" * 80)
    print()
    
    # Load graph
    print(f"Loading graph from {args.graph}")
    with open(args.graph, "rb") as f:
        graph = pickle.load(f)
    print(f"  ✓ Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print()
    
    # Initialize embedding model
    print("Initializing embedding model...")
    emb_init = EmbeddingInitializer(
        model_path=args.encoder_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    embed_dim = emb_init.embed_dim()
    print(f"  ✓ Embedding model ready: embed_dim={embed_dim}")
    print()
    
    # Load GNN model
    print(f"Loading GNN checkpoint from {args.checkpoint}")
    gnn = GraphColumnRetrieverFrozen(
        embed_dim=embed_dim,
        hid_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    gnn = gnn.to(device_str)
    
    chk = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    gnn.load_state_dict(chk.get("model_state_dict", chk), strict=False)
    gnn.eval()
    print(f"  ✓ GNN loaded: {args.num_layers} layers, hidden_dim={args.hidden_dim}, embed_dim={embed_dim}")
    print()
    
    # Filter top-k columns
    print(f"Filtering top-{args.top_k} columns for question: '{args.question}'")
    top_columns = filter_top_k_columns_with_model(
        args.question, graph, args.top_k, gnn, emb_init, args.batch_size, device_str
    )
    print()
    
    print("Top columns:")
    print("-" * 80)
    for i, (col_name, score) in enumerate(top_columns, 1):
        node = graph.nodes[col_name]
        print(f"{i}. {col_name} (score: {score:.4f})")
        print(f"   Meaning: {node.get('meaning', 'N/A')}")
        print(f"   Type: {node.get('type', 'N/A')}")
        samples = node.get('similar_values', [])[:2]
        print(f"   Examples: {', '.join(map(str, samples))}")
        print()
    
    print("=" * 80)
    print("Filtering completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
