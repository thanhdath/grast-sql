"""
Data utilities for Graph Reranker: loading embeddings/metadata and building PyG Data objects.
"""

from pathlib import Path
from typing import Dict, Any, Sequence, List

import io
import json
import pickle

import networkx as nx
import torch
from torch_geometric.data import Data


EDGE_TYPE_MAP = {
    "foreign_key": 0,
    "col_to_foreign_key": 1,
    "col_to_primary_key": 1,
}


def load_embeddings_and_metadata(embeddings_dir: Path, dataset: str, reranker_type: str, split: str):
    """Load embeddings and metadata from a directory that contains exactly:
    - one embeddings .pkl file
    - one metadata*.json file

    The function intentionally ignores dataset/reranker/split naming and
    simply picks the single matching files, keeping the logic minimal and
    directory-driven.
    """
    assert split in ("train", "dev"), f"Invalid split: {split}"

    # Locate embeddings (.pkl)
    emb_candidates = list(embeddings_dir.glob("*.pkl"))
    if len(emb_candidates) != 1:
        names = ", ".join(p.name for p in emb_candidates) or "<none>"
        raise FileNotFoundError(
            f"Expected exactly one .pkl in {embeddings_dir}, found {len(emb_candidates)}: {names}"
        )
    emb_path = emb_candidates[0]

    # Locate metadata (metadata*.json)
    meta_candidates = list(embeddings_dir.glob("metadata*.json"))
    if len(meta_candidates) != 1:
        names = ", ".join(p.name for p in meta_candidates) or "<none>"
        raise FileNotFoundError(
            f"Expected exactly one metadata*.json in {embeddings_dir}, found {len(meta_candidates)}: {names}"
        )
    metadata_path = meta_candidates[0]

    print(f"[INFO] Using embeddings: {emb_path.name}")
    print(f"[INFO] Using metadata  : {metadata_path.name}")

    # CPU-safe load
    orig_load_from_bytes = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu')
    try:
        with open(emb_path, 'rb') as f:
            embeddings = pickle.load(f)
    except Exception:
        with open(emb_path, 'rb') as f:
            embeddings = torch.load(f, map_location=torch.device('cpu'))
    finally:
        torch.storage._load_from_bytes = orig_load_from_bytes

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded {len(embeddings)} {split} samples")
    print(f"Embedding dimension: {metadata['embed_dim']}")
    return embeddings, metadata



def graph_to_data_with_embeddings(g: nx.DiGraph, query: str,
                                  true_cols: Sequence[str],
                                  embeddings: torch.Tensor) -> Data:
    """Convert graph to PyTorch Geometric Data with pre-computed embeddings."""
    names = sorted(g.nodes())
    idx = {n: i for i, n in enumerate(names)}

    # Pre-computed node embeddings (Tensor: [num_nodes, embed_dim])
    x = embeddings

    labels = torch.tensor([1.0 if n in true_cols else 0.0 for n in names], dtype=torch.float32)

    e_src, e_dst, e_attr = [], [], []
    for u, v, ed in g.edges(data=True):
        if u not in idx or v not in idx:
            continue
        e_src.append(idx[u])
        e_dst.append(idx[v])
        et = EDGE_TYPE_MAP.get(ed.get("edge_type", "foreign_key"), 1)
        e_attr.append([1.0, 0.0] if et == 0 else [0.0, 1.0])

    if not e_src:  # no edges - create self-edges for all nodes
        # Create self-edges: each node connects to itself
        e_src = list(range(len(names)))
        e_dst = list(range(len(names)))
        e_attr = [[0.0, 1.0] for _ in range(len(names))]  # Use "other" edge type for self-edges
        edge_index = torch.tensor([e_src, e_dst], dtype=torch.long)
        edge_attr = torch.tensor(e_attr, dtype=torch.float32)
    else:
        edge_index = torch.tensor([e_src, e_dst], dtype=torch.long)
        edge_attr = torch.tensor(e_attr, dtype=torch.float32)

    data = Data(x=x, q_raw=query, orig_names=names,
                y=labels, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = len(names)
    return data



def create_dataset_from_embeddings(embeddings_dict: Dict[str, Any]) -> List[Data]:
    """Create dataset from pre-computed embeddings (new format)."""
    print(f"Creating dataset from embeddings…")
    data: List[Data] = []
    for emb in embeddings_dict.values():
        q = emb['query']
        embeddings = emb['embeddings']
        positives = emb['positives']
        G = emb['G']
        # Create Data object
        data_obj = graph_to_data_with_embeddings(G, q, positives, embeddings)
        data.append(data_obj)
    print(f"Created {len(data)} data samples")
    return data 