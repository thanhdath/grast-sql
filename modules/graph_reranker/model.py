"""
Graph Reranker model and evaluation helpers.
"""
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch_geometric.data import Batch
from torch_geometric.nn import TransformerConv
from torch_geometric.loader import DataLoader


class GraphColumnRetrieverFrozen(nn.Module):
    def __init__(self, embed_dim: int, hid_dim: int = 1024, num_layers: int = 2, edge_dim: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_gnn = num_layers > 0

        if self.use_gnn:
            self.gnn = nn.ModuleList([
                TransformerConv(in_channels=embed_dim if i == 0 else hid_dim,
                                out_channels=hid_dim, heads=4, concat=False,
                                edge_dim=edge_dim, dropout=0.1)
                for i in range(num_layers)
            ])
            proj_in = hid_dim
        else:
            proj_in = embed_dim

        self.cls_head = nn.Linear(proj_in, 1)

    def forward(self, data: Batch):
        x = data.x  # type: ignore[attr-defined]  # Pre-computed embeddings

        if self.use_gnn and x.size(0):
            # Always use GNN layers since we now have edges (either real or self-edges)
            ea = data.edge_attr if data.edge_attr.numel() else None  # type: ignore[attr-defined]
            for layer in self.gnn:
                x = F.relu(layer((x, x), data.edge_index, ea))  # type: ignore[attr-defined]

        return self.cls_head(x).squeeze(-1)
