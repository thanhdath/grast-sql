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
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch


class GlobalAttnLayer(nn.Module):
    """M x M column-embedding attention
    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1, ffn_mult: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_mult * dim, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, batch):
        # x: [num_nodes, dim] node list; batch: [num_nodes] graph-id per node
        dense, mask = to_dense_batch(x, batch)  # dense:[G,Lmax,dim] mask:[G,Lmax] True=real
        key_pad = ~mask  # True where padding -> ignored by attention
        h = self.norm1(dense)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_pad, need_weights=False)
        dense = dense + self.drop(attn_out)
        h = self.norm2(dense)
        dense = dense + self.drop(self.ffn(h))
        return dense[mask]  # scatter back to node list [num_nodes, dim]


class GraphColumnRetrieverFrozen(nn.Module):
    # Deep-attention column ranker (GRAST-SQL): 
    # + local FD message passing view
    # + global per-query column-attention view
    # -> fuses 2 views 
    # + post-GNN attention stack,
    # + residual skip connection
    def __init__(self, embed_dim: int, hid_dim: int = 2048, num_layers: int = 3, edge_dim: int = 2,
                 mlp_head: bool = False, apply_norms: bool = False, skip_connection: bool = True,
                 jk_concat: bool = False, dropout: float = 0.1,
                 global_attn: str = "hybrid", attn_heads: int = 8,
                 size_head: bool = False, attn_post: int = 2, attn_gate: bool = False):
        super().__init__()
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.use_gnn = num_layers > 0
        self.num_layers = num_layers
        self.apply_norms = apply_norms
        self.mlp_head = mlp_head
        
        # column attends to every other column within its own query's schema.
        #   "none"   -> original sparse FK TransformerConv stack (baseline)
        #   "full"   -> variant A: REPLACE sparse stack with N global-self-attention layers
        #   "hybrid" -> variant B: keep FK TransformerConv (local) AND add a global-attn block per layer
        self.global_attn = global_attn
        self.attn_heads = attn_heads

        self.skip_connection = skip_connection
        self.jk_concat = jk_concat

        if self.global_attn == "full" and self.use_gnn:
            # Variant A: pure full attention. Project input -> hid_dim, then N dense self-attn layers.
            self.in_proj = nn.Linear(embed_dim, hid_dim)
            self.gattn = nn.ModuleList([
                GlobalAttnLayer(hid_dim, heads=attn_heads, dropout=dropout)
                for _ in range(num_layers)
            ])
            proj_in = hid_dim
        elif self.use_gnn:
            self.gnn = nn.ModuleList()
            self.norms = nn.ModuleList()  # Thêm LayerNorm
            if self.global_attn == "hybrid":
                # Variant B: a global-attn block running in parallel with each local FK conv.
                self.gattn = nn.ModuleList([
                    GlobalAttnLayer(hid_dim, heads=attn_heads, dropout=dropout)
                    for _ in range(num_layers)
                ])
            for i in range(num_layers):
                in_channels = embed_dim if i == 0 else hid_dim
                self.gnn.append(
                    TransformerConv(in_channels=in_channels,
                                    out_channels=hid_dim,
                                    heads=4,
                                    concat=False,
                                    edge_dim=edge_dim,
                                    dropout=dropout)
                )
                # LayerNorm giúp model ổn định khi node degree thay đổi đột ngột giữa train/test
                self.norms.append(nn.LayerNorm(hid_dim))

            proj_in = hid_dim
        else:
            proj_in = embed_dim

        if self.jk_concat and self.use_gnn:
            # head sees: input embedding + each layer's output
            proj_in = embed_dim + num_layers * hid_dim

        # (enables sigmoid-threshold selection).
        if mlp_head:
            self.cls_head = nn.Sequential(
                nn.Linear(proj_in, proj_in // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(proj_in // 2, 1),
            )
        else:
            self.cls_head = nn.Linear(proj_in, 1)

        if self.skip_connection:
            self.skip_head = nn.Linear(embed_dim, 1)

        self.attn_post = attn_post
        if attn_post and self.use_gnn:
            self.attn_post_layers = nn.ModuleList([
                GlobalAttnLayer(hid_dim, heads=attn_heads, dropout=dropout) for _ in range(attn_post)
            ])
 
        self.attn_gate_enabled = bool(attn_post) and self.use_gnn and bool(attn_gate)
        if self.attn_gate_enabled:
            self.attn_gate = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
            nn.init.constant_(self.attn_gate[-1].bias, 2.0)  # sigmoid(2)=0.88 -> mostly open initially

        self.size_head_enabled = size_head
        if size_head:
            self.size_head = nn.Sequential(
                nn.Linear(2 * proj_in, proj_in), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(proj_in, 1),
            )

    def forward(self, data: Batch):
        x = data.x  # type: ignore[attr-defined]  # Pre-computed embeddings
        x_in = x    # query-aware reranker embedding (for the residual skip)

        feats = [x_in]  # for JK-concat
        # graph-id per node so dense attention never crosses graphs in a batch
        batch_vec = getattr(data, "batch", None)
        if batch_vec is None:
            batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.global_attn == "full" and self.use_gnn and x.size(0):
            # Variant A: pure global self-attention (KaSLA-style joint full attention).
            x = self.in_proj(x)
            for layer in self.gattn:
                x = layer(x, batch_vec)
                feats.append(x)
        elif self.use_gnn and x.size(0):
            # Always use GNN layers since we now have edges (either real or self-edges)
            ea = data.edge_attr if data.edge_attr.numel() else None  # type: ignore[attr-defined]
            for i, layer in enumerate(self.gnn):
                x = layer((x, x), data.edge_index, ea)  # type: ignore[attr-defined]
                if self.global_attn == "hybrid":
                    # Variant B: add global (dense) attention on top of the local FK message passing.
                    x = x + self.gattn[i](x, batch_vec)
                if self.apply_norms:
                    x = self.norms[i](x)   # LayerNorm: stabilizes when node degree shifts train→test
                x = F.relu(x)
                feats.append(x)

        # Deep joint cross-column attention on top of the GNN (RoBERTa-style, per-graph isolated).
        if self.attn_post and self.use_gnn and x.size(0):
            gate = 1.0
            if self.attn_gate_enabled:
                ei = data.edge_index  # type: ignore[attr-defined]
                dg = torch.zeros(x.size(0), device=x.device)
                if ei.numel():
                    dg = dg.scatter_add(0, ei[1], torch.ones(ei.size(1), device=x.device))
                cnt = torch.bincount(batch_vec, minlength=int(batch_vec.max().item()) + 1).float()
                npg = cnt[batch_vec]
                gate = torch.sigmoid(self.attn_gate(torch.stack([torch.log1p(dg), torch.log1p(npg)], dim=-1)))  # [N,1]
            for layer in self.attn_post_layers:
                x = x + gate * layer(x, batch_vec)

        if self.jk_concat and self.use_gnn and x.size(0):
            x = torch.cat(feats, dim=-1)  # [input ⊕ L1 ⊕ ... ⊕ Ln]

        out = self.cls_head(x).squeeze(-1)
        if self.skip_connection:
            out = out + self.skip_head(x_in).squeeze(-1)

        if self.size_head_enabled:
            # Graph-level size prediction. k_logit is per-graph; downstream applies softplus to get
            # a positive predicted count k̂ and selects top-round(k̂) columns per query.
            if x.size(0):
                pooled = torch.cat([global_mean_pool(x, batch_vec),
                                    global_max_pool(x, batch_vec)], dim=-1)  # [G, 2*proj_in]
                k_logit = self.size_head(pooled).squeeze(-1)  # [G]
            else:
                G = int(batch_vec.max().item()) + 1 if batch_vec.numel() else 1
                k_logit = torch.zeros(G, device=x.device)
            return out, k_logit
        return out
