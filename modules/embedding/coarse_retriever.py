"""Stage-I coarse retriever — bi-encoder embedding -> top-M candidate columns per question.

First stage of the GRAST-SL pipeline. A sentence-transformer bi-encoder embeds each column's
schema-aware text (question-independent) and the question, then keeps the top-M columns by
dot-product similarity. The resulting candidate pool feeds the question-aware column encoder
(`modules.column_encoder`), the deep-attention GNN ranker (`modules.graph_reranker`), and the
table-aware Steiner closure (`modules.steiner_tree_spanner`).
"""
from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from modules.column_encoder.init_embeddings import make_desc

# Prompt prepended to the question before encoding (matches the trained Stage-I encoder).
INSTRUCT = "Instruct: Retrieve columns used for writing SQL for given question.\nQuery: "


class CoarseRetriever:
    """Wraps a bi-encoder and returns the top-M columns of a schema graph for a question."""

    def __init__(self, model_path: str, device: str = "cuda",
                 max_seq_length: int = 512, dtype=torch.float16):
        self.model = SentenceTransformer(
            model_path, trust_remote_code=True, device=device,
            model_kwargs={"torch_dtype": dtype},
        )
        self.model.max_seq_length = max_seq_length

    def embed_columns(self, nodes: List[dict], batch_size: int = 256) -> np.ndarray:
        """Embed column node-attribute dicts (from the FD graph) -> [N, d] normalized vectors."""
        descs = [make_desc(n) for n in nodes]
        return self.model.encode(descs, batch_size=batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True)

    def embed_query(self, question: str) -> np.ndarray:
        return self.model.encode(INSTRUCT + question,
                                 normalize_embeddings=True, convert_to_numpy=True)

    def retrieve(self, G, question: str, top_m: int) -> Tuple[List[str], np.ndarray]:
        """Return (top_names, column_embeddings) for the top-M columns of graph ``G`` by
        similarity to ``question``. When the schema has <= top_m columns, all are kept (ranked).
        ``column_embeddings`` is aligned to ``top_names`` (ranked order)."""
        names = sorted(G.nodes())
        col_vecs = self.embed_columns([G.nodes[c] for c in names])   # [N, d]
        q_vec = self.embed_query(question)                           # [d]
        order = np.argsort(-(col_vecs @ q_vec))[:top_m]
        top_names = [names[i] for i in order]
        return top_names, col_vecs[order]
