"""Column vector index for Stage-I coarse retrieval (paper: Column Embedding Initialization).

Offline step: precompute a bi-encoder embedding for every column of a database schema and store
it in a persistent ChromaDB collection. At query time, embed the question and run nearest-neighbor
(inner-product) search to get the top-M candidate columns — the input to the question-aware encoder,
deep-attention GNN, and table-aware Steiner closure.

Build once per database; reuse across queries.
"""
from typing import List, Optional

from .coarse_retriever import CoarseRetriever


class ColumnIndex:
    """Persistent ChromaDB index of column embeddings for one or more databases."""

    def __init__(self, retriever: CoarseRetriever, persist_dir: str, collection: str = "grast_columns"):
        import chromadb  # lazy import; only needed when building/using a persistent index
        self.retriever = retriever
        self.client = chromadb.PersistentClient(path=persist_dir)
        # inner product == cosine because the retriever returns L2-normalized vectors
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "ip"})

    def build(self, G, db_id: str = "") -> int:
        """Embed every column of schema graph ``G`` and upsert into the index. Returns #columns."""
        names = sorted(G.nodes())
        if not names:
            return 0
        vecs = self.retriever.embed_columns([G.nodes[c] for c in names])
        self.col.upsert(
            ids=[f"{db_id}::{n}" for n in names],
            embeddings=[v.tolist() for v in vecs],
            metadatas=[{"db_id": db_id, "column": n} for n in names],
        )
        return len(names)

    def query(self, question: str, top_m: int, db_id: Optional[str] = None) -> List[str]:
        """Return the top-M column names for ``question`` (optionally scoped to one ``db_id``)."""
        q = self.retriever.embed_query(question)
        where = {"db_id": db_id} if db_id else None
        res = self.col.query(query_embeddings=[q.tolist()], n_results=top_m, where=where)
        return [m["column"] for m in res["metadatas"][0]]
