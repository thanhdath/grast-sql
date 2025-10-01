"""
frozen_encoder_trainer/utils.py
──────────────────────────────
Reusable helpers for graph-schema experiments.
"""

from typing import Sequence, Tuple
import networkx as nx
from networkx.algorithms.approximation import steiner_tree

__all__ = ["get_steiner_subgraph", "pr", "pr_at_k"]


# ───────────────────────────── Graph helper ──────────────────────────────
def get_steiner_subgraph(G: "nx.Graph", terminals: Sequence[str]) -> "nx.Graph":
    """
    Return an (approximate) Steiner forest connecting *terminals* in *G*.

    Works per connected component – if a component has ≥2 terminals we run the
    NetworkX approximation, otherwise we just keep the lone terminal.
    """
    G_u = G.to_undirected()
    forest = nx.Graph()
    terms = set(terminals)

    for comp_nodes in nx.connected_components(G_u):
        comp_terms = terms & comp_nodes
        if not comp_terms:
            continue

        sub = G_u.subgraph(comp_nodes).copy()
        if len(comp_terms) == 1:
            n = next(iter(comp_terms))
            forest.add_node(n, **G.nodes[n])
        else:
            forest = nx.compose(forest, steiner_tree(sub, comp_terms))

    return forest


# ─────────────────────── Precision / Recall helpers ──────────────────────
def pr(preds: Sequence[str], gold: Sequence[str]) -> Tuple[float, float]:
    """
    Precision & recall between *preds* and *gold* (both iterables of strings).
    """
    tp = len(set(preds) & set(gold))
    precision = tp / len(preds) if preds else 0.0
    recall    = tp / len(gold)  if gold else 0.0
    return precision, recall


def pr_at_k(preds: Sequence[str], gold: Sequence[str], k: int) -> Tuple[float, float]:
    """Precision & recall at rank *k* (uses the first *k* predictions)."""
    return pr(preds[:k], gold)
