from typing import List, Tuple
import networkx as nx
from networkx.algorithms.approximation import steiner_tree


def get_steiner_subgraph(G: nx.Graph, terminals: List[str]) -> nx.Graph:
    """Build a Steiner forest on the undirected version of G, per connected component.

    - Works even when terminals span multiple connected components by returning a forest
      composed of per-component Steiner trees.
    - Copies node attributes for single-terminal components so downstream code can access
      metadata on those nodes (e.g., for printing column details).
    """
    G_u = G.to_undirected()
    forest = nx.Graph()
    terms = set(terminals)
    for comp in nx.connected_components(G_u):
        comp_terms = terms & comp
        if not comp_terms:
            continue
        sub = G_u.subgraph(comp).copy()
        if len(comp_terms) == 1:
            node = next(iter(comp_terms))
            forest.add_node(node, **G.nodes[node])
        else:
            forest = nx.compose(forest, steiner_tree(sub, comp_terms))
    return forest


def select_top_k_with_steiner(
    names: List[str],
    scores: List[float],
    G: nx.Graph,
    k: int,
) -> List[Tuple[str, float]]:
    """Rank by score, take top-k terminals, apply Steiner, prioritize Steiner nodes, backfill.

    Returns a list of (name, score) with length up to k.
    """
    name_to_score = {n: float(s) for n, s in zip(names, scores)}
    ranked = sorted(names, key=lambda n: name_to_score[n], reverse=True)
    terminals = ranked[:k]
    st_nodes = list(get_steiner_subgraph(G, terminals).nodes())
    st_set = set(st_nodes)
    prioritized = sorted(st_nodes, key=lambda n: name_to_score.get(n, -1e9), reverse=True)
    result: List[str] = []
    for n in prioritized:
        if len(result) >= k:
            break
        result.append(n)
    if len(result) < k:
        for n in ranked:
            if n in st_set:
                continue
            result.append(n)
            if len(result) >= k:
                break
    return [(n, name_to_score[n]) for n in result[:k]]


