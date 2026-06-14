from typing import List, Tuple, Set
import networkx as nx
from networkx.algorithms.approximation import steiner_tree


def _table_of(name: str) -> str:
    """`table.column` -> `table` (handles column names that contain dots defensively)."""
    return name.rsplit(".", 1)[0]


def table_join_keys(G_u: nx.Graph, selected_tables: Set[str]) -> Set[str]:
    """Cross-table FK join-key columns needed to connect the *distinct* selected tables.
    """
    if len(selected_tables) < 2:
        return set()
    # one representative cross-table FK edge per table pair
    cross = {}
    for u, w in G_u.edges():
        tu, tw = _table_of(u), _table_of(w)
        if tu != tw:
            cross.setdefault(frozenset((tu, tw)), (u, w))
    GT = nx.Graph()
    for pair, (cu, cw) in cross.items():
        a, b = tuple(pair)
        GT.add_edge(a, b, cols=(cu, cw))
    add: Set[str] = set()
    for comp in nx.connected_components(GT):
        comp_tables = selected_tables & comp
        if len(comp_tables) >= 2:
            try:
                st = steiner_tree(GT.subgraph(comp).copy(), list(comp_tables))
                for a, b in st.edges():
                    cu, cw = GT[a][b]["cols"]
                    add.add(cu)
                    add.add(cw)
            except Exception:
                pass
    return add


def get_steiner_subgraph(G: nx.Graph, terminals: List[str]) -> nx.Graph:
    """Steiner closure that guarantees joinability of a selected column set.
    """
    G_u = G.to_undirected()
    terms = [t for t in terminals if t in G_u]
    forest = nx.Graph()
    for n in terms:
        forest.add_node(n, **G.nodes[n])
    selected_tables = {_table_of(n) for n in terms}
    for c in table_join_keys(G_u, selected_tables):
        if c in G_u and c not in forest:
            forest.add_node(c, **(G.nodes[c] if c in G.nodes else {}))
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
