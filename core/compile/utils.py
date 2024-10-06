from collections import defaultdict
from .schema import Graph

def build_edge_lookup(graph: Graph):
    edge_lookup = defaultdict(list)

    for edge in graph.edges:
        edge_lookup[edge.source].append(edge)

    return edge_lookup


def get_top_parent(graph: Graph, node_id: str, level: str = "session") -> str:
    """Get the parent id of a node at a certain depth."""
    node = graph.lookup[node_id]

    while node.parent != level:
        node = graph.lookup[node.parent]

    return node.id

def get_adj_list(graph: Graph, reverse: bool = False) -> dict:
    adj_list = defaultdict(list)

    for edge in graph.edges:
        if reverse:
            adj_list[edge.target].append(edge.source)
        else:
            adj_list[edge.source].append(edge.target)

    return adj_list
