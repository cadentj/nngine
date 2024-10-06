from collections import defaultdict, deque
from typing import List

from .schema import Graph
from .precompile import precompile, prepare
from .utils import get_adj_list


def get_in_degree(graph: Graph) -> dict:
    """Compute the in-degree of each node."""
    in_degree = {node.id: 0 for node in graph.nodes}

    for edge in graph.edges:
        in_degree[edge.target] += 1

    return in_degree


def topological_sort(graph: Graph) -> List[str]:
    """Returns a list of node_ids in topologically sorted order."""
    adj_list = get_adj_list(graph)
    in_degree = get_in_degree(graph)

    zero_degree = [node_id for node_id, degree in in_degree.items() if (degree == 0)]

    queue = deque(zero_degree)

    topological_order = []
    grouped = defaultdict(list)

    while queue:
        node_id = queue.popleft()
        topological_order.append(node_id)

        node = graph.lookup[node_id]
        grouped[node.parent].append(node)

        for neighbor in adj_list.get(node_id, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topological_order) == len(graph.nodes):
        return topological_order, grouped
    else:
        raise ValueError(
            "The graph has at least one cycle and cannot be topologically sorted."
        )


def compile(graph: Graph) -> tuple:
    """Compile a graph into executable NNsight code."""
    
    # NOTE: Should move this out of here.
    
    prepare(graph)
    sorted_ids, grouped = topological_sort(graph)
    sorted_nodes = [graph.lookup[node_id] for node_id in sorted_ids]

    code = precompile(graph, sorted_nodes, get_adj_list(graph, reverse=True))

    visited = set()

    def expand(node):
        """Expand the node and its children into code."""
        if node.id in visited:
            return

        visited.add(node.id)

        # Only recurse into context nodes
        if node.data.variant == "context":
            code.append(node.compile())

            for child in grouped[node.id]:
                expand(child)

        else:
            code.append(node.compile())

    # Expand the nodes in topological order
    for node in sorted_nodes:
        expand(node)

    return "\n".join(code)
