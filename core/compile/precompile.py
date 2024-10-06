from typing import List, Dict

from .utils import build_edge_lookup, get_top_parent
from .schema import Node, Edge, Graph

def resolve_dependencies(graph: Graph):
    """Ensure contexts are processed correctly."""

    edges = []
    for edge in graph.edges:
        src = graph.lookup[edge.source]
        tar = graph.lookup[edge.target]

        match (len(src.data.parents), len(tar.data.parents)):
            # A connection to a node of greater depth should connect
            # to the top parent of the target node. This ensures that
            # nodes are topologically sorted correctly.

            case (x, y) if (
                x < y
                and not (
                    "input" in src.id
                    and "batch" in tar.id
                )
            ):
                other = get_top_parent(graph, tar.id, level=src.parent)
                edges.append(Edge(source=src.id, target=other))

            # A connection to a node of equal depth within a different
            # context should resolve contexts sequentially.
            case (x, y) if (x == y and src.parent != tar.parent):
                edges.append(Edge(source=src.id, target=tar.parent))

    graph.edges.extend(edges)


def unfold_edges(graph: Graph):
    def unfold_context(node: Node):
        if node.data.parents == [""]:
            return

        parent_id = node.data.parents[-1]
        edge = Edge(source=parent_id, target=node.id)
        graph.edges.append(edge)

    for node in graph.nodes:
        unfold_context(node)


def resolve_edges(graph: Graph, sorted_nodes: List[Node]):
    edge_lookup = build_edge_lookup(graph)

    for node in sorted_nodes:
        for edge in edge_lookup.get(node.id, []):
            src = graph.lookup[edge.source]
            tar = graph.lookup[edge.target]

            match (src.data.variant, tar.data.variant):
                case ("module", ("module" | "function")):
                    src.protocol = "getter"
                    tar.protocol = "setter"
                    
                case ("module", "list"):
                    src.protocol = "append"
                    tar.set_input_id(src)

                case ("function", "list"):
                    src.protocol = "append"
                    tar.set_input_id(src)

                case ("list", "list"):
                    tar.protocol = "append"
                    tar.set_input_id(src)

                case ("context", "list"):
                    src.add_default(tar)


def prepare(graph: Graph):
    resolve_dependencies(graph)

    # Unfold context -> child edges after resolving dependencies.
    unfold_edges(graph)

def precompile(
    graph, sorted_nodes: List[Node], r_adj_list: Dict[str, List[str]]
) -> List[str]:
    
    resolve_edges(graph, sorted_nodes)

    code = []

    for node in graph.nodes:
        inputs = [graph.lookup[input_id] for input_id in r_adj_list[node.id]]

        line: str | None = node.precompile(inputs)
        if line is not None:
            code.append(line)

    return code
