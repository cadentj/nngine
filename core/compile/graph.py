from typing import List, Union, Dict

from pydantic import BaseModel, model_validator

from .nodes import (
    Node,
    SessionNode,
    InputNode,
    ModuleNode,
    RunNode,
    FunctionNode,
    BatchNode,
    LoopNode,
    ListNode,
    GraphNode,
    ChatNode
)

class Edge(BaseModel):
    source: str
    target: str

class Graph(BaseModel):
    nodes: List[
        Union[
            GraphNode,
            InputNode,
            ModuleNode,
            RunNode,
            FunctionNode,
            BatchNode,
            LoopNode,
            ListNode,
            ChatNode
        ]
    ]
    edges: List[Edge]

    lookup: Dict[str, Node] = {}

    @model_validator(mode="after")
    def add_session(self):
        self.nodes.append(SessionNode(id="session"))
        return self

    @model_validator(mode="after")
    def add_lookup(self):
        self.lookup = {node.id: node for node in self.nodes}
        return self
    
    def get_nodes(self, type: List[str]):
        """Return nodes of a specific type."""
        return [node for node in self.nodes if node.type in types]