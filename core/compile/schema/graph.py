from typing import List, Union, Dict

from pydantic import BaseModel, model_validator

from .edges import Edge
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