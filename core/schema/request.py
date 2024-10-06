from typing import Literal

from pydantic import BaseModel

from ..compile import Graph

class NNsightRequestModel(BaseModel):
    op: Literal["code", "run", "chat"]
    graph: Graph

class ModelConfigModel(BaseModel):
    repo_id: str