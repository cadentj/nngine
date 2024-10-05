from pydantic import BaseModel

class Edge(BaseModel):
    source: str
    target: str


