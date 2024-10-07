from fastapi import FastAPI
from ray import serve

from .model import ModelDeployment
from .core.schema import NNsightRequestModel

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class FastAPIDeployment:
    def __init__(self, model):
        self.model = model

    # FastAPI will automatically parse the HTTP request for us.
    @app.get("/hello")
    def say_hello(self, name: str) -> str:
        return f"Hello {name}!"
    
    @app.post("/code")
    async def remote(self, request: NNsightRequestModel):
        return await self.model.remote(request)


model = ModelDeployment.bind("Qwen/Qwen2.5-0.5B-Instruct")
head = FastAPIDeployment.bind(model)