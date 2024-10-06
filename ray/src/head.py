from ray import serve
from .model import ModelDeployment
from typing import List

from .core.schema import NNsightRequestModel

models = {
    "Llama-3.1-8B" : "meta-llama/Llama-3.1-8B",
}

@serve.deployment(name="request")
class RayHead:
    def __init__(self, deployments):
        self.deployments = deployments

    async def __call__(self, request: NNsightRequestModel):
        return await self.deployments["Llama-3.1-8B"].remote(request)

deployments = {
    name : ModelDeployment.options(name=name).bind(model)
    for name, model in models.items()
}

head = RayHead.bind(deployments)