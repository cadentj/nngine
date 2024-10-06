from fastapi import FastAPI
from ray import serve

from .core.schema import NNsightRequestModel, ModelConfigModel
from .core.model import load

import uvicorn

app = FastAPI()

@app.post("/remote")
async def ray_request(request: NNsightRequestModel):
    return await serve.get_app_handle("nngine").remote(request)

@app.post("/load-model")
async def load_model(request: ModelConfigModel):
    return {
        "pytree": load(request.repo_id)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)  # Change the port number here
