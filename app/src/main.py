from fastapi import FastAPI
from ray import serve

from .core.schema import NNsightRequestModel

import uvicorn

app = FastAPI()

@app.post("/remote")
async def ray_request(request: NNsightRequestModel):
    return await serve.get_app_handle("ndif").remote(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)  # Change the port number here
