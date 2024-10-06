from fastapi import FastAPI
from ray import serve

import uvicorn

app = FastAPI()

@app.get("/request")
async def request():
    return await serve.get_app_handle("default").remote()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)  # Change the port number here
