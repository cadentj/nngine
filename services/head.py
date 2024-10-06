from ray import serve
from model import app as model_app

@serve.deployment(name="request")
class RayHead:
    def __init__(self):
        self.model = model_app()

    def respawn(self):
        pass

    def __call__(self):
        return "Hello World!"
    
head = RayHead.bind()