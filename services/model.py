from ray import serve

@serve.deployment(name="model")
class ModelDeployment:
    def __init__(self):
        self.a = "b"

    def respawn(self):
        pass

    def __call__(self):
        return self.a
    
def app():
    return ModelDeployment.bind()