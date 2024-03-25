from .gpt2 import gpt2_attr_map, GPT2Patcher

model_alterations = {
    "openai-community/gpt2" : (gpt2_attr_map, GPT2Patcher)
}

def get_attr_map(model_name):
    if model_name in model_alterations:
        return model_alterations[model_name][0]
    else:
        raise ValueError(f"Attribute mappings for {model_name} not supported.")
    
def get_alteration(model_name):
    if model_name in model_alterations:
        return model_alterations[model_name][1]
    else:
        raise ValueError(f"Model alterations for {model_name} not supported.")

class Alteration:
    def __init__(self, model_name):
        self.model_name = model_name
        self.patcher = get_alteration(model_name)
        
    def __call__(self):
        self.patcher.__enter__()

    def exit(self):
        self.patcher.__exit__()