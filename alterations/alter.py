from ..Envoy import Envoy


from nnsight.envoy import Envoy as NNsightEnvoy
from .FnEdit import FnEdit
from .Editor import Editor

from .gpt2 import gpt2

model_alterations = {
    "GPT2LMHeadModel" : gpt2
}

def alter(model):

    # Load alterations
    model_name = model._model.__class__.__name__
    name_alterations, fn_alterations, hidden, dimensions = model_alterations[model_name]()

    # Update config
    update_config(model, dimensions)

    # Wrap module in new Envoy class
    model._envoy = Envoy(model._model, attr_map=name_alterations, hidden=hidden)

    # Create Editor object
    editor = Editor(model._envoy, fn_alterations)

    # Clear existing _editor if it exists
    if hasattr(model, "_editor"):
        model._editor.__exit__(None, None, None)
    
    model._editor = editor
    model._editor.__enter__()

def update_config(model, updates):

    for key, value in updates.items():
        setattr(model.config, key, value)

def restore(model):
    model._editor.__exit__(None, None, None)

    model._envoy = NNsightEnvoy(model)