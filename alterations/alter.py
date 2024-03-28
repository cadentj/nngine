from ..envoy import Envoy

from nnsight.envoy import Envoy as NNsightEnvoy
from .FnEdit import FnEdit
from .Editor import Editor

from .gpt2 import gpt2

model_alterations = {
    "GPT2LMHeadModel" : gpt2
}

def alter(model):

    model_name = model._model.__class__.__name__
    name_alterations, fn_alterations = model_alterations[model_name]()

    model._envoy = Envoy(model._model, attr_map=name_alterations)

    fn_edits = [
        FnEdit(
            name, 
            base,
            fn_hook
        )
        for base, name, fn_hook in fn_alterations
    ]

    editor = Editor(model._envoy, fn_edits)

    model._editor = editor
    model._editor.__enter__()

def restore(model):
    model._editor.__exit__(None, None, None)

    model._envoy = NNsightEnvoy(model)