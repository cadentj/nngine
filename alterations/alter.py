from ..envoy import Envoy

from dataclasses import dataclass

from nnsight.envoy import Envoy as NNsightEnvoy
from .FnEdit import FnEdit
from .Editor import Editor

from .gpt2 import gpt2

model_alterations = {
    "GPT2LMHeadModel" : gpt2
}

@dataclass
class Altr:
    """Altr is a dataclass that holds the necessary information to alter a model.

    Args:
        base (str): The base is where the alteration takes place. All default interventions
            can reference modules relative to the base.
        target (str): The target of the alteration. The function envoy is appended as an 
            attribute of the target.
        name (str): The name of the alteration, referenced in the .trace context.
        fn_hook (callable): A hook that that creates a FnEnvoy which is added later.
    """

    base: str 
    target: str
    name: str
    fn_hook: callable

def alter(model):

    model_name = model._model.__class__.__name__
    name_alterations, fn_alterations, hidden = model_alterations[model_name]()

    model._envoy = Envoy(model._model, attr_map=name_alterations, hidden=hidden)

    fn_edits = [
        FnEdit(
            name, 
            base,
            target,
            fn_hook
        )
        for base, target, name, fn_hook in fn_alterations
    ]

    editor = Editor(model._envoy, fn_edits)

    model._editor = editor
    model._editor.__enter__()

def restore(model):
    model._editor.__exit__(None, None, None)

    model._envoy = NNsightEnvoy(model)