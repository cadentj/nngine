# %%

import json
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._pytree import tree_map_only
from nnsight import NNsight
from functools import partial

COLLECTION = (list, tuple, set, dict, frozenset)

def get_shapes(repo_id):

    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)

    def _str(x):
        if isinstance(x, torch.Tensor):
            return str(tuple(x.shape))
        return "None"

    def _shape(x):
        is_shape = False
        rs = tree_map_only(
            torch.Tensor,
            _str,
            x
        )

        if isinstance(rs, COLLECTION) and len(rs) > 0:
            rs = rs[0]
            is_shape = True

        return rs, is_shape

    with fake_mode:

        config = AutoConfig.from_pretrained(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_config(config=config)
        model.eval()

        shapes = {}

        def add_input_shape(_, input, name):
            shapes[name]['input'], shapes[name]['input_collection'] = _shape(input)

        def add_output_shape(_, input, output, name):
            shapes[name]['output'], shapes[name]['output_collection'] = _shape(output)

        for name, module in model.named_modules():

            if "drop" in name or "generator" in name:
                continue

            shapes[name] = {
                'input': None,
                'output': None,
                'input_collection' : False,
                'output_collection' : False
            }

            module.register_forward_pre_hook(
                partial(add_input_shape, name=name),
                prepend=True
            )

            module.register_forward_hook(
                partial(add_output_shape, name=name),
                prepend=True
            )


        input = tokenizer(" ", return_tensors='pt')
        output = model(**input)

        shapes[repo_id] = shapes.get('', {})
        shapes[repo_id]['input'] = str(tuple(input['input_ids'].shape))
        shapes[repo_id]['output'] = str(tuple(output.logits.shape))

        return model, shapes

def get_type(module):
    print(module)
    return "ModuleList" if isinstance(module._module, torch.nn.ModuleList) else "Module"

def generate_pytree(module, shapes, atomic='', path='', fold=False):
    module_type = get_type(module)

    if module_type == "ModuleList":
        fold = True

    pytree = {
        "name": path,
        "atomic" : atomic,
        "type": module_type,
        "input" : shapes[module.path[1:]]['input'],
        "output" : shapes[module.path[1:]]['output'],
        "input_collection" : shapes[module.path[1:]]['input_collection'],
        "output_collection" : shapes[module.path[1:]]['output_collection'],
        "submodules": []
    }

    for submodule in module._sub_envoys:
        name = submodule.path.split('.')[-1]

        if "drop" in name or "generator" in name:
            continue

        sub_path = f"{path}.{name}" if fold else name

        pytree["submodules"].append(
            generate_pytree(submodule, shapes, atomic=submodule.path, path=sub_path)
        )

    if not pytree["submodules"]:
        pytree.pop("submodules")

    return pytree
        

def load_pytree(repo_id: str):

    model, shapes = get_shapes(repo_id)

    model = NNsight(model)

    graph = generate_pytree(model, shapes, atomic='model', path='model')

    return graph


