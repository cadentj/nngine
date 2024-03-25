from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from transformers.models import gpt2

from nnsight import util
from nnsight.patching import Patch, Patcher

import einops
from ..envoy import FnEnvoy


### ATTRIBUTE MAP ###

name_alterations = {
    "wte" : "embed",
    "wpe" : "pos_embed",
    "drop" : "dropout",
    "h" : "layers" ,
    "ln_f" : "ln_final",
    "lm_head" : "unembed"
}

### ALTERATIONS ###

def head_hook(base, target):
    rearrange = lambda x : einops.rearrange(x, "batch seq (heads head_dim) -> batch heads seq head_dim", heads=12)
    revert = lambda x : einops.rearrange(x, "batch heads seq head_dim -> batch seq (heads head_dim)")

    fn_hook = FnEnvoy(base, target, rearrange, revert)

    return fn_hook

fn_alterations = [
    (
        f".transformer.h.{base}.attn", 
        f".transformer.h.{base}.attn.c_proj", 
        f".transformer.h.{target}.attn.c_proj", 
        "heads",
        head_hook
    )
    for base, target in [
        (i, i + 1) for i in range(11)
    ]
] + [
    (
        f".transformer.h.11.attn", 
        f".transformer.h.11.attn.c_proj", 
        f".transformer.ln_f", 
        "heads",
        head_hook
    )
]

def gpt2():
    return (name_alterations, fn_alterations)