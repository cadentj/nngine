from .interventions import *
from dataclasses import dataclass
from typing import Callable

from ..FnEdit import FnEdit

name_alterations = {
    "wte" : "embed",
    "wpe" : "pos_embed",
    "drop" : "dropout",
    "h" : "layers" ,
    "ln_f" : "ln_final",
    "lm_head" : "unembed"
}

dimensions = {
    "n_heads" : 12,
    "n_layers" : 12,
    "d_model" : 768,
}

hidden = []

blocks = [
    f".transformer.h.{layer_idx}"
    for layer_idx in range(12)
]

attention = [
    f".transformer.h.{layer_idx}.attn"
    for layer_idx in range(12)
]

alterations = [
    ("qkv", attention, attention, qkv_hook),
    ("q", attention, attention, lambda x: indv_qkv_hook(x, 0)),
    ("k", attention, attention, lambda x: indv_qkv_hook(x, 1)),
    ("v", attention, attention, lambda x: indv_qkv_hook(x, 2)),
    ("split_q_input", blocks, blocks, block_input_hook),
    ("split_k_input", blocks, blocks, block_input_hook),
    ("split_v_input", blocks, blocks, block_input_hook),
    ("split_q", blocks, attention, lambda x: split_qkv_hook(x, 0)),
    ("split_k", blocks, attention, lambda x: split_qkv_hook(x, 1)),
    ("split_v", blocks, attention, lambda x: split_qkv_hook(x, 2)),
    ("heads", attention, attention, head_hook),
    ("attn_result", attention, attention, attn_result_hook),
]

alterations = [FnEdit(*alteration) for alteration in alterations]

def gpt2():
    return name_alterations, alterations, hidden, dimensions