import einops
from ..FnEnvoy import FnEnvoy

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

def qkv_hook(envoy, head_type = 0):
    # where qkv is 0, 1, 2
    split = lambda x: einops.rearrange(x.output, "batch seq (d qkv) -> qkv batch seq d", qkv=3, d=768)[head_type,...]

    def revert(x):
        split = einops.rearrange(envoy.output, "batch seq (d qkv) -> qkv batch seq d", qkv=3, d=768)
        split[head_type,...] = x
        return einops.rearrange(split, "qkv batch seq d -> batch seq (d qkv)", qkv=3, d=768)

    hook = FnEnvoy(
        envoy,
        split,
        inverse=revert,
    )

    return hook

q_alter = [
    (
        f".transformer.h.{layer_idx}.attn", 
        f".transformer.h.{layer_idx}.attn.c_attn",  
        "q",
        lambda x: qkv_hook(x, 0),
    )
    for layer_idx in range(12)
]

k_alter = [
    (
        f".transformer.h.{layer_idx}.attn", 
        f".transformer.h.{layer_idx}.attn.c_attn",  
        "k",
        lambda x: qkv_hook(x, 1),
    )
    for layer_idx in range(12)
] 

v_alter = [
    (
        f".transformer.h.{layer_idx}.attn", 
        f".transformer.h.{layer_idx}.attn.c_attn",  
        "v",
        lambda x: qkv_hook(x, 2),
    )
    for layer_idx in range(12)
] 

def head_hook(envoy):
    split = lambda x: einops.rearrange(x.input[0][0], "batch seq (n_heads head_dim) -> batch seq n_heads head_dim", n_heads=12, head_dim=64)
    revert = lambda x: einops.rearrange(x, "batch seq n_heads head_dim -> batch seq (n_heads head_dim)", n_heads=12, head_dim=64)

    hook = FnEnvoy(
        envoy,
        split,
        inverse=revert,
    )

    return hook

head_alterations = [
    (
        f".transformer.h.{layer_idx}.attn", 
        f".transformer.h.{layer_idx}.attn.c_proj",  
        "heads",
        head_hook,
    )
    for layer_idx in range(12)
]


def attn_result_hook(envoy):

    def attn_result(x):

        attn_heads_out = einops.rearrange(
            x.c_proj.input[0][0],
            "batch pos (head_idx head_dim) \
                -> batch pos head_idx head_dim",
            head_idx=12,
            head_dim=64
        )

        w_o_split = einops.rearrange(
            x.c_proj.weight,
            "(head_idx head_dim) d_model \
                -> head_idx head_dim d_model",
            head_idx=12,
            head_dim=64
        )

        attn_out = einops.einsum(
            attn_heads_out, w_o_split,
            "batch pos head_idx head_dim, head_idx head_dim d_model -> batch pos head_idx d_model",
        )

        return attn_out

    def revert(x):
        import torch
        torch.sum(x, dim=2)

    hook = FnEnvoy(
        envoy,
        attn_result,
        inverse=revert
    )

    return hook

attn_alterations = [
    (
        f".transformer.h.{layer_idx}.attn", 
        f".transformer.h.{layer_idx}.attn",  
        "attn_result",
        attn_result_hook,
    )
    for layer_idx in range(12)
]

fn_alterations = [
    *q_alter,
    *k_alter,
    *v_alter,
    *head_alterations,
    *attn_alterations
]

def gpt2():
    return (name_alterations, fn_alterations)