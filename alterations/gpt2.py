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

def qkv_hook(envoy):
    split = lambda x: einops.rearrange(x, "batch seq (d qkv) -> qkv batch seq d", qkv=3, d=768)
    revert = lambda x: einops.rearrange(x, "qkv batch seq d -> batch seq (d qkv)", qkv=3, d=768)

    hook = FnEnvoy(
        envoy,
        split,
        revert,
    )

    return hook

qkv_alterations = [
    (
        f".transformer.h.{layer_idx}.attn", 
        f".transformer.h.{layer_idx}.attn.c_attn",  
        "qkv",
        qkv_hook
    )
    for layer_idx in range(12)
]

def head_hook(envoy):
    split = lambda x: einops.rearrange(x[0][0], "batch seq (n_heads head_dim) -> batch seq n_heads head_dim", n_heads=12, head_dim=64)
    revert = lambda x: einops.rearrange(x, "batch seq n_heads head_dim -> batch seq (n_heads head_dim)", n_heads=12, head_dim=64)

    hook = FnEnvoy(
        envoy,
        split,
        revert,
        io = "input"
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

fn_alterations = [
    *qkv_alterations,
    *head_alterations
]

def gpt2():
    return (name_alterations, fn_alterations)