import einops
import torch

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

def qkv_hook(attn):

    split = lambda x: \
        einops.rearrange(
            x.output,
            "batch seq (d qkv) -> qkv batch seq d",
            qkv=3,
            d=768
        )

    revert = lambda x: \
        einops.rearrange(
            x,
            "qkv batch seq d -> batch seq (d qkv)",
            qkv=3,
            d=768
        )
    
    hook = FnEnvoy(
        attn.c_attn,
        split,
        inverse=revert,
    )

    return hook

qkv_alteration = [
    (
        f".transformer.h.{layer_idx}.attn",  
        "qkv",
        qkv_hook,
    )
    for layer_idx in range(12)
]

def indv_qkv_hook(attn, slice_index):
    
    # q : 0, k : 1, v : 2
    split = lambda x: attn.qkv[slice_index,...]

    def revert(x):
        split = einops.rearrange(
            attn.output, 
            "batch seq (d qkv) -> qkv batch seq d", 
            qkv=3, 
            d=768
        )

        split[slice_index, ...] = x
        
        return einops.rearrange(
            split, 
            "qkv batch seq d -> batch seq (d qkv)", 
            qkv=3, 
            d=768
        )

    hook = FnEnvoy(
        attn.c_attn,
        split,
        inverse=revert,
    )

    return hook

q_alter = [
    (
        f".transformer.h.{layer_idx}.attn",  
        "q",
        lambda x: indv_qkv_hook(x, "q"),
    )
    for layer_idx in range(12)
]

k_alter = [
    (
        f".transformer.h.{layer_idx}.attn", 
        "k",
        lambda x: indv_qkv_hook(x, "k"),
    )
    for layer_idx in range(12)
] 

v_alter = [
    (
        f".transformer.h.{layer_idx}.attn",
        "v",
        lambda x: indv_qkv_hook(x, "v"),
    )
    for layer_idx in range(12)
] 


def split_qkv_hook(attn, head_type):
    
    slice_index = {"q":0, "k":1, "v":2}[head_type]

    def split_head(c_attn):
        resid_pre = attn.input[0][0]

        repeated_tensor = einops.repeat(
            resid_pre,
            "batch pos d_model -> batch pos head_idx d_model",
            head_idx=12,
        )

        split_weight = einops.rearrange(
            c_attn.weight,
            "d_model (qkv head_idx d_head) -> qkv head_idx d_head d_model",
            qkv=3,
            head_idx=12,
            d_head=64,
        )[slice_index,...]

        split_bias = einops.rearrange(
            c_attn.bias,
            "(qkv head_idx d_head) -> qkv head_idx d_head",
            qkv=3,
            head_idx=12,
            d_head=64,
        )[slice_index,...]
        
        split_out = einops.einsum(
            repeated_tensor, split_weight,
            "batch pos head_idx d_model, head_idx d_head d_model -> batch pos head_idx d_head",
        ) + split_bias

        return split_out

    def revert(x):
        torch.sum(x, dim=2)

    hook = FnEnvoy(
        attn.c_attn,
        split_head,
        inverse=revert,
    )

    return hook

q_split_alterations = [
    (
        f".transformer.h.{layer_idx}.attn",
        "split_q",
        lambda x: split_qkv_hook(x, "q"),
    )
    for layer_idx in range(12)
]

k_split_alterations = [
    (
        f".transformer.h.{layer_idx}.attn",
        "split_k",
        lambda x: split_qkv_hook(x, "k"),
    )
    for layer_idx in range(12)
]

v_split_alterations = [
    (
        f".transformer.h.{layer_idx}.attn",
        "split_v",
        lambda x: split_qkv_hook(x, "v"),
    )
    for layer_idx in range(12)
]

def head_hook(base):
    split = lambda x: einops.rearrange(x.input[0][0], "batch seq (head_idx head_dim) -> batch seq head_idx head_dim", head_idx=12, head_dim=64)
    revert = lambda x: einops.rearrange(x, "batch seq head_idx head_dim -> batch seq (head_idx head_dim)", head_idx=12, head_dim=64)

    hook = FnEnvoy(
        base.c_proj,
        split,
        inverse=revert,
    )

    return hook

head_alterations = [
    (
        f".transformer.h.{layer_idx}.attn", 
        "heads",
        head_hook,
    )
    for layer_idx in range(12)
]


def attn_result_hook(attn):

    def attn_result(attn):

        attn_heads_out = attn.heads.output

        w_o_split = einops.rearrange(
            attn.c_proj.weight,
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
        torch.sum(x, dim=2)

    hook = FnEnvoy(
        attn,
        attn_result,
        inverse=revert
    )

    return hook

attn_alterations = [
    (
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
    *attn_alterations,
    *q_split_alterations
]

def gpt2():
    return (name_alterations, fn_alterations)