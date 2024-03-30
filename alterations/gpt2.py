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

hidden = [
    "c_attn",
    "c_proj",
    "qkv",
]

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
    split = lambda _: attn.qkv.output[slice_index]

    def revert(x):
        split = einops.rearrange(
            attn.output, 
            "batch seq (d qkv) -> qkv batch seq d", 
            qkv=3, 
            d=768
        )

        split[slice_index] = x
        
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
        lambda x: indv_qkv_hook(x, 0),
    )
    for layer_idx in range(12)
]

k_alter = [
    (
        f".transformer.h.{layer_idx}.attn", 
        "k",
        lambda x: indv_qkv_hook(x, 1),
    )
    for layer_idx in range(12)
] 

v_alter = [
    (
        f".transformer.h.{layer_idx}.attn",
        "v",
        lambda x: indv_qkv_hook(x, 2),
    )
    for layer_idx in range(12)
] 

def split_attn_input(attention):

    def add_head_dim(attn):
        resid_pre = attn.input[0][0]

        # `einops.repeat` uses a view in torch, so we generally clone the tensor to avoid using shared storage for each head entry
        return einops.repeat(
            resid_pre,
            "batch pos d_model -> batch pos head_idx d_model",
            head_idx=12,
        ).clone()

    def revert(x):
        torch.sum(x, dim=2)

    hook = FnEnvoy(
        attention,
        add_head_dim,
        inverse=revert,
        io = "input"
    )

    return hook

attn_in_alterations = [
    (
        f".transformer.h.{layer_idx}.attn",
        "attn_input",
        split_attn_input,
    )
    for layer_idx in range(12)
]


def split_qkv_hook(attn, slice_index):

    def split_head(attn_input):

        split_weight = einops.rearrange(
            attn.c_attn.weight,
            "d_model (qkv head_idx d_head) -> qkv head_idx d_head d_model",
            qkv=3,
            head_idx=12,
            d_head=64,
        )[slice_index]

        split_bias = einops.rearrange(
            attn.c_attn.bias,
            "(qkv head_idx d_head) -> qkv head_idx d_head",
            qkv=3,
            head_idx=12,
            d_head=64,
        )[slice_index]
        
        split_out = einops.einsum(
            attn_input, split_weight,
            "batch pos head_idx d_model, head_idx d_head d_model -> batch pos head_idx d_head",
        ) + split_bias

        return split_out

    def revert(x):
        torch.sum(x, dim=2)

    hook = FnEnvoy(
        attn.attn_input,
        split_head,
        inverse=revert,
    )

    return hook

q_split_alterations = [
    (
        f".transformer.h.{layer_idx}.attn",
        "split_q",
        lambda x: split_qkv_hook(x, 0),
    )
    for layer_idx in range(12)
]

k_split_alterations = [
    (
        f".transformer.h.{layer_idx}.attn",
        "split_k",
        lambda x: split_qkv_hook(x, 1),
    )
    for layer_idx in range(12)
]

v_split_alterations = [
    (
        f".transformer.h.{layer_idx}.attn",
        "split_v",
        lambda x: split_qkv_hook(x, 2),
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


def attn_result_hook(attention):

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
        attention,
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
    *attn_in_alterations,
    *qkv_alteration,
    *q_split_alterations,
    *k_split_alterations,
    *v_split_alterations,
]

def gpt2():
    return (name_alterations, fn_alterations, hidden)