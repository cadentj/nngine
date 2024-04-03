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
    # "c_attn",
    # "c_proj",
    # "qkv",
]

### ALTERATIONS ###

def qkv_hook(attn):

    def split(c_attn):
        return c_attn.output.split(768, dim=2)
    
    def revert(base, x):
        base.output = torch.cat(x, dim=2)

    hook = FnEnvoy(
        attn.c_attn,
        split,
        inverse=revert,
    )

    return hook

qkv_alteration = [
    (
        f".transformer.h.{layer_idx}.attn",  
        f".transformer.h.{layer_idx}.attn",  
        "qkv",
        qkv_hook,
    )
    for layer_idx in range(12)
]

def indv_qkv_hook(attn, slice_index):
    
    split = lambda qkv: qkv.output[slice_index]

    def revert(base, x):
        split = base.output
        split[slice_index] = x
        
        # NOTE: We don't .cat here because
        # .qkv does so already
        base.output = split

    hook = FnEnvoy(
        attn.qkv,
        split,
        inverse=revert,
    )

    return hook

q_alter = [
    (
        f".transformer.h.{layer_idx}.attn",  
        f".transformer.h.{layer_idx}.attn",  
        "q",
        lambda x: indv_qkv_hook(x, 0),
    )
    for layer_idx in range(12)
]

k_alter = [
    (
        f".transformer.h.{layer_idx}.attn", 
        f".transformer.h.{layer_idx}.attn", 
        "k",
        lambda x: indv_qkv_hook(x, 1),
    )
    for layer_idx in range(12)
] 

v_alter = [
    (
        f".transformer.h.{layer_idx}.attn",
        f".transformer.h.{layer_idx}.attn", 
        "v",
        lambda x: indv_qkv_hook(x, 2),
    )
    for layer_idx in range(12)
] 

def block_input_hook(transformer_block):
    def block_input(block):
        block_in = block.input[0][0]

        return einops.repeat(
            block_in,
            "batch pos d_model -> batch pos head_idx d_model",
            head_idx=12,
        ).clone()

    def revert(base, x):
        base.input[0][0] = x.sum(dim=-2)

    hook = FnEnvoy(
        transformer_block,
        block_input,
        inverse=revert,
        replace=False
    )

    return hook

split_q_input = [
    (
        f".transformer.h.{layer_idx}",
        f".transformer.h.{layer_idx}", 
        "split_q_input",
        lambda x: block_input_hook(x),
    )
    for layer_idx in range(12)
] 

split_k_input = [
    (
        f".transformer.h.{layer_idx}",
        f".transformer.h.{layer_idx}", 
        "split_k_input",
        lambda x: block_input_hook(x),
    )
    for layer_idx in range(12)
] 

split_v_input = [
    (
        f".transformer.h.{layer_idx}",
        f".transformer.h.{layer_idx}", 
        "split_v_input",
        lambda x: block_input_hook(x),
    )
    for layer_idx in range(12)
] 

def split_qkv_hook(block, slice_index):
    attention = block.attn

    slice_map = {0: block.split_q_input, 1: block.split_k_input, 2: block.split_v_input}
    repeated_attn = slice_map[slice_index]

    def split_head(block_input):

        attn_input = block.ln_1(block_input.output)

        weight = torch.tensor_split(attention.c_attn.weight, 3, dim=1)[slice_index]
        split_weight = einops.rearrange(
            weight, 
            "d_model (head_index d_head) -> head_index d_model d_head",
            head_index=12
        )

        split_bias = einops.rearrange(
            attention.c_attn.bias,
            "(qkv head_idx d_head) -> qkv head_idx d_head",
            qkv=3,
            head_idx=12,
            d_head=64,
        )[slice_index]
        
        split_out = einops.einsum(
            attn_input, split_weight,
            "batch pos head_idx d_model, head_idx d_model d_head -> batch pos head_idx d_head",
        ) + split_bias
        
        return split_out

    def revert(base, x):
        split_attn = list(attention.qkv.output)

        split_attn[slice_index] = einops.rearrange(
            x, 
            "batch pos head_idx d_head -> batch pos (head_idx d_head)",
        )
        attention.qkv.output = split_attn

    # TODO: Remove replace keyword? 
    # Maybe add some input function? 
    hook = FnEnvoy(
        repeated_attn,
        split_head,
        inverse=revert,
    )

    return hook

q_split_alterations = [
    (
        f".transformer.h.{layer_idx}",
        f".transformer.h.{layer_idx}.attn",
        "split_q",
        lambda x: split_qkv_hook(x, 0),
    )
    for layer_idx in range(12)
]

k_split_alterations = [
    (
        f".transformer.h.{layer_idx}",
        f".transformer.h.{layer_idx}.attn",
        "split_k",
        lambda x: split_qkv_hook(x, 1),
    )
    for layer_idx in range(12)
]

v_split_alterations = [
    (
        f".transformer.h.{layer_idx}",
        f".transformer.h.{layer_idx}.attn",
        "split_v",
        lambda x: split_qkv_hook(x, 2),
    )
    for layer_idx in range(12)
]

def head_hook(base):
    split = lambda c_proj: einops.rearrange(c_proj.input[0][0], "batch seq (head_idx head_dim) -> batch seq head_idx head_dim", head_idx=12, head_dim=64)

    def revert(base, c_proj):
        base.output = einops.rearrange(c_proj, "batch seq head_idx head_dim -> batch seq (head_idx head_dim)", head_idx=12, head_dim=64)

    hook = FnEnvoy(
        base.c_proj,
        split,
        inverse=revert,
    )

    return hook

head_alterations = [
    (
        f".transformer.h.{layer_idx}.attn", 
        f".transformer.h.{layer_idx}.attn", 
        "heads",
        head_hook,
    )
    for layer_idx in range(12)
]


def attn_result_hook(attention):

    def attn_result(c_proj):
        heads = einops.rearrange(
            c_proj.input[0][0], 
            "batch seq (head_idx head_dim) -> batch seq head_idx head_dim", 
            head_idx=12, 
            head_dim=64
        )

        w_o_split = einops.rearrange(
            c_proj.weight,
            "(head_idx head_dim) d_model \
                -> head_idx head_dim d_model",
            head_idx=12,
            head_dim=64
        )

        attn_out = einops.einsum(
            heads, w_o_split,
            "batch pos head_idx head_dim, head_idx head_dim d_model -> batch pos head_idx d_model",
        )

        return attn_out

    def revert(base, x):
        base.output = torch.sum(x, dim=2)

    hook = FnEnvoy(
        attention.c_proj,
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
    *qkv_alteration,
    *q_alter,
    *k_alter,
    *v_alter,
    # *head_alterations,
    *attn_alterations,
    # *attn_in_alterations,
    *split_q_input,
    *split_k_input,
    *split_v_input,
    *q_split_alterations,
    *k_split_alterations,
    *v_split_alterations,
]

def gpt2():
    return (name_alterations, fn_alterations, hidden)