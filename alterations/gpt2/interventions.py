import einops
import torch

from ...FnEnvoy import FnEnvoy

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

def block_input_hook(transformer_block):
    def block_input(block):
        block_in = block.input[0][0].clone()

        return einops.repeat(
            block_in,
            "batch pos d_model -> batch pos head_idx d_model",
            head_idx=12,
        )

    def revert(base, x):
        print("shouldn't be called")

    hook = FnEnvoy(
        transformer_block,
        block_input,
        inverse=revert,
        replace=False
    )

    return hook


def split_qkv_hook(block, slice_index):
    attention = block.attn

    slice_map = {0: block.split_q_input, 1: block.split_k_input, 2: block.split_v_input}
    repeated_attn = slice_map[slice_index]

    def split_head(block_input):
        nonlocal block

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

        split_attn =  list(block.attn.c_attn.output.split(768, dim=2))

        # split_attn = list(attention.qkv.output)

        split_attn[slice_index] = einops.rearrange(
            x, 
            "batch pos head_idx d_head -> batch pos (head_idx d_head)",
        )
        block.attn.c_attn.output = torch.cat(split_attn, dim=2)

    # TODO: Remove replace keyword? 
    # Maybe add some input function? 
    hook = FnEnvoy(
        repeated_attn,
        split_head,
        inverse=revert,
    )

    return hook

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
        base = torch.sum(x, dim=2) + base.bias

    hook = FnEnvoy(
        attention.c_proj,
        attn_result,
        inverse=revert,
        replace=False
    )

    return hook

def mlp_in_hook(block):
    
    def mlp_in(b):
        resid_pre = b.input[0][0]
        attn_out = b.attn.output[0]        

        resid_mid = resid_pre + attn_out

        return resid_mid

    def revert(base, x):
        base.mlp.input[0][0][:] = block.ln_2(x)

    hook = FnEnvoy(
        block,
        mlp_in,
        inverse=revert,
    )

    return hook