"""NNsight implementation of Edge Attribution Patching (arXiv:2310.10348)
This implementation is based on Can and Aaquib's version in TransformerLens
Link to repo: https://github.com/canrager/clas/blob/main/tl_utils.py
"""

import torch as t

import einops
from torch import Tensor 
from jaxtyping import Float, Int
from typing import Dict, Callable, List, Union
import numpy as np

from nnsight import LanguageModel
from nngine import alter

class EAP:

    def __init__(
        self,
        config,
        components: List[str] = ["head", "mlp"],
        device = "cpu",
        dtype=t.float32
    ):

        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.d_model = config.d_model
        self.device = device
        self.dtype = dtype

        # q, k, and v up-projections
        self.num_projections = 3

        if components == ["head", "mlp"]:
            self.upstream_hook_types = ["attn.hook_result", "hook_mlp_out"]
            self.downstream_hook_types = ["hook_q_input", "hook_k_input", "hook_v_input", "hook_mlp_in"]

            self.n_upstream_nodes = self.n_layers * (self.n_heads + 1)
            self.n_downstream_nodes = self.n_layers * self.n_heads * self.num_projections + self.n_layers

        elif components == ["head"]:
            self.upstream_hook_types = ["attn.hook_result"]
            self.downstream_hook_types = ["hook_q_input", "hook_k_input", "hook_v_input"]

            self.n_upstream_nodes = self.n_layers * self.n_heads
            self.n_downstream_nodes = self.n_layers * self.n_heads * self.num_projections

        self.create_slices()

    def create_slices(self):
        # Reset or initialize hook slices and node slices dictionaries.
        self.upstream_hook_slices = {}
        self.downstream_hook_slices = {}
        self.upstream_nodes_before_layer = {}
        self.upstream_nodes_before_mlp_layer = {}
        self.upstream_node_names = []
        self.downstream_node_names = []

        for layer in range(self.n_layers):
            # Compute base slices for attention and MLP for the current layer.
            attn_base = layer * (self.n_heads + 1)
            mlp_base = attn_base + self.n_heads

            self.upstream_nodes_before_layer[layer] = slice(0, attn_base)
            self.upstream_nodes_before_mlp_layer[layer] = slice(0, mlp_base)

            # Upstream slices
            for hook_type in self.upstream_hook_types:
                hook_key = f"blocks.{layer}.{hook_type}"
                if hook_type == "attn.hook_result":
                    self.upstream_hook_slices[hook_key] = slice(attn_base, attn_base + self.n_heads)
                elif hook_type == "hook_mlp_out":
                    self.upstream_hook_slices[hook_key] = slice(mlp_base, mlp_base + 1)

            for head in range(self.n_heads):
                self.upstream_node_names.append(f"head.{layer}.{head}")

            self.upstream_node_names.append(f"mlp.{layer}")

            # Downstream slices
            for hook_type in self.downstream_hook_types:
                layer_base = layer * self.n_heads * self.num_projections + layer
                hook_key = f"blocks.{layer}.{hook_type}"

                if hook_type == "hook_q_input":
                    self.downstream_hook_slices[hook_key] = slice(layer_base, layer_base + self.n_heads)
                elif hook_type == "hook_k_input":
                    self.downstream_hook_slices[hook_key] = slice(layer_base + self.n_heads, layer_base + 2 * self.n_heads)
                elif hook_type == "hook_v_input":
                    self.downstream_hook_slices[hook_key] = slice(layer_base + 2 * self.n_heads, layer_base + 3 * self.n_heads)
                elif hook_type == "hook_mlp_in":
                    self.downstream_hook_slices[hook_key] = slice(layer_base + 3 * self.n_heads, layer_base + 3 * self.n_heads + 1)

                if hook_type != "hook_mlp_in":
                    for head in range(self.n_heads):
                        self.downstream_node_names.append(f"head.{layer}.{head}.{hook_type.split('_')[1]}")

            self.downstream_node_names.append(f"mlp.{layer}")

    def reset_scores(self):
        self.eap_scores = t.zeros(
            (self.n_upstream_nodes, self.n_downstream_nodes),
            device=self.device
        )

    def run(
        self,
        model,
        clean_tokens: Int[Tensor, "batch_size seq_len"],
        corrupted_tokens: Int[Tensor, "batch_size seq_len"],
        # TODO: Implement batch_size
        batch_size: Int,
        metric: Callable,
    ):
        assert clean_tokens.shape == corrupted_tokens.shape

        self.reset_scores()

        seq_len = clean_tokens.shape[1]
        upstream_activations_difference = t.zeros(
            (batch_size, seq_len, self.n_upstream_nodes, self.d_model),
            device=self.device,
            dtype=self.dtype,
            requires_grad=False
        )
        
        corrupted_out = {}
        with t.no_grad():
            with model.trace(corrupted_tokens):
                for i, layer in enumerate(model.transformer.layers):
                    corrupted_out[f"blocks.{i}.attn.hook_result"] = layer.attn.attn_result.output.save()

                    # # TODO: Fix to just look for upstream slices.
                    if "hook_mlp_out" in self.upstream_hook_types:
                        corrupted_out[f"blocks.{i}.hook_mlp_out"] = layer.mlp.output.save()

        for component, activations in corrupted_out.items():
            if "mlp" in component:
                activations = activations.value.unsqueeze(-2)
            else:
                activations = activations.value

            upstream_activations_difference[:, :, self.upstream_hook_slices[component], :] = -activations

        del corrupted_out
        t.cuda.empty_cache()


        clean_out = {}
        gradients = {}


        with model.trace(clean_tokens):

            for i, layer in enumerate(model.transformer.layers):

                gradients[f"blocks.{i}.hook_q_input"] = layer.attn.split_q.input.grad.save()
                gradients[f"blocks.{i}.hook_k_input"] = layer.attn.split_k.input.grad.save()
                gradients[f"blocks.{i}.hook_v_input"] = layer.attn.split_v.input.grad.save()

                clean_out[f"blocks.{i}.attn.hook_result"] = layer.attn.attn_result.output.save()

                if "hook_mlp_out" in self.upstream_hook_types:
                    # This corresponds to MLP input
                    # mlp_in = layer.ln_2.input[0][0].grad.save()

                    # gradients[f"blocks.{i}.hook_mlp_in"] = mlp_in
                    clean_out[f"blocks.{i}.hook_mlp_out"] = layer.mlp.output.save()

            logits = model.output.logits
            value = metric(logits)
            value.backward()

        # print(gradients[f"blocks.{0}.hook_mlp_in"])
        print(clean_out[f"blocks.{0}.hook_mlp_out"])
        
        for component, activations in clean_out.items():
            if "mlp" in component:
                activations = activations.value.unsqueeze(-2)
            else:
                activations = activations.value

            upstream_activations_difference[:, :, self.upstream_hook_slices[component], :] += activations

        del clean_out
        t.cuda.empty_cache()

        for component, activations in gradients.items():
            layer = int(component.split(".")[1])

            if "mlp" in component:
                grad = activations.value.unsqueeze(-2)
                upstream_slice = self.upstream_nodes_before_mlp_layer[layer]
            else:
                grad = activations.value
                upstream_slice = self.upstream_nodes_before_layer[layer]


            result = t.matmul(
                upstream_activations_difference[:, :, upstream_slice],
                grad.transpose(-1, -2)
            ).sum(dim=0).sum(dim=0)

            self.eap_scores[upstream_slice, self.downstream_hook_slices[component]] += result
    
        self.eap_scores /= batch_size

        self.eap_scores = self.eap_scores.cpu()

    def top_edges(
            self,
            n: int = 1000,
            threshold: float = None,
            abs_scores: bool = True,
            format: bool = False,
        ):

        # get indices of maximum values in 2d tensor
        if abs_scores:
            top_scores, top_indices = t.topk(self.eap_scores.flatten().abs(), k=n, dim=0)
        else:
            top_scores, top_indices = t.topk(self.eap_scores.flatten(), k=n, dim=0)

        top_edges = []
        for i, (abs_score, index) in enumerate(zip(top_scores, top_indices)):
            if threshold is not None and abs_score < threshold:
                break

            upstream_node_idx, downstream_node_idx = np.unravel_index(index, self.eap_scores.shape)
            score = self.eap_scores[upstream_node_idx, downstream_node_idx]

            top_edges.append((self.upstream_node_names[upstream_node_idx], self.downstream_node_names[downstream_node_idx], score.item()))

        if format:
            for from_edge, to_edge, score in top_edges:
                print(f'{from_edge} -> [{round(score, 3)}] -> {to_edge}')

        return top_edges
                    