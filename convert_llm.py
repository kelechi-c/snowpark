# adapted from https://github.com/GallagherCommaJack/flux-jax and https://github.com/ayaka14732/llama-2-jax

import jax, torch
import jax.numpy as jnp
import torch
from torch import nn
from flax import nnx
from einops import rearrange
from jax import Array
import numpy as np
from .smollm import Llama
from .reference.hfmodels.llama.modeling_llama import LlamaForCausalLM
# from .reference.smollm.vision.m4.smolvlm.vllama3.make_tiny_llama3 import LlamaForCausalLM
from .utils import (
    torch_layernorm_to_jax_layernorm,
    torch_linear_to_jax_linear,
    torch_embedding_to_jax_embedding
)


def convert_attention(jax_layer: nnx.Module, torch_layer: nn.Module):
    jax_layer.query = torch_linear_to_jax_linear(torch_layer.query)
    jax_layer.key = torch_linear_to_jax_linear(torch_layer.key)
    jax_layer.value = torch_linear_to_jax_linear(torch_layer.value)
    print("attn block")

    return jax_layer


def convert_mlp_swiglu(jax_layer: nnx.Module, torch_layer: nn.Module):
    jax_layer.up_proj = torch_linear_to_jax_linear(torch_layer.up_proj)
    jax_layer.gate_proj = torch_linear_to_jax_linear(torch_layer.gate_proj)
    jax_layer.down_proj = torch_linear_to_jax_linear(torch_layer.down_proj)

    print("mlp block")

    return jax_layer


def convert_layer(jax_layer, torch_layer):
    jax_layer.attention = convert_attention(
        jax_layer.attention, torch_layer.attention.attention
    )
    jax_layer.attention.output = torch_linear_to_jax_linear(
        torch_layer.attention.output.dense
    )
    
    jax_layer.mlp = convert_mlp(jax_layer.mlp, torch_layer.mlp)
    print("dinov2 layer")

    return jax_layer


def convert_llama(jax_model: Llama, torch_model: LlamaForCausalLM):
    jax_model.embed_tokens = torch_embedding_to_jax_embedding(torch_model.model.embed_tokens)
    jax_model.layers = [
        convert_layer(jax_model.layers[x], torch_model.model.layers[x])
        for x in range(jax_model.depth)
    ]
    jax_model.linear_head = torch_linear_to_jax_linear(torch_model.lm_head)
    print("llama model(smollm) online")

    return jax_model


from jax import random as jrand
import optax


def test_outputs(jax_model, torch_model):
    rand_tensor = torch.randn(1, 128)
    rand_img = (rand_tensor).transpose(0, 2, 3, 1)

    out_ours = jax_model(rand_img)
    out_torch = jnp.array(torch_model(rand_tensor).last_hidden_state.detach().numpy())
    sim = optax.cosine_similarity(out_ours, out_torch)

    print(f"similarity: {sim.mean().item() * 100:.3f}%")