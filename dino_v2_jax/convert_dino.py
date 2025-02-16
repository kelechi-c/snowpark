# adapted from https://github.com/GallagherCommaJack/flux-jax and https://github.com/ayaka14732/llama-2-jax

import jax, torch
import jax.numpy as jnp
import torch
from torch import nn
from flax import nnx
from einops import rearrange
from jax import Array
import numpy as np
from dinov2 import DinoViT
from reference.hfmodels.dinov2.modeling_dinov2 import Dinov2Model
from ..utils import (
    torch_embedding_to_jax_embedding,
    torch_layernorm_to_jax_layernorm,
    torch_linear_to_jax_linear,
    torch_conv_to_jax_conv
)


def convert_patchembed(jax_layer: nnx.Module, torch_layer: nn.Module):
    jax_layer.projection = torch_conv_to_jax_conv(
        torch_layer.patch_embeddings.projection
    )
    print("patch embed")
    return jax_layer


def convert_attention(jax_layer: nnx.Module, torch_layer: nn.Module):
    jax_layer.query = torch_linear_to_jax_linear(torch_layer.query)
    jax_layer.key = torch_linear_to_jax_linear(torch_layer.key)
    jax_layer.value = torch_linear_to_jax_linear(torch_layer.value)
    print("attn block")

    return jax_layer


def convert_mlp(jax_layer: nnx.Module, torch_layer: nn.Module):
    jax_layer.fc1 = torch_linear_to_jax_linear(torch_layer.fc1)
    jax_layer.fc2 = torch_linear_to_jax_linear(torch_layer.fc2)
    print("mlp block")

    return jax_layer


def convert_vit_layer(jax_layer, torch_layer):
    jax_layer.attention = convert_attention(
        jax_layer.attention, torch_layer.attention.attention
    )
    jax_layer.attention.output = torch_linear_to_jax_linear(
        torch_layer.attention.output.dense
    )

    jax_layer.norm1 = torch_layernorm_to_jax_layernorm(torch_layer.norm1)
    jax_layer.norm2 = torch_layernorm_to_jax_layernorm(torch_layer.norm2)

    jax_layer.layer_scale1.gamma.value = jnp.array(
        torch_layer.layer_scale1.lambda1.detach().numpy()
    )
    jax_layer.layer_scale2.gamma.value = jnp.array(
        torch_layer.layer_scale2.lambda1.detach().numpy()
    )

    jax_layer.mlp = convert_mlp(jax_layer.mlp, torch_layer.mlp)
    print("dinov2 layer")

    return jax_layer


def convert_dinov2(jax_model: DinoViT, torch_model):
    jax_model.patch_embed = convert_patchembed(
        jax_model.patch_embed, torch_model.embeddings
    )
    jax_model.cls_token.value = jnp.array(
        torch_model.embeddings.cls_token.detach().numpy()
    )
    jax_model.pos_embed.value = jnp.array(
        torch_model.embeddings.position_embeddings.detach().numpy()
    )
    jax_model.layer = [
        convert_vit_layer(jax_model.layer[x], torch_model.encoder.layer[x])
        for x in range(jax_model.depth)
    ]
    jax_model.layernorm = torch_layernorm_to_jax_layernorm(torch_model.layernorm)
    print("Dinov2-jax online")

    return jax_model


# jax_dinov2 = convert_dinov2(dino_model, torch_model)


# saving utils

from jax import random as jrand
import optax

rkey = jrand.key(0)

def test_outputs(jax_model, torch_model):
    rand_img = jrand.normal(rkey, (1, 224, 224, 3))
    rand_tensor = torch.randn(1, 3, 224, 224)
    
    out_ours = jax_model(rand_img)
    out_torch = jnp.array(torch_model(rand_tensor).detach().numpy())
    
    sim = optax.cosine_similarity(out_ours, out_torch)
    optax.l
    print(f'similarity: {sim:.3f}')
