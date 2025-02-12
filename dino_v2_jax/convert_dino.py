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


def torch_embedding_to_jax_embedding(torch_embedding: torch.nn.Embedding) -> nnx.Embed:
    jax_embedding: nnx.Embed = nnx.eval_shape(
        lambda: nnx.Embed(
            num_embeddings=torch_embedding.num_embeddings,
            features=torch_embedding.embedding_dim,
            rngs=nnx.Rngs(0),
        )
    )
    jax_embedding.embedding.value = jnp.array(torch_embedding.weight.detach().numpy())
    return jax_embedding


def torch_linear_to_jax_linear(torch_linear: torch.nn.Linear) -> nnx.Linear:
    dense: nnx.Linear = nnx.eval_shape(
        lambda: nnx.Linear(
            in_features=torch_linear.in_features,
            out_features=torch_linear.out_features,
            use_bias=torch_linear.bias is not None,
            rngs=nnx.Rngs(0),
        )
    )
    dense.kernel.value = jnp.array(torch_linear.weight.T.detach().numpy())
    if torch_linear.bias is not None:
        dense.bias.value = jnp.array(torch_linear.bias.detach().numpy())
    return dense


def torch_layernorm_to_jax_layernorm(
    torch_layernorm: torch.nn.LayerNorm,
) -> nnx.LayerNorm:
    jax_layernorm: nnx.LayerNorm = nnx.eval_shape(
        lambda: nnx.LayerNorm(
            num_features=torch_layernorm.normalized_shape[0],
            epsilon=torch_layernorm.eps,
            use_bias=torch_layernorm.bias is not None,
            use_scale=torch_layernorm.weight is not None,
            rngs=nnx.Rngs(0),
        )
    )
    if torch_layernorm.weight is not None:
        jax_layernorm.scale.value = jnp.array(torch_layernorm.weight.detach().numpy())
    if torch_layernorm.bias is not None:
        jax_layernorm.bias.value = jnp.array(torch_layernorm.bias.detach().numpy())
    return jax_layernorm


def torch_conv_to_jax_conv(torch_conv: torch.nn.Conv2d) -> nnx.Conv:
    dense: nnx.Linear = nnx.eval_shape(
        lambda: nnx.Linear(
            in_features=torch_conv.in_channels,
            out_features=torch_conv.out_channels,
            use_bias=torch_conv.bias is not None,
            rngs=nnx.Rngs(0),
        )
    )
    res_weight = rearrange(torch_conv.weight, "d c h w -> h w c d")
    dense.kernel.value = jnp.array(res_weight.detach().numpy())
    if torch_conv.bias is not None:
        dense.bias.value = jnp.array(torch_conv.bias.detach().numpy())
    return dense


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


jax_dinov2 = convert_dinov2(dino_model, torch_model)


# saving utils
import flax.traverse_util, pickle
from flax.serialization import to_state_dict, from_state_dict
from flax.core import freeze, unfreeze


def save_modelpickle(model, filename="model.pkl"):
    params = nnx.state(model)
    params = jax.device_get(params)

    state_dict = to_state_dict(params)
    frozen_state_dict = freeze(state_dict)

    flat_state_dict = flax.traverse_util.flatten_dict(frozen_state_dict, sep=".")

    with open(filename, "wb") as f:
        pickle.dump(frozen_state_dict, f)

    return flat_state_dict

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