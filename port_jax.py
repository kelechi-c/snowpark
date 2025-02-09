# adapted from https://github.com/GallagherCommaJack/flux-jax and https://github.com/ayaka14732/llama-2-jax

import jax, torch
import jax.numpy as jnp
import torch
from flax import nnx
from typing import Callable
from jax import Array
import numpy as np


def jax2np(x: Array) -> np.ndarray:
    return np.asarray(x)


def np2jax(x: np.ndarray) -> Array:
    return jnp.asarray(x)


def pt2np(x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return x.numpy()


def pt2jax(x: torch.Tensor) -> Array:
    return np2jax(pt2np(x))


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
    dense.kernel.value = jnp.array(torch_linear.weight.T.numpy())
    if torch_linear.bias is not None:
        dense.bias.value = jnp.array(torch_linear.bias.numpy())
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
