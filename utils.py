# adapted from https://github.com/GallagherCommaJack/flux-jax and https://github.com/ayaka14732/llama-2-jax
import jax.numpy as jnp
import torch, jax
from torch import nn
from flax import nnx
from einops import rearrange
import pickle
from flax.serialization import to_state_dict
from flax.core import freeze


def save_modelpickle(model, filename="model.pkl"):
    params = nnx.state(model)
    params = jax.device_get(params)

    state_dict = to_state_dict(params)
    frozen_state_dict = freeze(state_dict)
    # flat_state_dict = flax.traverse_util.flatten_dict(frozen_state_dict, sep=".")

    with open(filename, "wb") as f:
        pickle.dump(frozen_state_dict, f)



def torch_embedding_to_jax_embedding(torch_embedding: nn.Embedding) -> nnx.Embed:
    jax_embedding: nnx.Embed = nnx.eval_shape(
        lambda: nnx.Embed(
            num_embeddings=torch_embedding.num_embeddings,
            features=torch_embedding.embedding_dim,
            rngs=nnx.Rngs(0),
        )
    )
    jax_embedding.embedding.value = jnp.array(torch_embedding.weight.detach().numpy())
    return jax_embedding


def torch_linear_to_jax_linear(torch_linear: nn.Linear) -> nnx.Linear:
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


def torch_conv_to_jax_conv(torch_conv: nn.Conv2d) -> nnx.Conv:
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
