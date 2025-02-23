'''
smollm modelling code, it's Llama-based anyways so...
'''

import jax
import numpy as np
from jax import Array, numpy as jnp, random as jrand
from flax import nnx


# for 135M model
class smollm_tiny_config:
    vocab_size=49152
    hidden_size=576
    intermediate_size=1536
    num_hidden_layers=30
    num_attention_heads=9
    num_key_value_heads=3
    hidden_act="silu"
    max_position_embeddings=2048
    initializer_range=0.02
    rms_norm_eps=1e-05
    use_cache=True
    pad_token_id=None
    bos_token_id=0
    eos_token_id=0
    pretraining_tp=1
    tie_word_embeddings=False
    rope_theta=10000.0
    rope_scaling=None
    attention_bias=False
    attention_dropout=0.0
    mlp_bias=False


def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]),
        axis=-1,
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


class LlamaRMSNorm(nnx.Module):
    def __init__(self, config: smollm_tiny_config):
        super().__init__()
        self.config = config
        self.epsilon = self.config.rms_norm_eps
        self.weight = nnx.Param()

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)


class LlamaRotaryEmbedding(nnx.Module):
    def __init__(self, config: smollm_tiny_config):
        super().__init__()
        self.config = config
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.sincos = create_sinusoidal_positions(
            self.config.max_position_embeddings, head_dim
        )

    def __call__(self, key, query, position_ids):
        sincos = self.sincos[position_ids]
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)

        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

        key = jnp.asarray(key)
        query = jnp.asarray(query)

        return key, query
