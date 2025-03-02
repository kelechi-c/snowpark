'''
smollm modelling code, it's Llama-based anyways so...
'''

import jax, math
import numpy as np
from jax import Array, numpy as jnp, random as jrand
from flax import nnx
from typing import Optional, Tuple

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

rngs = nnx.Rngs(0)
xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0)


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


class SwigluFFN(nnx.Module):
    def __init__(self, config=smollm_tiny_config):
        super().__init__()
        
        outfeat = config.hidden_size
        hidden = config.intermediate_size or config.hidden_size

        self.up_project = nnx.Linear(
            config.hidden_size, hidden, use_bias=False, kernel_init=xavier_init, rngs=rngs
        )

        self.gate_project = nnx.Linear(
            hidden, outfeat, kernel_init=xavier_init, use_bias=False, rngs=rngs
        )
        self.down_project = nnx.Linear(
            hidden, outfeat, kernel_init=xavier_init, use_bias=False, rngs=rngs
        )

    def __call__(self, x):
        out = self.down_project(nnx.silu(self.gate_project(x)) * self.up_project(x))
        return out


def repeat_kv(hidden_states: jax.Array, num_key_value_groups: int) -> jax.Array:
    if num_key_value_groups == 1:
        return hidden_states
    batch, seq_len, num_heads, head_dim = hidden_states.shape
    hidden_states = jnp.tile(hidden_states, (1, num_key_value_groups, 1, 1))
    return hidden_states

def create_mask(seq_len):
    N = seq_len
    attention_mask = jnp.ones((N, N))
    causal_mask = jnp.tril(jnp.ones((N, N)))

    attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
    causal_mask = jnp.expand_dims(causal_mask, axis=(-3, -2))
    print(f'{attention_mask.shape = }, {causal_mask.shape = }')
    
    mask = attention_mask * causal_mask
    
    
    return mask

class LlamaAttention(nnx.Module):
    def __init__(self, config: smollm_tiny_config, layer_id=None, qkv_bias=False, drop=0.0):
        super().__init__()
        dim = config.hidden_size
        num_heads = config.num_attention_heads
        self.num_attention_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_key_value_groups = self.num_attention_heads // num_heads
        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.is_causal = True
        self.layer_id = layer_id
        self.rope_theta = config.rope_theta
        self.max_pos_embeddings = config.max_position_embeddings

        self.rotary_emb = LlamaRotaryEmbedding(config)

        self.query = nnx.Linear(
            dim, dim, use_bias=qkv_bias, kernel_init=xavier_init, rngs=rngs
        )
        self.key = nnx.Linear(
            dim, dim, use_bias=qkv_bias, rngs=rngs, kernel_init=xavier_init
        )
        self.value = nnx.Linear(
            dim, dim, use_bias=qkv_bias, rngs=rngs, kernel_init=xavier_init
        )

        self.output = nnx.Linear(dim, dim, rngs=rngs, kernel_init=xavier_init, bias_init=zero_init)
        self.dropout = nnx.Dropout(0.0, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
        past_key_value: Optional[Tuple[jax.Array, jax.Array]] = None,  # type: ignore
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[jax.Array] = None,
        position_embeddings: Optional[
            Tuple[jax.Array, jax.Array]
        ] = None, 
        **kwargs,
    ) -> Tuple:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)

        query_states = jnp.reshape(
            query_states, (bsz, q_len, self.num_attention_heads, self.head_dim)
        )
        key_states = jnp.reshape(
            key_states, (bsz, q_len, self.num_attention_heads, self.head_dim)
        )
        value_states = jnp.reshape(
            value_states, (bsz, q_len, self.num_attention_heads, self.head_dim)
        )

        if position_embeddings is None:
            cos, sin = self.rotary_emb(key_states, query_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states, sin, cos
        ), apply_rotary_pos_emb(key_states, sin, cos)

        if past_key_value is not None:
            key_states = jnp.concatenate((past_key_value[0], key_states), axis=1)
            value_states = jnp.concatenate((past_key_value[1], value_states), axis=1)

        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        key_states = jnp.repeat(key_states, self.num_key_value_groups, axis=2)
        value_states = jnp.repeat(value_states, self.num_key_value_groups, axis=2)

        causal_mask = None
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            # attn_weights = attn_weights + causal_mask

        if attention_mask is None:
            attention_mask = create_mask(q_len)
            causal_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        attn_output = nnx.dot_product_attention(
            query_states, key_states, value_states, mask=attention_mask
        )

        # attn_weights = jnp.matmul(
        #     query_states, jnp.transpose(key_states, (0, 1, 3, 2))
        # ) / math.sqrt(self.head_dim)

        # # upcast attention to fp32
        # attn_weights = nnx.softmax(attn_weights, axis=-1, dtype=jnp.float32)
        # attn_weights = nnx.Dropout(rate=self.attention_dropout, rngs=rngs)(
        #     attn_weights
        # )  # type:ignore
        # attn_output = jnp.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_attention_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, q_len, self.num_attention_heads, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        # attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = jnp.reshape(attn_output, (bsz, q_len, self.hidden_size))

        attn_output = self.output(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nnx.Module):
    def __init__(self, config: smollm_tiny_config = smollm_tiny_config, layer_id=None):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_id=layer_id)
        self.mlp = SwigluFFN(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, weights, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # residual connection
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + hidden_states

        return hidden_states
    
    
class Llama3(nnx.Module):
    def __init__(self, config=smollm_tiny_config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nnx.Embed(config.vocab_size, config.hidden_size, self.padding_idx, rngs=rngs)
        self.layers = [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
    
    def __call__(self):
        
