# adapted from https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/models/whisper/modeling_whisper.py
import jax, math
from jax import Array, numpy as jnp
from flax import nnx
from einops import rearrange

rngs = nnx.Rngs(0)
xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0)

class whisper_base_config:
    ffn_dim = 2048
    embed_dim = 512
    attn_heads = 8
    encoder_depth = 6
    decoder_depth = 6
    vocab_size = 51865
    num_mel_bins = 80
    pad_idx = 50257
    max_len = 448
    max_source_positions = 1500
    init_std = 0.02
    scale_embed = False


class WhisperAttention(nnx.Module):
    def __init__(self, config: whisper_base_config, qkv_bias=False):
        super().__init__()
        dim = config.embed_dim
        self.num_attention_heads = config.attn_heads
        self.head_dim = dim // config.attn_heads

        self.scale = self.head_dim**-0.5
        self.query = nnx.Linear(
            dim, dim, use_bias=qkv_bias, kernel_init=xavier_init, rngs=rngs
        )
        self.key = nnx.Linear(
            dim, dim, use_bias=qkv_bias, rngs=rngs, kernel_init=xavier_init
        )
        self.value = nnx.Linear(
            dim, dim, use_bias=qkv_bias, rngs=rngs, kernel_init=xavier_init
        )

        self.output = nnx.Linear(
            dim, dim, rngs=rngs, kernel_init=xavier_init, bias_init=zero_init
        )

    def __call__(self, x):
        B, N, C = x.shape

        q = jnp.reshape(self.query(x), (B, N, self.num_attention_heads, self.head_dim))
        k = jnp.reshape(self.key(x), (B, N, self.num_attention_heads, self.head_dim))
        v = jnp.reshape(self.value(x), (B, N, self.num_attention_heads, self.head_dim))

        x = nnx.dot_product_attention(q, k, v)
        x = x.reshape((B, N, C))
        x = self.output(x)

        return x

class WhisperEncoderLayer(nnx.Module):
    def __init__(self, config: whisper_base_config):
        super().__init__()
        dim = config.embed_dim

        self.attention = WhisperAttention(dim, config.attn_heads)
        self.attn_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.linear_1 = nnx.Linear(
            dim, config.ffn_dim, use_bias=False, kernel_init=xavier_init, rngs=rngs
        )
        self.linear_2 = nnx.Linear(
            dim, config.ffn_dim, use_bias=False, kernel_init=xavier_init, rngs=rngs
        )
        self.final_norm = nnx.LayerNorm(dim, rngs=rngs)

    def __call__(self, x: Array):
        res = x
        x = res + self.attention(self.attn_norm(x))
        
        res = x
        x = nnx.gelu(self.linear_1(self.final_norm(x)))
        x = res + self.linear_2(x)
        
        return x

class WhisperEncoder(nnx.Module):
    def __init__(self, config: whisper_base_config):
        super().__init__()
        dim = config.embed_dim

        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.pad_idx = config.pad_idx
        self.embed_scale = math.sqrt(dim) if config.scale_embed else 1.0

        self.conv_1 = nnx.Conv(self.num_mel_bins, dim, kernel_size=3, padding=1, rngs=rngs)
        self.conv_2 = nnx.Conv(dim, dim, kernel_size=3, strides=2, padding=1, rngs=rngs)

        self.embed_positions = nnx.Embed(self.max_source_positions, dim, rngs=rngs)
        jax.lax.stop_gradient(self.embed_positions.embedding.value)

        self.layers = [WhisperEncoderLayer(config) for _ in range(config.encoder_depth)]
        self.layernorm = nnx.LayerNorm(dim, rngs=rngs)

    def __call__(self, x: Array):
        expected_seqlen = self.max_source_positions * self.conv_1.strides[0] * self.conv_2.strides[0]
        if x.shape[-1] != expected_seqlen:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seqlen}, but found {x.shape[-1]}. Make sure to pad the input mel features to {expected_seqlen}."
            )
        input_embeds = nnx.gelu(self.conv_1(x))
        input_embeds = nnx.gelu(self.conv_2(input_embeds)).transpose(0, 2, 1)
        embed_pos = self.embed_positions.embedding.value

        hidden_state = input_embeds + embed_pos

        for encoder_layer in self.layers:
            hidden_state = encoder_layer(hidden_state)

        hidden_state = self.layernorm(hidden_state)

        return hidden_state

