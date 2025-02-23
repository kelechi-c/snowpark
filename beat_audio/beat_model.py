import jax, math
from flax import nnx
from jax import numpy as jnp, Array

rngs = nnx.Rngs(0)


class beats_config:
    input_patch_size: int = -1  # path size of patch embedding
    embed_dim: int = 512  # patch embedding dimension
    conv_bias: bool = False  # include bias in conv encoder

    encoder_layers: int = 12  # num encoder layers in the transformer
    encoder_embed_dim: int = 768  # encoder embedding dimension
    encoder_ffn_embed_dim: int = 3072  # encoder embedding dimension for FFN
    encoder_attention_heads: int = 12  # num encoder attention heads
    activation_fn: str = "gelu"  # activation function to use

    layer_wise_gradient_decay_ratio: float = 1.0  # ratio for layer-wise gradient decay
    layer_norm_first: bool = False  # apply layernorm first in the transformer
    deep_norm: bool = False  # apply deep_norm first in the transformer

    # dropouts
    dropout: float = 0.1  # dropout probability for the transformer
    attention_dropout: float = 0.1  # dropout probability for attention weights
    activation_dropout: float = 0.0  # dropout probability after activation in FFN
    encoder_layerdrop: float = 0.0  # probability of dropping a tarnsformer layer
    dropout_input: float = 0.0  # dropout to apply to the input (after feat extr)

    # positional embeddings
    conv_pos: int = 128  # number of filters for convolutional positional embeddings
    conv_pos_groups: int = 16  # number of groups for convolutional positional embedding

    # relative position embedding
    relative_position_embedding: bool = False  # apply relative position embedding
    num_buckets: int = 320  # number of buckets for relative position embedding
    max_distance: int = 1280  # maximum distance for relative position embedding
    gru_rel_pos: bool = False  # apply gated relative position embedding

    # label predictor
    finetuned_model: bool = False  # whether the model is a fine-tuned model.
    predictor_dropout: float = 0.1  # dropout probability for the predictor
    predictor_class: int = 527  # target class number for the predictor


class SamePad(nnx.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class SimpleAttention(nnx.Module):
    def __init__(
        self,
        config: beats_config,
        is_decoder=False,
        is_causal=False,
        qkv_bias=False,
        layer_idx=None,
    ):
        super().__init__()
        dim = config.embed_dim
        self.num_attention_heads = config.attn_heads
        self.head_dim = dim // config.attn_heads

        self.is_causal = is_causal
        self.is_decoder = is_decoder
        self.layer_idx = layer_idx

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

    def __call__(
        self,
        x: Array,
        key_value: Array = None,
        mask: Array = None,
        past_key_value: Array = None,
    ):
        B, N, C = x.shape

        kv = key_value if key_value is not None else x

        if self.is_causal and mask is None:
            mask = jnp.ones((N, N))

        q = jnp.reshape(self.query(x), (B, N, self.num_attention_heads, self.head_dim))
        k = jnp.reshape(self.key(kv), (B, N, self.num_attention_heads, self.head_dim))
        v = jnp.reshape(self.value(kv), (B, N, self.num_attention_heads, self.head_dim))

        x = nnx.dot_product_attention(q, k, v, mask=mask)
        x = x.reshape((B, N, C))
        x = self.output(x)

        return x
    
    

# (Optionally define a quant_noise wrapper; here we simply pass-through.)
def quant_noise(module, q_noise, qn_block_size):
    # In this port, we assume no quant noise.
    return module


class MultiheadAttention(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: int = None,
        vdim: int = None,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        gru_rel_pos: bool = False,
        rescale_init: bool = False,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.gru_rel_pos = gru_rel_pos
        self.add_zero_attn = add_zero_attn

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        # Dropout layer (using the default dropout stream)
        self.dropout = nnx.Dropout(self.dropout_rate, rngs=rngs)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nnx.Embed(
                num_embeddings=num_buckets, features=num_heads, rngs=rngs
            )

        # Projections – we use nnx.Linear (and wrap with quant_noise if needed)
        self.q_proj = quant_noise(
            nnx.Linear(
                in_features=embed_dim, out_features=embed_dim, rngs=rngs, use_bias=bias
            ),
            q_noise,
            qn_block_size,
        )
        self.k_proj = quant_noise(
            nnx.Linear(
                in_features=self.kdim,
                out_features=embed_dim,
                rngs=rngs,
                use_bias=(not rescale_init),
            ),
            q_noise,
            qn_block_size,
        )
        self.v_proj = quant_noise(
            nnx.Linear(
                in_features=self.vdim, out_features=embed_dim, rngs=rngs, use_bias=bias
            ),
            q_noise,
            qn_block_size,
        )
        self.out_proj = quant_noise(
            nnx.Linear(
                in_features=embed_dim, out_features=embed_dim, rngs=rngs, use_bias=bias
            ),
            q_noise,
            qn_block_size,
        )

        if add_bias_kv:
            self.bias_k = nnx.Param(jnp.zeros((1, 1, embed_dim)))
            self.bias_v = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        else:
            self.bias_k = None
            self.bias_v = None

        if gru_rel_pos:
            self.grep_linear = nnx.Linear(
                in_features=self.head_dim, out_features=8, rngs=rngs
            )
            self.grep_a = nnx.Param(jnp.ones((1, num_heads, 1, 1)))

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = jnp.zeros_like(relative_positions, dtype=jnp.int32)
        if bidirectional:
            num_buckets = num_buckets // 2
            sign = (relative_positions > 0).astype(jnp.int32)
            relative_buckets += sign * num_buckets
            relative_positions = jnp.abs(relative_positions)
        else:
            relative_positions = -jnp.minimum(relative_positions, 0)
        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact
        relative_pos_if_large = max_exact + (
            jnp.log(relative_positions.astype(jnp.float32) / max_exact + 1e-6)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(jnp.int32)
        relative_pos_if_large = jnp.minimum(relative_pos_if_large, num_buckets - 1)
        relative_buckets += jnp.where(
            is_small, relative_positions, relative_pos_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = jnp.arange(query_length)[:, None]
        memory_position = jnp.arange(key_length)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(
            relative_position, bidirectional=True
        )
        # Look up bias: shape (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # Permute to (num_heads, query_length, key_length)
        values = jnp.transpose(values, (2, 0, 1))
        return values

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        # Stub: simply return the weights unchanged.
        return attn_weights

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray = None,
        value: jnp.ndarray = None,
        key_padding_mask: jnp.ndarray = None,
        incremental_state: dict = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: jnp.ndarray = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        position_bias: jnp.ndarray = None,
    ):
        # Assume inputs have shape [tgt_len, bsz, embed_dim]
        tgt_len, bsz, embed_dim = query.shape
        if key is None:
            key = query
        if value is None:
            value = key

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            k = self.k_proj(key) if key is not None else None
            v = self.v_proj(key) if key is not None else None
        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q = q * self.scaling
        # Additional scaling (alpha = 32) as in the original code
        alpha = 32.0
        q = q / alpha

        if self.bias_k is not None and self.bias_v is not None:
            bias_k = jnp.tile(self.bias_k.value, (1, bsz, 1))
            bias_v = jnp.tile(self.bias_v.value, (1, bsz, 1))
            k = jnp.concatenate([k, bias_k], axis=0)
            v = jnp.concatenate([v, bias_v], axis=0)
            if attn_mask is not None:
                attn_mask = jnp.concatenate(
                    [attn_mask, jnp.zeros((attn_mask.shape[0], 1))], axis=1
                )
            if key_padding_mask is not None:
                key_padding_mask = jnp.concatenate(
                    [key_padding_mask, jnp.zeros((key_padding_mask.shape[0], 1))],
                    axis=1,
                )

        # Reshape q, k, v for multi-head attention.
        q = q.reshape(tgt_len, bsz, self.num_heads, self.head_dim)
        q = jnp.transpose(q, (1, 0, 2, 3)).reshape(
            bsz * self.num_heads, tgt_len, self.head_dim
        )

        k = k.reshape(-1, bsz, self.num_heads, self.head_dim)
        k = jnp.transpose(k, (1, 0, 2, 3)).reshape(
            bsz * self.num_heads, -1, self.head_dim
        )

        v = v.reshape(-1, bsz, self.num_heads, self.head_dim)
        v = jnp.transpose(v, (1, 0, 2, 3)).reshape(
            bsz * self.num_heads, -1, self.head_dim
        )

        src_len = k.shape[1]

        # (Incremental state caching omitted for brevity.)

        attn_weights = jnp.matmul(q, jnp.transpose(k, (0, 2, 1)))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            mask = jnp.expand_dims(key_padding_mask, axis=1)  # (bsz, 1, src_len)
            mask = jnp.repeat(
                mask, self.num_heads, axis=0
            )  # (bsz*num_heads, 1, src_len)
            attn_weights = jnp.where(mask, -1e9, attn_weights)

        if before_softmax:
            return attn_weights, v, position_bias

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = jnp.expand_dims(position_bias, axis=0)
            position_bias = jnp.tile(position_bias, (bsz, 1, 1, 1))
            position_bias = position_bias.reshape(
                bsz * self.num_heads, tgt_len, src_len
            )

        if position_bias is not None:
            if self.gru_rel_pos:
                query_layer = (
                    q.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
                    * alpha
                    / self.scaling
                )
                gate = jax.nn.sigmoid(
                    self.grep_linear(query_layer).sum(-1, keepdims=True)
                )
                gate = gate * self.grep_a.value
                gate = gate.reshape(bsz * self.num_heads, tgt_len, 1)
                position_bias = gate * position_bias
            attn_weights = attn_weights + position_bias

        attn_weights_float = jax.nn.softmax(attn_weights, axis=-1)
        attn_probs = self.dropout(attn_weights_float)
        attn = jnp.matmul(attn_probs, v)

        # Reshape back to (tgt_len, bsz, embed_dim)
        attn = attn.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
        attn = jnp.transpose(attn, (2, 0, 1, 3)).reshape(tgt_len, bsz, self.embed_dim)
        output = self.out_proj(attn)

        if need_weights:
            attn_weights_avg = attn_weights_float.reshape(
                bsz, self.num_heads, tgt_len, src_len
            )
            if not need_head_weights:
                attn_weights_avg = jnp.mean(attn_weights_avg, axis=1)
            return output, attn_weights_avg, position_bias
        else:
            return output, None, position_bias


class BeatEncoder(nnx.Module):
    def __init__(self, config: beats_config):
        super().__init__()
        self.config = config
        self.embed_dim = config.encoder_embed_dim
        self.pos_conv = nnx.Conv(
            self.embed_dim, self.embed_dim,
            kernel_size=config.conv_pos, padding=config.conv_pos // 2,
            feature_group_count=config.conv_pos_groups, rngs=rngs
        )
        self.pos_conv = nnx.Sequential(self.pos_conv,  nnx.gelu)        
