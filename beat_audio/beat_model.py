import jax
from flax import nnx
from jax import numpy as jnp, Array


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