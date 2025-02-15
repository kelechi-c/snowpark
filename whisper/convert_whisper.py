import jax, torch
import jax.numpy as jnp
import torch
from torch import nn
from flax import nnx
from einops import rearrange
from jax import Array

from ..utils import torch_conv_to_jax_conv, torch_embedding_to_jax_embedding, torch_layernorm_to_jax_layernorm, torch_linear_to_jax_linear
from ..whisper import Whisper
from ..reference.modeling_whisper import WhisperModel, WhisperForConditionalGeneration


def convert_attention(jax_layer: nnx.Module, torch_layer: nn.Module):
    jax_layer.query = torch_linear_to_jax_linear(torch_layer.q_proj)
    jax_layer.key = torch_linear_to_jax_linear(torch_layer.k_proj)
    jax_layer.value = torch_linear_to_jax_linear(torch_layer.v_proj)
    jax_layer.output = torch_linear_to_jax_linear(torch_layer.out_proj)
    print("attn block")

    return jax_layer


# def convert_encoder(jax_layer: nnx.Module, torch_layer: nn.Module):
#     jax_layer.fc1 = torch_linear_to_jax_linear(torch_layer.fc1)
#     jax_layer.fc2 = torch_linear_to_jax_linear(torch_layer.fc2)
#     print("mlp block")

#     return jax_layer

def convert_encoder_layer(jax_layer, torch_layer):
    jax_layer.attention = convert_attention(
        jax_layer.attention, torch_layer.self_attn
    )

    jax_layer.attn_norm = torch_layernorm_to_jax_layernorm(torch_layer.self_attn_layer_norm)
    jax_layer.final_norm = torch_layernorm_to_jax_layernorm(torch_layer.final_layer_norm)

    jax_layer.linear_1 = torch_linear_to_jax_linear(torch_layer.fc1)
    jax_layer.linear_2 = torch_linear_to_jax_linear(torch_layer.fc2)    
    print("encoder layer")

    return jax_layer


def convert_decoder_layer(jax_layer, torch_layer):
    jax_layer.self_attn = convert_attention(
        jax_layer.self_attn, torch_layer.self_attn
    )
    jax_layer.cross_attn = convert_attention(jax_layer.cross_attn, torch_layer.encoder_attn)

    jax_layer.self_attn_layernorm = torch_layernorm_to_jax_layernorm(torch_layer.self_attn_layer_norm)
    jax_layer.cross_attn_norm = torch_layernorm_to_jax_layernorm(torch_layer.encoder_attn_layer_norm)

    jax_layer.linear_1 = torch_linear_to_jax_linear(torch_layer.fc1)
    jax_layer.linear_2 = torch_linear_to_jax_linear(torch_layer.fc2)    
    
    jax_layer.final_layer_norm = torch_layernorm_to_jax_layernorm(torch_layer.final_layer_norm)
    
    print("decoder layer")
    return jax_layer


def convert_encoder(jax_layer, torch_layer) -> nnx.Module:
    jax_layer.conv_1 = torch_conv_to_jax_conv(torch_layer.conv1)
    jax_layer.conv_2 = torch_conv_to_jax_conv(torch_layer.conv2)

    jax_layer.embed_positions = torch_embedding_to_jax_embedding(torch_layer.embed_positions)

    jax_layer.layers = [
        convert_encoder_layer(jax_layer.layers[x], torch_layer.layers[x])
        for x in range(jax_layer.config.encoder_depth)
    ]
    jax_layer.layernorm = torch_layernorm_to_jax_layernorm(torch_layer.layernorm) 
    print("encoder.../")

    return jax_layer


def convert_decoder(jax_layer, torch_layer) -> nnx.Module:
    jax_layer.embed_tokens = torch_embedding_to_jax_embedding(torch_layer.embed_tokens)
    jax_layer.embed_positions = torch_embedding_to_jax_embedding(torch_layer.embed_positions)

    jax_layer.layers = [
        convert_decoder_layer(jax_layer.layers[x], torch_layer.layers[x])
        for x in range(jax_layer.config.decoder_depth)
    ]
    jax_layer.layernorm = torch_layernorm_to_jax_layernorm(torch_layer.layernorm)
    print('decoder.../')
    
    return jax_layer


def convert_whisper(jax_model: Whisper, torch_model: WhisperForConditionalGeneration) -> nnx.Module:
    jax_model.encoder = convert_encoder(torch_model.model.encoder)
    jax_model.decoder = convert_decoder(jax_model.decoder, torch_model.model.decoder)
    jax_model.linear_head = torch_linear_to_jax_linear(torch_model.proj_out)
    print("whisper-jax online :)")

    return jax_model


# jax_whisper = convert_whisper(whisper_model, torch_model)


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
    print(f"similarity: {sim:.3f}")
