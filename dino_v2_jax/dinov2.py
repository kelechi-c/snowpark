# Adapted from https://github.com/cloneofsimo/minDinoV2
import jax, math
from jax import Array, numpy as jnp
from flax import nnx
from einops import rearrange

rngs = nnx.Rngs(0)
xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0)


def linear_bicubic_interpolate(patch_pos_embed, M, dim, target_size, **kwargs):

    reshaped = patch_pos_embed.reshape(
        M, M, dim
    )  # remove batch dimension for jax resize
    resized = jax.image.resize(reshaped, shape=target_size + (dim,), method="bicubic")
    # Add back the batch dimension and change data format to pytorch's NCHW
    resized = jnp.expand_dims(resized, axis=0)
    return resized


class Mlp(nnx.Module):
    def __init__(
        self, in_feat, out_feat=None, hidden=None, bias=True, act_layer=nnx.gelu
    ):
        super().__init__()
        hidden = hidden or in_feat
        out_feat = out_feat or in_feat

        self.fc1 = nnx.Linear(
            in_feat, hidden, use_bias=bias, kernel_init=xavier_init, rngs=rngs
        )
        self.activation = act_layer
        self.fc2 = nnx.Linear(
            hidden, out_feat, use_bias=bias, kernel_init=xavier_init, rngs=rngs
        )

    def __call__(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x


class PatchEmbed(nnx.Module):
    def __init__(self, img_size=224, patch_size=16, in_channel=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        self.patch_size = patch_size
        patch_size = (patch_size, patch_size)

        self.num_channels = in_channel
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.projection = nnx.Conv(
            in_channel,
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            kernel_init=xavier_init,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        num_patches_side_h = H // self.patch_size
        num_patches_side_w = W // self.patch_size

        x = self.projection(x)  # (B, P, P, hidden_size)
        x = rearrange(
            x, "b h w c -> b (h w) c", h=num_patches_side_h, w=num_patches_side_w
        )

        return x


class Attention(nnx.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, drop=0.0):
        super().__init__()
        self.num_attention_heads = num_heads
        self.head_dim = dim // num_heads

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
        self.dropout = nnx.Dropout(0.0, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape

        q = jnp.reshape(self.query(x), (B, N, self.num_attention_heads, self.head_dim))
        k = jnp.reshape(self.key(x), (B, N, self.num_attention_heads, self.head_dim))
        v = jnp.reshape(self.value(x), (B, N, self.num_attention_heads, self.head_dim))

        x = nnx.dot_product_attention(q, k, v)
        x = x.reshape((B, N, C))
        x = self.output(x)

        return x


class LayerScale(nnx.Module):
    def __init__(self, dim, init_val=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nnx.Param(init_val * jnp.ones((dim)))

    def __call__(self, x):
        return x * self.gamma.value


class Block(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nnx.LayerNorm(dim, scale_init=zero_init, rngs=rngs)
        self.attention = Attention(dim, num_heads)
        self.layer_scale1 = LayerScale(dim)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)

        self.mlp = Mlp(dim, hidden=int(dim * mlp_ratio))
        self.layer_scale2 = LayerScale(dim)

    def __call__(self, x):
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))

        return x


class DinoViT(nnx.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        in_channels=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        interpolate_offset=0.1,
        num_reg_tokens=0,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.depth = depth
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim=embed_dim)

        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        self.pos_embed = nnx.Param(jnp.zeros((1, 1370, embed_dim)))

        self.register_tokens = (
            nnx.Param(jnp.zeros((1, num_reg_tokens, embed_dim)))
            if num_reg_tokens
            else None
        )
        for i in range(depth):
            block = Block(embed_dim, num_heads, mlp_ratio)
            setattr(self, f"blocks_{i}", block)

        self.layernorm = nnx.LayerNorm(embed_dim, rngs=rngs)

        self.mask_token = nnx.Param(jnp.zeros((1, embed_dim)))

    def interpolate_posencoding(self, x, h, w):
        npatches = x.shape[1] - 1
        N = self.pos_embed.value.shape[1] - 1

        if npatches == N and w == h:
            return self.pos_embed.value

        pos_embed = self.pos_embed.value
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0, h0 = w // self.patch_size, h // self.patch_size
        M = int(math.sqrt(N))

        kwargs = {}
        assert N == M * M
        kwargs["size"] = (w0, h0)

        patch_pos_embed = patch_pos_embed.reshape((1, M, M, dim)).transpose(0, 3, 1, 2)
        patch_pos_embed = (
            linear_bicubic_interpolate(
                patch_pos_embed, M, dim, target_size=kwargs["size"]
            )
            .transpose(0, 2, 3, 1)
            .reshape((1, -1, dim))
        )

        class_pos_embed = class_pos_embed[None]
        # patch_pos_embed = patch_pos_embed[None]
        # print(f"{class_pos_embed.shape = }")
        # print(f"{patch_pos_embed.shape = }")
        interp_out = jnp.concat((class_pos_embed, patch_pos_embed), axis=1)

        return interp_out

    def _prepare_tokens_with_masks(self, x, masks=None):
        b, h, w, c = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = jnp.where(masks[..., None], self.mask_token.value[None], x)

        exp_cls = jnp.repeat(self.cls_token.value, b, axis=0)
        x = jnp.concat([exp_cls, x], axis=1)

        x = x + self.interpolate_posencoding(x, h, w)
        print(f"interp {x.shape = }")
        if self.register_tokens is not None:
            exp_reg = jnp.repeat(self.register_tokens.value, b, axis=0)
            x = jnp.concat([x[:, :1], exp_reg, x[:, 1:]])

        return x

    def __call__(self, x, masks=None):
        x = self._prepare_tokens_with_masks(x, masks)

        for i in range(self.depth):
            _block = getattr(self, f"blocks_{i}")
            x = _block(x)

        x = self.layernorm(x)

        return x


dino_model = DinoViT(
    img_size=518,
    patch_size=14,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4,
    num_reg_tokens=0,
)
