# Adapted from https://github.com/cloneofsimo/minDinoV2
import jax, math
from jax import Array, numpy as jnp
from flax import nnx
from einops import rearrange


xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0)


def linear_bicubic_interpolate(x, size=None):
    in_shape = x.shape

    scale_factors = tuple(o / i for o, i in zip(size, in_shape[2:]))

    n, c, l_out = jnp.indices((x.shape[0], x.shape[1]) + size)
    l_in_float = l_out / scale_factors[0]
    l_in = jnp.floor(l_in_float).astype(int)
    l_in_next = jnp.clip(l_in + 1, 0, in_shape[2] - 1)
    l_weight = l_in_float - l_in
    interpolated_values = (1 - l_weight) * x[n, c, l_in] + l_weight * x[n, c, l_in_next]

    return interpolated_values


class Mlp(nnx.Module):
    def __init__(self, in_feat, out_feat, hidden, bias=True, act_layer=nnx.gelu):
        super().__init__()
        hidden = hidden or in_feat
        out_feat = out_feat or in_feat
        
        self.lin_1 = nnx.Linear(in_feat, hidden, use_bias=bias, kernel_init=xavier_init)
        self.act = act_layer
        self.lin_2 = nnx.Linear(hidden, out_feat, use_bias=bias, kernel_init=xavier_init)
        
    def __call__(self, x):
        x = self.act(self.lin_1(x))
        x = self.lin_2(x)
        
        return x

class SwigluFFN(nnx.Module):
    def __init__(self, infeat, hidden):
        super().__init__()
        outfeat = infeat
        hidden = hidden or infeat
        
        self.w_12 = nnx.Linear(infeat, 2*hidden, use_bias=False, kernel_init=xavier_init)
        self.w_3 = nnx.Linear(hidden, outfeat, kernel_init=xavier_init, use_bias=True)
        
    def __call__(self, x):
        x_12 = self.w_12(x)
        x_1, x_2 = jnp.array_split(x_12, 2, axis=-1)
        hidden = nnx.silu(x_1) * x_2
        
        return self.w_3(hidden)

class PatchEmbed(nnx.Module):
    def __init__(self, img_size=224, patch_size=16, in_channel=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.conv_proj = nnx.Conv(
            in_channel, embed_dim, 
            kernel_size=patch_size, strides=patch_size,
            kernel_init=xavier_init, use_bias=False
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        num_patches_side_h = H // self.patch_size
        num_patches_side_w = W // self.patch_size

        x = self.conv_project(x)  # (B, P, P, hidden_size)
        x = rearrange(
            x, "b h w c -> b (h w) c", h=num_patches_side_h, w=num_patches_side_w
        )

        return x

class Attention(nnx.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = head_dim ** -0.5
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, kernel_init=xavier_init)
        self.out_project = nnx.Linear(dim, dim, kernel_init=xavier_init, bias_init=zero_init)
    
    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = jnp.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads)).transpose(2, 0, 1, 3, 4)
        q, k, v = jnp.array_split(qkv, 3, axis=0)
        x = nnx.dot_product_attention(q, k, v)
        x = x.reshape((B, N, C))
        x = self.out_project(x)
        
        return x

class LayerScale(nnx.Module):
    def __init__(self, dim, init_val=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nnx.Param(jnp.ones((dim)))
    
    def __call__(self, x):
        return x * self.gamma.value


class Block(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm_1 = nnx.LayerNorm(dim, scale_init=zero_init)
        self.attn = Attention(dim, num_heads)
        self.ls_1 = LayerScale(dim)
        self.norm_2 = nnx.LayerNorm(dim)

        self.mlp = Mlp(dim, hidden=int(dim*mlp_ratio))
        self.ls_2 = LayerScale(dim)

    def __call__(self, x):
        x = x + self.ls_1(self.attn(self.norm_1(x)))
        x = x + self.ls_2(self.mlp(self.norm_2(x)))

        return x

class DinoViT(nnx.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_channels=3, embed_dim=768,
        depth=12, num_heads=12, mlp_ratio=4, interpolate_offset=0.1,
        num_reg_tokens=0 
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

        self.register_tokens = nnx.Param(jnp.zeros((1, num_reg_tokens, embed_dim))) if num_reg_tokens else None

        layer_list = [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        self.layers = nnx.Sequential(*layer_list)

        self.norm = nnx.LayerNorm(embed_dim)

        self.mask_token = nnx.Param(jnp.zeros((1, embed_dim)))

    def interpolate_posencoding(self, x, h, w):
        npatches = x.shape[1] - 1
        N = self.pos_embed.value.shape[1] - 1

        if npatches == N and w == h:
            return self.pos_embed.value

        pos_embed = self.pos_embed.value.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0, h0 = w // self.patch_size, h // self.patch_size
        M = int(math.sqrt(N))

        kwargs = {}
        assert N == M * M
        kwargs['size'] = (w0, h0)

        patch_pos_embed = patch_pos_embed.reshape((1, M, M, dim)).transpose(0, 3, 1, 2)
        patch_pos_embed = linear_bicubic_interpolate(patch_pos_embed, size=kwargs['size']).transpose(0, 2, 3, 1)

        return jnp.concat((class_pos_embed[None], patch_pos_embed), axis=1)

    def _prepare_tokens_with_masks(self, x, masks=None):
        b, h, w, c = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = jnp.where(masks[..., None], self.mask_token.value[None], x)

        exp_cls = jnp.repeat(self.cls_token.value, b, axis=0)
        x = jnp.concat([exp_cls, x], axis=1)

        x = x + self.interpolate_posencoding(x, h, w)
        if self.register_tokens is not None:
            exp_reg = jnp.repeat(self.register_tokens.value, b, axis=0)
            x = jnp.concat([x[:, :1], exp_reg, x[:, 1:]])

        return x

    def __call__(self, x, masks=None):
        x = self._prepare_tokens_with_masks(x, masks)    
        x = self.layers(x)
        x = self.norm(x)

        return x


def vit_small(patch_size=14, num_register_tokens=0, **kwargs):
    return DinoViT(
        img_size=518,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
