import jax
from jax import Array, numpy as jnp
from flax import nnx
from einops import rearrange


xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0)

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