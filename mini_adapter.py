import jax
from jax import Array, numpy as jnp
from flax import nnx

rngs = nnx.Rngs(0)

class AdapterProjection(nnx.Module):
    def __init__(self, dim, scale=4):
        super().__init__()
        self.fc_linear = nnx.Sequential(
            nnx.Linear(dim, dim // scale, use_bias=False, rngs=rngs),
            nnx.relu, 
            nnx.Linear(dim // scale, dim, use_bias=False, rngs=rngs),
            nnx.relu
        )
        
    def __call__(self, x):
        x = self.fc_linear(x)
        
        return x


