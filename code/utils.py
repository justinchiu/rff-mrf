import jax
import jax.numpy as jnp

def renorm(x, max_value, ord=None, axis=None):
    norm = jnp.linalg.norm(x, ord, axis, keepdims=True)
    return x / norm

