import jax
import jax.numpy as jnp

def renorm(x, ord=None, axis=None):
    norm = jnp.linalg.norm(x, ord, axis, keepdims=True)
    return x / norm

def renorm_stopgrad(x, ord=None, axis=None):
    norm = jax.lax.stop_gradient(jnp.linalg.norm(x, ord, axis, keepdims=True))
    return x / norm

