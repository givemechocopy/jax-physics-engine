import jax.numpy as jnp

def normalize(v, eps=1e-6):
    norm = jnp.linalg.norm(v)
    return v / (norm + eps)

def clamp(x, min_val, max_val):
    return jnp.minimum(jnp.maximum(x, min_val), max_val)
