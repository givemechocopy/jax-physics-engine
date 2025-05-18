import jax.numpy as jnp
from config import GRAVITY, FRICTION_COEFF, USE_3D

def apply_gravity(velocity, dt):
    g_vector = jnp.array([0.0, -GRAVITY]) if not USE_3D else jnp.array([0.0, -GRAVITY, 0.0])
    return velocity + g_vector * dt


def apply_friction(velocity):
    return velocity * FRICTION_COEFF


def apply_external_force(velocity, force, mass, dt):
    # F = ma → a = F / m
    acceleration = force / mass[:, None]
    return velocity + acceleration * dt
