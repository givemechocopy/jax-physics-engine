import jax.numpy as jnp
import jax
from config import RESTITUTION, USE_MASS

def resolve_collision(i, j, position, velocity, mass, radius):
    delta = position[i] - position[j]
    dist = jnp.linalg.norm(delta)
    min_dist = radius[i] + radius[j]

    def do_resolve(_):
        normal = delta / (dist + 1e-6)
        rel_vel = velocity[i] - velocity[j]
        vel_along_normal = jnp.dot(rel_vel, normal)

        def skip(_):
            return velocity[i], velocity[j]

        def resolve(_):
            if USE_MASS:
                mi, mj = mass[i], mass[j]
                impulse = -(1 + RESTITUTION) * vel_along_normal
                impulse /= (1 / mi + 1 / mj)
            else:
                impulse = -(1 + RESTITUTION) * vel_along_normal / 2

            impulse_vec = impulse * normal
            vi = velocity[i] + impulse_vec / mass[i]
            vj = velocity[j] - impulse_vec / mass[j]
            return vi, vj

        return jax.lax.cond(vel_along_normal > 0, skip, resolve, operand=None)

    def no_resolve(_):
        return velocity[i], velocity[j]

    return jax.lax.cond(dist < min_dist, do_resolve, no_resolve, operand=None)

def detect_and_resolve_collisions(position, velocity, mass, radius):
    num = position.shape[0]
    for i in range(num):
        for j in range(i + 1, num):
            vi_new, vj_new = resolve_collision(i, j, position, velocity, mass, radius)
            velocity = velocity.at[i].set(vi_new)
            velocity = velocity.at[j].set(vj_new)
    return velocity