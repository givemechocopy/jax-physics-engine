# backend/simulator.py

import jax.numpy as jnp
import jax.random as random

def run_simulation(num_balls=5, steps=500, dt=0.01):
    key = random.PRNGKey(0)
    subkey1, subkey2 = random.split(key)

    pos = random.uniform(subkey1, shape=(num_balls, 2), minval=0.2, maxval=0.8)
    vel = random.uniform(subkey2, shape=(num_balls, 2), minval=-0.1, maxval=0.1)

    trajectory = []

    for _ in range(steps):
        vel += jnp.array([0.0, -9.8]) * dt
        pos += vel * dt

        # 충돌 처리
        vel = jnp.where(pos < 0.0, -vel * 0.9, vel)
        vel = jnp.where(pos > 1.0, -vel * 0.9, vel)
        pos = jnp.clip(pos, 0.0, 1.0)

        trajectory.append(pos)

    return jnp.stack(trajectory).tolist()
