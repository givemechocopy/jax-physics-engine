import jax.numpy as jnp
import jax.random as random
from config import NUM_BALLS, INIT_POS_RANGE, INIT_VEL_RANGE, DIM, RADIUS, MASS, USE_MASS, USE_ROTATION, USE_3D

def init_state(key):
    subkeys = random.split(key, 4)

    # 위치 및 속도 초기화
    position = random.uniform(subkeys[0], shape=(NUM_BALLS, DIM),
                              minval=INIT_POS_RANGE[0], maxval=INIT_POS_RANGE[1])
    velocity = random.uniform(subkeys[1], shape=(NUM_BALLS, DIM),
                              minval=INIT_VEL_RANGE[0], maxval=INIT_VEL_RANGE[1])

    # 회전 각도 및 속도 (2D only)
    if USE_ROTATION and not USE_3D:
        angle = random.uniform(subkeys[2], shape=(NUM_BALLS,), minval=0, maxval=2*jnp.pi)
        angular_velocity = random.uniform(subkeys[3], shape=(NUM_BALLS,), minval=-1.0, maxval=1.0)
    else:
        angle = jnp.zeros(NUM_BALLS)
        angular_velocity = jnp.zeros(NUM_BALLS)

    return {
        "position": position,
        "velocity": velocity,
        "angle": angle,
        "angular_velocity": angular_velocity,
        "radius": RADIUS,
        "mass": MASS if USE_MASS else jnp.ones(NUM_BALLS)
    }
