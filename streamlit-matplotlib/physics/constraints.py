import jax.numpy as jnp
from config import BOUNDS, RESTITUTION, USE_ROTATION, USE_3D


def handle_wall_collision(position, velocity, angle, angular_velocity, radius):
    x_min, x_max, y_min, y_max = BOUNDS

    # X축 충돌
    hit_left = position[:, 0] - radius < x_min
    hit_right = position[:, 0] + radius > x_max

    vx = jnp.where(hit_left | hit_right, -velocity[:, 0] * RESTITUTION, velocity[:, 0])

    # Y축 충돌
    hit_bottom = position[:, 1] - radius < y_min
    hit_top = position[:, 1] + radius > y_max

    vy = jnp.where(hit_bottom | hit_top, -velocity[:, 1] * RESTITUTION, velocity[:, 1])

    new_velocity = jnp.stack([vx, vy], axis=1)

    # 3D 무시, 회전 감쇠
    if USE_ROTATION and not USE_3D:
        ang_v = jnp.where(hit_left | hit_right | hit_bottom | hit_top,
                          -angular_velocity * RESTITUTION,
                          angular_velocity)
    else:
        ang_v = angular_velocity

    return new_velocity, ang_v
