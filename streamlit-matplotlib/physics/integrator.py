import jax.numpy as jnp
from physics.forces import apply_gravity, apply_friction
from physics.collision import detect_and_resolve_collisions
from physics.constraints import handle_wall_collision
from config import DT


def integrate(state):
    pos = state["position"]
    vel = state["velocity"]
    angle = state["angle"]
    ang_vel = state["angular_velocity"]
    radius = state["radius"]
    mass = state["mass"]

    # 물리 적용
    vel = apply_gravity(vel, DT)
    vel = apply_friction(vel)
    vel = detect_and_resolve_collisions(pos, vel, mass, radius)
    vel, ang_vel = handle_wall_collision(pos, vel, angle, ang_vel, radius)

    # 위치/회전 업데이트
    pos = pos + vel * DT
    angle = angle + ang_vel * DT

    # 업데이트된 상태 반환
    return {
        "position": pos,
        "velocity": vel,
        "angle": angle,
        "angular_velocity": ang_vel,
        "radius": radius,
        "mass": mass
    }
