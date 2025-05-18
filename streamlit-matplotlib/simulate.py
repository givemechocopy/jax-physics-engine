import jax.numpy as jnp
from state.initializer import init_state
from physics.integrator import integrate
from config import STEPS
import jax

# JIT 제거: 내부 조건문 및 for 루프가 Tracer를 유발하므로 jit 사용 불가
# @jax.jit

def simulate_step(state, _):
    next_state = integrate(state)
    return next_state, next_state["position"]

def run_simulation(key):
    state = init_state(key)
    _, trajectory = jax.lax.scan(simulate_step, state, None, length=STEPS)
    return trajectory  # shape: (STEPS, NUM_BALLS, DIM)