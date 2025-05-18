import sys, os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from config import RADIUS, BOUNDS
from simulate import run_simulation
import jax.random as random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def streamlit_animation(trajectory):
    st.title("💫 JAX Rigid Body Simulation (2D)")

    num_frames = trajectory.shape[0]
    num_balls = trajectory.shape[1]
    radius = np.array(RADIUS)

    frame = st.slider("📽 프레임 선택", 0, num_frames - 1, 0, step=1)

    fig, ax = plt.subplots()
    ax.set_xlim(BOUNDS[0], BOUNDS[1])
    ax.set_ylim(BOUNDS[2], BOUNDS[3])
    ax.set_aspect("equal")
    ax.set_title(f"Frame {frame}")

    for i in range(num_balls):
        x, y = trajectory[frame, i]
        circle = plt.Circle((x, y), radius[i], color='royalblue')
        ax.add_patch(circle)

    st.pyplot(fig)

# 🎯 실행 진입점
if __name__ == "__main__":
    key = random.PRNGKey(0)
    trajectory = run_simulation(key)
    streamlit_animation(trajectory)
