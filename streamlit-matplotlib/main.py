# main.py

import argparse
import jax.random as random
from simulate import run_simulation
from utils.profiler import profile

# 선택적 임포트 (지연 로딩 구조)
def import_gui(gui_type):
    if gui_type == "gui":
        from render.visualize_matplotlib import animate_simulation
        return animate_simulation
    elif gui_type == "web":
        from render.web_gui import streamlit_animation
        return streamlit_animation
    else:
        raise ValueError("지원하지 않는 시각화 모드입니다: 'gui' 또는 'web' 중 선택하세요.")


@profile
def main():
    parser = argparse.ArgumentParser(description="JAX Rigid Body Simulator")
    parser.add_argument("--mode", choices=["gui", "web"], default="gui", help="시각화 모드: 'gui' 또는 'web'")
    args = parser.parse_args()

    key = random.PRNGKey(42)
    trajectory = run_simulation(key)

    render_fn = import_gui(args.mode)
    render_fn(trajectory)


if __name__ == "__main__":
    main()
