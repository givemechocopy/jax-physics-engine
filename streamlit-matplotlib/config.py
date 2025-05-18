import jax.numpy as jnp

# 시뮬레이션 설정
NUM_BALLS = 10
DT = 0.01
STEPS = 1000
DIM = 2  # 2D or 3D 지원 가능

# 공간 경계 (x_min, x_max, y_min, y_max)
BOUNDS = (0.0, 1.0, 0.0, 1.0)

# 초기화 범위
INIT_POS_RANGE = (0.2, 0.8)
INIT_VEL_RANGE = (-0.2, 0.2)

# 물리 상수
GRAVITY = 9.8
FRICTION_COEFF = 0.98
RESTITUTION = 0.8

# 물리 옵션
USE_MASS = True
USE_ROTATION = True
USE_TORQUE = True
USE_3D = False

# 시각화
FPS = 60
RADIUS = jnp.ones(NUM_BALLS) * 0.03
MASS = jnp.ones(NUM_BALLS) * 1.0  # kg
