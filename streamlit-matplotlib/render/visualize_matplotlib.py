import matplotlib.pyplot as plt
from matplotlib import animation
from config import BOUNDS, RADIUS, FPS
import numpy as np

def animate_simulation(trajectory):
    num_balls = trajectory.shape[1]
    radius = np.array(RADIUS)

    fig, ax = plt.subplots()
    ax.set_xlim(BOUNDS[0], BOUNDS[1])
    ax.set_ylim(BOUNDS[2], BOUNDS[3])
    ax.set_aspect('equal')

    balls = [plt.Circle((0, 0), radius[i], color='skyblue') for i in range(num_balls)]
    for ball in balls:
        ax.add_patch(ball)

    def init():
        for ball in balls:
            ball.center = (0, 0)
        return balls

    def update(frame):
        for i, ball in enumerate(balls):
            x, y = trajectory[frame, i]
            ball.center = (x, y)
        return balls

    ani = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=len(trajectory),
                                  interval=1000 / FPS,
                                  blit=True)
    plt.show()
