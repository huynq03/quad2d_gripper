import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
MASS = 0.600
INERTIA = 0.15
LENGTH = 0.2
GRAVITY = 9.81
DELTA_T = 0.01
NUMBER_STATES = 6
NUMBER_CONTROLS = 2

def get_next_state(z, u):
    x, vx, y, vy, theta, omega = z
    dydt = np.zeros(NUMBER_STATES)
    dydt[0] = vx
    dydt[1] = (-(u[0] + u[1]) * np.sin(theta)) / MASS
    dydt[2] = vy
    dydt[3] = ((u[0] + u[1]) * np.cos(theta) - MASS * GRAVITY) / MASS
    dydt[4] = omega
    dydt[5] = (LENGTH * (u[0] - u[1])) / INERTIA
    z_next = z + dydt * DELTA_T
    return z_next

def simulate(z0, controller, horizon_length, disturbance=False):
    t = np.zeros(horizon_length + 1)
    z = np.empty((NUMBER_STATES, horizon_length + 1))
    u = np.zeros((NUMBER_CONTROLS, horizon_length))
    z[:, 0] = z0
    for i in range(horizon_length):
        u[:, i] = controller(z[:, i], i)
        z[:, i + 1] = get_next_state(z[:, i], u[:, i])
        if disturbance and (i % 100 == 0):
            dist = np.zeros(NUMBER_STATES)
            dist[1::2] = np.random.uniform(-1.0, 1.0, size=3)
            z[:, i + 1] += dist
        t[i + 1] = t[i] + DELTA_T
    return t, z, u

def animate_robot(x, u, dt=0.01):
    min_dt = 0.1
    if dt < min_dt:
        steps = int(min_dt / dt)
        use_dt = int(round(min_dt * 1000))
    else:
        steps = 1
        use_dt = int(round(dt * 1000))

    plotx = x[:, ::steps]
    plotx = plotx[:, :-1]
    plotu = u[:, ::steps]

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_xlim(-10, 10)  # tăng giới hạn ngang để bạn thấy di chuyển
    ax.set_ylim(-8, 8)
    ax.grid()

    lines = []
    for lw in [6, 4, 4, 1, 1]:
        line, = ax.plot([], [], 'k' if lw == 6 else 'b' if lw == 4 else 'r', lw=lw)
        lines.append(line)

    def _animate(i):
        theta = plotx[4, i]
        x_pos = plotx[0, i]
        y_pos = plotx[2, i]
        trans = np.array([[x_pos, x_pos], [y_pos, y_pos]])
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        main = np.array([[-LENGTH, LENGTH], [0, 0]])
        main = rot @ main + trans
        left_prop = np.array([[-1.3 * LENGTH, -0.7 * LENGTH], [0.1, 0.1]])
        left_prop = rot @ left_prop + trans
        right_prop = np.array([[1.3 * LENGTH, 0.7 * LENGTH], [0.1, 0.1]])
        right_prop = rot @ right_prop + trans
        left_th = np.array([[LENGTH, LENGTH], [0.1, 0.1 + plotu[0, i] * 0.04]])
        left_th = rot @ left_th + trans
        right_th = np.array([[-LENGTH, -LENGTH], [0.1, 0.1 + plotu[1, i] * 0.04]])
        right_th = rot @ right_th + trans

        lines[0].set_data(main[0], main[1])
        lines[1].set_data(left_prop[0], left_prop[1])
        lines[2].set_data(right_prop[0], right_prop[1])
        lines[3].set_data(left_th[0], left_th[1])
        lines[4].set_data(right_th[0], right_th[1])

        return lines

    def _init():
        for l in lines:
            l.set_data([], [])
        return lines

    ani = animation.FuncAnimation(fig, _animate, frames=plotx.shape[1],
                                  init_func=_init, interval=use_dt, blit=True)
    return fig, ani

def run_and_show():
    z0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [x, vx, y, vy, theta, omega]
    horizon = 800

    def goto_target(z, i):
        # trạng thái hiện tại
        x, vx, y, vy, theta, omega = z
        m = MASS
        g = GRAVITY

        # ----------- Điều khiển trục dọc (y) bằng thrust tổng -----------
        y_des = 5.0
        vy_des = 0.0
        ky_p, ky_d = 10.0, 6.0
        ey = y_des - y
        evy = vy_des - vy
        # thrust cần để giữ + correction
        u_total = m * g + ky_p * ey + ky_d * evy
        if u_total < 0: u_total = 0.0

        # ----------- Điều khiển trục ngang (x) bằng góc nghiêng -----------
        x_des = 5.0
        vx_des = 0.0
        kx_p, kx_d = 1.5, 2.0
        ex = x_des - x
        evx = vx_des - vx
        # lệnh góc nghiêng mục tiêu (theta_c): để sinh lực ngang
        theta_c = kx_p * ex + kx_d * evx
        max_theta = np.radians(15)
        theta_c = np.clip(theta_c, -max_theta, max_theta)

        # ----------- PD để θ → θ_c -----------
        k_th_p, k_th_d = 80.0, 15.0
        u_diff = k_th_p * (theta_c - theta) - k_th_d * omega

        # ----------- Tính thrust từng cánh -----------
        u1 = u_total/2 + u_diff/2
        u2 = u_total/2 - u_diff/2
        u1 = max(0.0, u1)
        u2 = max(0.0, u2)

        return np.array([u1, u2])

    t, x, u = simulate(z0, goto_target, horizon)
    fig, ani = animate_robot(x, u)
    plt.show()


if __name__ == "__main__":
    run_and_show()
