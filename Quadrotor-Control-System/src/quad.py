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
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
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
    z0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    horizon = 800
    x_d = 2.0
    y_d = 1.0

    # Khởi tạo biến tích phân (nếu bạn muốn dùng I-term)
    integral_x = 0.0
    integral_y = 0.0

    def goto_target_fast_stable(target=(1.0, 2.0)):
        # trạng thái tích phân (giữ bên ngoài hàm con)
        int_x = {"v": 0.0}
        int_y = {"v": 0.0}
        last_theta_c = {"v": 0.0}

        # tham số
        kx_p, kx_d, kx_i = 2.8, 1.2, 0.6     # ngang: mạnh hơn để rút thời gian bám
        ky_p, ky_d, ky_i = 12.0, 7.0, 3.5    # dọc: mạnh & nhiều damping để ít vượt đỉnh
        aw_limit = 0.6                       # anti-windup clamp (tích phân tối đa)
        theta_max = np.radians(18)           # giới hạn góc nghiêng
        theta_rate_max = np.radians(120)     # giới hạn tốc độ thay đổi góc mục tiêu (deg/s)
        k_th_p, k_th_d = 110.0, 18.0         # vòng attitude nhanh hơn nhiều

        def controller(z, i):
            x, vx, y, vy, theta, omega = z
            xd, yd = target
            m, g = MASS, GRAVITY

            # --- outer loop: vị trí -> gia tốc mong muốn ---
            ex, evx = xd - x, -vx
            ey, evy = yd - y, -vy

            # anti-windup cho tích phân (clamp)
            int_x["v"] = np.clip(int_x["v"] + ex * DELTA_T, -aw_limit, aw_limit)
            int_y["v"] = np.clip(int_y["v"] + ey * DELTA_T, -aw_limit, aw_limit)

            ax_des = kx_p*ex + kx_d*evx + kx_i*int_x["v"]
            ay_des = ky_p*ey + ky_d*evy + ky_i*int_y["v"]

            # --- thrust tổng (bù trọng lực) ---
            u_total = m * (g + ay_des)
            u_total = max(0.0, u_total)

            # --- map ax_des -> theta_c (ổn định & chính xác hơn small-angle) ---
            theta_c = -np.arctan2(m * ax_des, u_total + 1e-6)
            theta_c = np.clip(theta_c, -theta_max, theta_max)

            # --- hạn tốc độ thay đổi theta_c để tránh jerk ---
            theta_c_rate = np.clip((theta_c - last_theta_c["v"]) / DELTA_T,
                                -theta_rate_max, theta_rate_max)
            theta_c = last_theta_c["v"] + theta_c_rate * DELTA_T
            last_theta_c["v"] = theta_c

            # --- inner loop: attitude PD nhanh ---
            u_diff = k_th_p * (theta_c - theta) - k_th_d * omega

            u1 = u_total/2 + u_diff/2
            u2 = u_total/2 - u_diff/2
            return np.array([max(0.0, u1), max(0.0, u2)])

        return controller
    controller = goto_target_fast_stable(target=(1.0, 2.0))
    t, x, u = simulate(z0, controller, horizon)

    # t, x, u = simulate(z0, goto_point_controller, horizon)
    fig, ani = animate_robot(x, u)
    plt.show()



if __name__ == "__main__":
    run_and_show()
