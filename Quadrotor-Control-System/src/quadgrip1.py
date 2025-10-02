import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# =========================
#  PHYSICS & MODEL PARAMS
# =========================
MASS_BASE    = 0.600
INERTIA_BASE = 0.15
LENGTH       = 0.2
GRAVITY      = 9.81
DELTA_T      = 0.01

GRIPPER_MASS = MASS_BASE * 0.25
L_GRIP       = 0.35

MASS_TOTAL = MASS_BASE + GRIPPER_MASS
INERTIA    = INERTIA_BASE

NUMBER_STATES   = 8
NUMBER_CONTROLS = 2

BETA_MAX = jnp.radians(60.0)
K_STOP   = 50.0
D_STOP   = 5.0

# =========================
#  DYNAMICS (JAX)
# =========================
@jax.jit
def get_next_state(z, u):
    x, vx, y, vy, theta, omega, beta, betadot = z

    T  = u[0] + u[1]
    ax = -(T * jnp.sin(theta)) / MASS_TOTAL
    ay =  (T * jnp.cos(theta) - MASS_TOTAL * GRAVITY) / MASS_TOTAL

    tau = LENGTH * (u[0] - u[1])
    omegadot = tau / INERTIA

    betaddot = -(GRAVITY / L_GRIP) * jnp.sin(beta) \
               - (ax / L_GRIP) * jnp.cos(beta) \
               - (ay / L_GRIP) * jnp.sin(beta)

    # bump-stop
    betaddot = jnp.where(beta > BETA_MAX,
                         betaddot + -K_STOP * (beta - BETA_MAX) - D_STOP * betadot,
                         betaddot)
    betaddot = jnp.where(beta < -BETA_MAX,
                         betaddot + -K_STOP * (beta + BETA_MAX) - D_STOP * betadot,
                         betaddot)

    dz = jnp.array([
        vx,
        ax,
        vy,
        ay,
        omega,
        omegadot,
        betadot,
        betaddot
    ])

    return z + dz * DELTA_T

# =========================
#  SIMULATE LOOP
# =========================
def simulate(z0, controller, horizon_length, disturbance=False):
    t = np.zeros(horizon_length + 1)
    z = np.empty((NUMBER_STATES, horizon_length + 1))
    u = np.zeros((NUMBER_CONTROLS, horizon_length))
    z[:, 0] = np.array(z0)

    for i in range(horizon_length):
        u[:, i] = np.array(controller(z[:, i], i))  # controller trả numpy
        z[:, i + 1] = np.array(get_next_state(z[:, i], u[:, i]))  # dynamics JAX

        if disturbance and (i % 200 == 0) and i > 0:
            z[1, i + 1] += np.random.uniform(-0.3, 0.3)
            z[3, i + 1] += np.random.uniform(-0.3, 0.3)
            z[7, i + 1] += np.random.uniform(-0.5, 0.5)

        t[i + 1] = t[i] + DELTA_T

    return t, z, u

# =========================
#  CONTROLLER (giữ numpy)
# =========================
def goto_target_fast_stable(target=(5.0, 5.0)):
    int_x = {"v": 0.0}
    int_y = {"v": 0.0}
    last_theta_c = {"v": 0.0}

    kx_p, kx_d, kx_i = 2.6, 1.2, 0.5
    ky_p, ky_d, ky_i = 11.0, 6.5, 3.0
    aw_limit = 0.6

    theta_max      = np.radians(18)
    theta_rate_max = np.radians(120)
    k_th_p, k_th_d = 110.0, 18.0

    def controller(z, i):
        x, vx, y, vy, theta, omega, beta, betadot = z
        xd, yd = target

        ex, evx = xd - x, -vx
        ey, evy = yd - y, -vy

        int_x["v"] = np.clip(int_x["v"] + ex * DELTA_T, -aw_limit, aw_limit)
        int_y["v"] = np.clip(int_y["v"] + ey * DELTA_T, -aw_limit, aw_limit)

        ax_des = kx_p*ex + kx_d*evx + kx_i*int_x["v"]
        ay_des = ky_p*ey + ky_d*evy + ky_i*int_y["v"]

        u_total = MASS_TOTAL * (GRAVITY + ay_des)
        u_total = max(0.0, u_total)

        theta_c = -np.arctan2(MASS_TOTAL * ax_des, u_total + 1e-6)
        theta_c = np.clip(theta_c, -theta_max, theta_max)

        theta_c_rate = np.clip((theta_c - last_theta_c["v"]) / DELTA_T,
                               -theta_rate_max, theta_rate_max)
        theta_c = last_theta_c["v"] + theta_c_rate * DELTA_T
        last_theta_c["v"] = theta_c

        u_diff = k_th_p * (theta_c - theta) - k_th_d * omega

        u1 = u_total/2 + u_diff/2
        u2 = u_total/2 - u_diff/2
        return np.array([max(0.0, u1), max(0.0, u2)])

    return controller

# =========================
#  VISUALIZATION (giữ nguyên numpy)
# =========================
def animate_robot(x, u, target=None, dt=DELTA_T):
    min_dt = 0.1
    if dt < min_dt:
        steps = max(1, int(min_dt / dt))
        use_dt = int(round(min_dt * 1000))
    else:
        steps = 1
        use_dt = int(round(dt * 1000))

    plotx = x[:, ::steps]
    plotx = plotx[:, :-1]
    plotu = u[:, ::steps]
    xs = plotx[0, :]
    ys = plotx[2, :]

    pad = 1.0
    if target is None:
        xmin, xmax = xs.min() - pad, xs.max() + pad
        ymin, ymax = ys.min() - pad, ys.max() + pad
    else:
        xmin = min(xs.min(), target[0]) - pad
        xmax = max(xs.max(), target[0]) + pad
        ymin = min(ys.min(), target[1]) - pad
        ymax = max(ys.max(), target[1]) + pad

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Quad + Pendulum Gripper (limit ±{np.degrees(BETA_MAX):.0f}°)")

    frame_line, = ax.plot([], [], 'k', lw=6)
    lprop_line, = ax.plot([], [], 'b', lw=4)
    rprop_line, = ax.plot([], [], 'b', lw=4)
    lth_line,   = ax.plot([], [], 'r', lw=1)
    rth_line,   = ax.plot([], [], 'r', lw=1)
    trail_line, = ax.plot([], [], '-', lw=1, alpha=0.6)
    grip_line,  = ax.plot([], [], color='gray', lw=3)

    if target is not None:
        tgt_patch = Circle((target[0], target[1]), 0.06, fc='none', ec='g', lw=2)
        ax.add_patch(tgt_patch)

    artists = [frame_line, lprop_line, rprop_line, lth_line, rth_line, trail_line, grip_line]

    def _init():
        for l in artists:
            if hasattr(l, "set_data"):
                l.set_data([], [])
        return artists

    def _animate(i):
        theta = plotx[4, i]
        px = plotx[0, i]
        py = plotx[2, i]
        beta = plotx[6, i]

        trans = np.array([[px, px], [py, py]])
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])

        # Body
        main = np.array([[-LENGTH, LENGTH], [0, 0]])
        main = rot @ main + trans
        frame_line.set_data(main[0], main[1])

        # Props
        lprop = rot @ np.array([[-1.3*LENGTH, -0.7*LENGTH], [0.1, 0.1]]) + trans
        rprop = rot @ np.array([[ 1.3*LENGTH,  0.7*LENGTH], [0.1, 0.1]]) + trans
        lprop_line.set_data(lprop[0], lprop[1])
        rprop_line.set_data(rprop[0], rprop[1])

        # Thrust
        scale = 0.04
        lth = rot @ np.array([[ LENGTH,  LENGTH], [0.1, 0.1 + plotu[0, i]*scale]]) + trans
        rth = rot @ np.array([[-LENGTH, -LENGTH], [0.1, 0.1 + plotu[1, i]*scale]]) + trans
        lth_line.set_data(lth[0], lth[1])
        rth_line.set_data(rth[0], rth[1])

        # Pendulum
        top_local = np.array([[0.0], [0.0]])
        bottom_local = np.array([[ L_GRIP*np.sin(beta)],
                                 [-L_GRIP*np.cos(beta)]])
        grip_local = np.hstack([top_local, bottom_local])
        grip_world = rot @ grip_local + trans
        grip_line.set_data(grip_world[0], grip_world[1])

        if i > 0:
            trail_line.set_data(xs[:i+1], ys[:i+1])

        return artists

    ani = animation.FuncAnimation(fig, _animate,
                                  frames=plotx.shape[1],
                                  init_func=_init,
                                  interval=use_dt,
                                  blit=True)
    return fig, ani

# =========================
#  RUN DEMO
# =========================
def run_and_show():
    z0 = jnp.array([0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0,
                    jnp.radians(50.0), 0.0])
    horizon = 1500
    target = (5.0, 5.0)

    controller = goto_target_fast_stable(target=target)
    t, x, u = simulate(z0, controller, horizon)

    # animate_robot giữ như cũ
    fig, ani = animate_robot(x, u, target=target, dt=DELTA_T)
    plt.show()

if __name__ == "__main__":
    run_and_show()
