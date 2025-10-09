import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================
# Parameters
# =============================
dt   = 0.02
T    = 10.0
m_q  = 0.6
m_p  = 0.15
g    = 9.81
I_xx = 0.15
l_p  = 0.35
l_q  = 0.2   # for visualization
J_g  = 0.0
J_q  = 0.15
L_g  = 0.35
m_g  = 0.15

# Moment quán tính con lắc quanh khớp (point mass ở cuối thanh)
I_p  = m_p * (l_p**2)

# Giới hạn điều khiển
U1_MIN, U1_MAX = 0.0, 30.0     # thrust (N)
U2_MIN, U2_MAX = -15.0, 15.0   # torque thân (N·m)
U3_MIN, U3_MAX = -8.0, 8.0     # torque pendulum (N·m)  <-- MỚI

# =============================
# Dynamics (JAX)
# state = [y, y_dot, z, z_dot, phi, phi_dot, theta, theta_dot]
# control = [u1 (thrust), u2 (torque_body), u3 (torque_pendulum)]
# =============================
def jax_dynamics_matrix(state, control, dt=dt):
    """
    state  = [y, y_dot, z, z_dot, phi, phi_dot, beta, beta_dot]
    control= [u1, u2, tau]
    """
    y, y_dot, z, z_dot, phi, phi_dot, beta, beta_dot = state
    u1, u2, tau = control

    M = m_q + m_g
    s = jnp.sin(beta)
    c = jnp.cos(beta)

    # D, C, G giống như trước
    D = jnp.array([
        [M,        0.0, 0.0,     -L_g*m_g*s],
        [0.0,      M,   0.0,     -L_g*m_g*c],
        [0.0,      0.0, J_q,      0.0     ],
        [-L_g*m_g*s, -L_g*m_g*c, 0.0, J_g + L_g**2 * m_g]
    ], dtype=state.dtype)

    C = jnp.array([
        [0.0, 0.0, 0.0, -L_g*m_g*c * beta_dot],
        [0.0, 0.0, 0.0,  L_g*m_g*s * beta_dot],
        [0.0, 0.0, 0.0,   0.0],
        [0.0, 0.0, 0.0,   0.0]
    ], dtype=state.dtype)

    G = jnp.array([
        0.0,
        g * M,
        0.0,
        -g * L_g * m_g * c
    ], dtype=state.dtype)

    F = jnp.array([
        u1 * jnp.sin(phi),
        u1 * jnp.cos(phi),
        u2 - tau,
        tau
    ], dtype=state.dtype)

    qdot = jnp.array([y_dot, z_dot, phi_dot, beta_dot], dtype=state.dtype)
    rhs = F - C @ qdot - G

    # Chuyển sang numpy để giải tuyến tính 4×4
    D_np = np.array(D)
    rhs_np = np.array(rhs)
    qddot_np = np.linalg.solve(D_np, rhs_np) # giai phương tuyến tính Ax = b => x = A^{-1}b , x la qddot
    # Chuyển trở lại jax
    qddot = jnp.array(qddot_np, dtype=state.dtype)

    y_ddot, z_ddot, phi_ddot, beta_ddot = qddot

    state_dot = jnp.array([
        y_dot, y_ddot,
        z_dot, z_ddot,
        phi_dot, phi_ddot,
        beta_dot, beta_ddot
    ], dtype=state.dtype)

    return state + state_dot * dt
# @jax.jit
# def jax_dynamics(state, control, dt=dt):
#     y, y_dot, z, z_dot, phi, phi_dot, theta, theta_dot = state
#     u1, u2, u3 = control

#     M = m_q + m_p

#     # Coupled translational accelerations (giữ như trước)
#     y_ddot = (m_q + m_p*jnp.cos(theta)**2) / (m_q*M) * u1 * jnp.sin(phi) \
#              + (m_p*jnp.cos(theta)*jnp.sin(theta)) / (m_q*M) * u1 * jnp.cos(phi) \
#              + (m_p*l_p*(theta_dot**2)*jnp.sin(theta)) / M

#     z_ddot = -g + (m_q + m_p*jnp.sin(theta)**2) / (m_q*M) * u1 * jnp.cos(phi) \
#              + (m_p*jnp.cos(theta)*jnp.sin(theta)) / (m_q*M) * u1 * jnp.sin(phi) \
#              + (m_p*l_p*(theta_dot**2)*jnp.cos(theta)) / M

#     # Attitude & pendulum with actuated joint:
#     # - u2 là torque lên thân do rotor
#     # - u3 là torque tác dụng lên pendulum; phản lực lên thân là -u3
#     phi_ddot   = (u2 - u3) / I_xx
#     theta_ddot = -jnp.cos(theta) / (m_q*l_p) * u1 * jnp.sin(phi) \
#                  - jnp.sin(theta) / (m_q*l_p) * u1 * jnp.cos(phi) \
#                  + (u3 / I_p)

#     next_state = state + jnp.array([
#         y_dot, y_ddot, z_dot, z_ddot, phi_dot, phi_ddot, theta_dot, theta_ddot
#     ]) * dt
#     return next_state

# =============================
# PID controller
# =============================
class PID:
    def __init__(self, Kp, Ki, Kd, out_min=None, out_max=None):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.i_term = 0.0
        self.prev_e = 0.0
        self.out_min, self.out_max = out_min, out_max

    def __call__(self, ref, meas, dt):
        e = ref - meas
        self.i_term += e * dt
        d = (e - self.prev_e) / dt if dt > 1e-6 else 0.0
        self.prev_e = e
        u = self.Kp*e + self.Ki*self.i_term + self.Kd*d
        if self.out_min is not None and self.out_max is not None:
            u = np.clip(u, self.out_min, self.out_max)
        return float(u)

# =============================
# Cascade + actuated pendulum PID
# =============================
class CascadePIDActuatedPendulum:
    def __init__(self, target=(5.0, 5.0)):
        self.y_ref, self.z_ref = target

        # z -> thrust
        self.pid_z   = PID(12.0, 1.5, 7.0,  out_min=-6.0, out_max=6.0)

        # y -> phi_ref (giới hạn vừa phải để tránh nghiêng quá)
        self.pid_y   = PID(1.0,  0.0, 0.35, out_min=-np.deg2rad(20), out_max=np.deg2rad(20))

        # phi -> u2 (torque thân)
        self.pid_phi = PID(65.0, 1.0, 18.0, out_min=U2_MIN, out_max=U2_MAX)
        # bắt đầu với gain vừa phải để tránh kích thích dao động
        self.pid_theta = PID(2.0, 0.0, 0.6, out_min=U3_MIN, out_max=U3_MAX)
        self.t = 0.0

    def step(self, state):
        y, ydot, z, zdot, phi, phidot, theta, thetad = state

        # z → u1 (thrust tổng)
        az_cmd = self.pid_z(self.z_ref, z, dt)      # m/s^2
        u1 = (m_q + m_p) * (g + az_cmd)
        u1 = np.clip(u1, U1_MIN, U1_MAX)

        # y → phi_ref
        phi_ref = self.pid_y(self.y_ref, y, dt)

        # phi → u2 (torque thân)
        u2 = self.pid_phi(phi_ref, phi, dt)

        # theta → u3 (torque khớp pendulum)
        u3 = self.pid_theta(0.2, theta, dt) + np.sin(self.t)*2 # thêm nhiễu nhỏ để kích thích con lắc
        self.t += dt
        return u1, u2, u3

# =============================
# Simulation
# =============================
def simulate(controller, T=T, dt=dt):
    N = int(T/dt)
    state = np.array([0.0, 0.0, 0.5, 0.0,
                      0.0, 0.0,
                      np.deg2rad(30.0), 0.0])
    states = [state]
    controls = []

    for _ in range(N):
        u1, u2, u3 = controller.step(state)
        u = jnp.array([u1, u2, u3], dtype=jnp.float32)
        # state = np.array(jax_dynamics(state, u))
        state = np.array(jax_dynamics_matrix(state, u))
        states.append(state)
        controls.append([u1, u2, u3])

    return np.array(states), np.array(controls)

# =============================
# Visualization
# =============================
def animate(states, controls, target=(5.0,5.0), dt=dt):
    y = states[:,0]; z = states[:,2]; phi = states[:,4]; theta = states[:,6]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(min(y.min(), target[0])-1, max(y.max(), target[0])+1)
    ax.set_ylim(min(z.min(), target[1])-1, max(z.max(), target[1])+1)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.add_patch(plt.Circle(target, 0.1, color="g", fill=False))

    frame_line, = ax.plot([], [], "k", lw=5)
    tether_line, = ax.plot([], [], "gray", lw=2)
    trail, = ax.plot([], [], "b-", lw=1, alpha=0.6)

    def update(i):
        yc, zc, phic, thetac = y[i], z[i], phi[i], theta[i]
        R = np.array([[np.cos(phic), -np.sin(phic)],
                      [np.sin(phic),  np.cos(phic)]])
        main = np.array([[-l_q, l_q],[0.0,0.0]])
        body = R @ main + np.array([[yc,yc],[zc,zc]])
        frame_line.set_data(body[0], body[1])

        pend = np.array([[0, l_p*np.sin(thetac)],
                         [0,-l_p*np.cos(thetac)]])
        pend_w = R @ pend + np.array([[yc,yc],[zc,zc]])
        tether_line.set_data(pend_w[0], pend_w[1])

        trail.set_data(y[:i+1], z[:i+1])
        return frame_line, tether_line, trail

    ani = FuncAnimation(fig, update, frames=len(states), interval=dt*1000, blit=True)
    plt.show()

# =============================
# Main
# =============================
def main():
    controller = CascadePIDActuatedPendulum(target=(5.0, 5.0))
    states, controls = simulate(controller)
    animate(states, controls, target=(5.0,5.0))

if __name__ == "__main__":
    main()
