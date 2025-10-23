import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==== Parameters ====
dt, T = 0.02, 10.0
m_q, m_g, g = 0.5, 0.158, 9.81
I_xx, l_p, l_q = 0.15, 0.35, 0.2
J_q, J_g, L_g = 0.15, 0.0, 0.35
# u1: Tổng lực nâng (thrust) tác động lên hệ (từ quadrotor), điều khiển chuyển động lên/xuống (z).
# u2: Mô men quay (torque) quanh trục chính của quadrotor, điều khiển góc nghiêng phi (phi).
# u3: Mô men quay tác động lên con lắc (pendulum), điều khiển góc lệch beta (beta).

U1_MIN, U1_MAX = 0.0, 30.0
U2_MIN, U2_MAX = -15.0, 15.0
U3_MIN, U3_MAX = -8.0, 8.0

# ==== Dynamics ====
def jax_dynamics_matrix(state, control, dt=dt):
    y, y_dot, z, z_dot, phi, phi_dot, beta, beta_dot = state
    u1, u2, tau = control
    M = m_q + m_g
    s, c = jnp.sin(beta), jnp.cos(beta)
    D = jnp.array([
        [M, 0, 0, -L_g*m_g*s],
        [0, M, 0, -L_g*m_g*c],
        [0, 0, J_q, 0],
        [-L_g*m_g*s, -L_g*m_g*c, 0, J_g + L_g**2 * m_g]
    ], dtype=state.dtype)
    C = jnp.array([
        [0, 0, 0, -L_g*m_g*c * beta_dot],
        [0, 0, 0,  L_g*m_g*s * beta_dot],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=state.dtype)
    G = jnp.array([0, g*M, 0, -g*L_g*m_g*c], dtype=state.dtype)
    F = jnp.array([u1*jnp.sin(phi), u1*jnp.cos(phi), u2-tau, tau], dtype=state.dtype)
    qdot = jnp.array([y_dot, z_dot, phi_dot, beta_dot], dtype=state.dtype)
    rhs = F - C @ qdot - G
    qddot = jnp.array(np.linalg.solve(np.array(D), np.array(rhs)), dtype=state.dtype) # D @ qddot = rhs => qddot = D^-1 @ rhs
    y_ddot, z_ddot, phi_ddot, beta_ddot = qddot
    state_dot = jnp.array([y_dot, y_ddot, z_dot, z_ddot, phi_dot, phi_ddot, beta_dot, beta_ddot], dtype=state.dtype)
    return state + state_dot * dt

# ==== PID ====
class PID:
    def __init__(self, Kp, Ki, Kd, out_min=None, out_max=None):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.i_term, self.prev_e = 0.0, 0.0
        self.out_min, self.out_max = out_min, out_max
    def __call__(self, ref, meas, dt):
        e = ref - meas
        self.i_term += e * dt  # tich phan e*dt
        d = (e - self.prev_e) / dt if dt > 1e-6 else 0.0 # dao ham de/dt
        self.prev_e = e # luu e hien tai
        u = self.Kp*e + self.Ki*self.i_term + self.Kd*d
        if self.out_min is not None and self.out_max is not None:
            u = np.clip(u, self.out_min, self.out_max)  # gioi han u
        return float(u)

# ==== Controller ====
class CascadePIDActuatedPendulum:
    def __init__(self, target=(5.0, 5.0)):
        self.y_ref, self.z_ref = target
        self.pid_z   = PID(12.0, 1.5, 7.0,  out_min=-6.0, out_max=6.0)
        self.pid_y   = PID(1.0,  0.0, 0.35, out_min=-np.deg2rad(20), out_max=np.deg2rad(20))
        self.pid_phi = PID(65.0, 1.0, 18.0, out_min=U2_MIN, out_max=U2_MAX)
        self.pid_theta = PID(2.0, 0.0, 0.6, out_min=U3_MIN, out_max=U3_MAX)
        self.t = 0.0
    def step(self, state):
        y, ydot, z, zdot, phi, phidot, theta, thetad = state
        az_cmd = self.pid_z(self.z_ref, z, dt) # z_ref la vi tri mong muon cua z
        u1 = (m_q + m_g) * (g + az_cmd)
        u1 = np.clip(u1, U1_MIN, U1_MAX)
        phi_ref = self.pid_y(self.y_ref, y, dt)
        u2 = self.pid_phi(phi_ref, phi, dt)
        u3 = self.pid_theta(0.2, theta, dt) + np.sin(self.t)*0.5
        u3 = np.clip(u3, U3_MIN, U3_MAX)
        self.t += dt
        return u1, u2, u3

# ==== Simulation ====
def simulate(controller, T=T, dt=dt):
    N = int(T/dt)
    state = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, np.deg2rad(-30.0), 0.0]) 
    # y, y_dot, z, z_dot, phi, phi_dot, beta, beta_dot
    states, controls = [state], []
    for _ in range(N):
        u1, u2, u3 = controller.step(state) # cap nhat dieu khien theo trang thai hien tai
        u = jnp.array([u1, u2, u3], dtype=jnp.float32)
        state = np.array(jax_dynamics_matrix(state, u))
        states.append(state)
        controls.append([u1, u2, u3])
    return np.array(states), np.array(controls)

# ==== Visualization ====
def animate(states, controls, target=(5.0,5.0), dt=dt):
    y, z, phi, theta = states[:,0], states[:,2], states[:,4], states[:,6]

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xlim(min(y.min(), target[0])-1, max(y.max(), target[0])+1)
    ax.set_ylim(min(z.min(), target[1])-1, max(z.max(), target[1])+1)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.add_patch(plt.Circle(target, 0.1, color="g", fill=False))

    frame_line, = ax.plot([], [], "k", lw=5)   # thân quad
    tether_line, = ax.plot([], [], "gray", lw=2)  # dây/con lắc
    trail, = ax.plot([], [], "b-", lw=1, alpha=0.6)
    left_line,  = ax.plot([], [], "r", lw=2)   # ngón trái
    right_line, = ax.plot([], [], "r", lw=2)   # ngón phải

    def update(i):
        yc, zc, phic, thetac = y[i], z[i], phi[i], theta[i]

        # --- 1) VẼ THÂN VỚI -phi (để khi bay sang phải thì đầu chúi xuống) ---
        # R_body = Rot(-phi)
        c, s = np.cos(phic), np.sin(phic)
        R_body = np.array([[ c,  s],
                           [-s,  c]])  # tương đương Rot(-phi)

        # Thanh thân (nằm dọc trục y cục bộ)
        main = np.array([[-l_q,  l_q],
                         [  0 ,   0 ]])
        body = R_body @ main + np.array([[yc, yc],
                                         [zc, zc]])
        frame_line.set_data(body[0], body[1])

        # --- 2) VẼ CON LẮC THEO GÓC THẾ GIỚI = phi + theta ---
        ang = phic + thetac
        pend_w = np.array([[0.0,                 l_p*np.sin(ang)],
                           [0.0,               - l_p*np.cos(ang)]])
        pend_w = pend_w + np.array([[yc, yc],
                                    [zc, zc]])
        tether_line.set_data(pend_w[0], pend_w[1])

        # --- 3) VẼ "NGÓN KẸP" THEO GÓC THẾ GIỚI CỦA ĐẦU CON LẮC ---
        end_x, end_y = pend_w[0,1], pend_w[1,1]
        grip_angle = ang
        vx, vy = np.sin(grip_angle), -np.cos(grip_angle)  # trục dọc theo thanh kẹp
        nx, ny = -vy, vx                                   # pháp tuyến

        L_finger, offset = 0.1, 0.05
        dx, dy = vx * L_finger, vy * L_finger

        # ngón trái
        left_line.set_data([end_x + nx*offset, end_x + nx*offset + dx],
                           [end_y + ny*offset, end_y + ny*offset + dy])
        # ngón phải
        right_line.set_data([end_x - nx*offset, end_x - nx*offset + dx],
                            [end_y - ny*offset, end_y - ny*offset + dy])

        # vệt đường đi
        trail.set_data(y[:i+1], z[:i+1])

        return frame_line, tether_line, trail, left_line, right_line

    ani = FuncAnimation(fig, update, frames=len(states), interval=dt*1000, blit=True)
    plt.show()


# ==== Main ====
def main():
    controller = CascadePIDActuatedPendulum(target=(5.0, 5.0))
    states, controls = simulate(controller)
    animate(states, controls, target=(5.0,5.0))

if __name__ == "__main__":
    main()
