import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==== Parameters ====
dt, T = 0.02, 10.0
# Bạn có thể thay đổi m_g ở đây, ví dụ:
m_q, m_g, g = 0.5, 0.158, 9.81  # Giảm m_g như bạn đã thử
I_xx, l_p, l_q = 0.15, 0.35, 0.2
J_q, J_g, L_g = 0.15, 0.0, 0.35

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
        # LƯU Ý: Các thông số PID này được điều chỉnh cho m_g = 0.158
        # Nếu bạn thay đổi m_g, bạn có thể cần phải "tuning" lại chúng để hệ thống ổn định!
        # Ví dụ tuning cho m_g = 0.08:
        # self.pid_z   = PID(8.0, 1.0, 5.0,  out_min=-6.0, out_max=6.0)
        # self.pid_y   = PID(0.7,  0.0, 0.3, out_min=-np.deg2rad(20), out_max=np.deg2rad(20))
        # self.pid_phi = PID(40.0, 0.8, 15.0, out_min=U2_MIN, out_max=U2_MAX)
        # self.pid_theta = PID(1.2, 0.0, 0.5, out_min=U3_MIN, out_max=U3_MAX)
        # Hoặc dùng mặc định nếu m_g không thay đổi quá nhiều so với 0.158
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
        u3 = self.pid_theta(0.2, theta, dt) # + np.sin(self.t)*0.5 # Tạm loại bỏ dao động để mượt hơn
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
    # ======= Tham số hiển thị (không ảnh hưởng dynamics) =======
    scale_draw   = 2        # phóng to hình
    l_q_vis      = l_q * scale_draw
    l_p_vis      = l_p * scale_draw
    L_finger     = 0.10 * scale_draw
    offset       = 0.05 * scale_draw

    lw_body      = 5 * scale_draw
    lw_pend      = 2 * scale_draw
    lw_finger    = 2 * scale_draw
    lw_trail     = 1 * scale_draw
    lw_thrust    = 2 * scale_draw

    thrust_scale = 0.04 * scale_draw   # hệ số đổi N -> chiều dài vẽ
    thrust_base  = 0.08 * scale_draw   # nhô ra một chút ở chân rotor khi vẽ thanh lực

    # ======= Dữ liệu trạng thái =======
    y, z, phi, theta = states[:,0], states[:,2], states[:,4], states[:,6]

    # --- TẠO 2 KHUNG HÌNH CON SONG SONG ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20*scale_draw/2, 10*scale_draw/2), dpi=120)
    fig.suptitle('Quadrotor Simulation: Camera Tracking (Left) vs. Full View (Right)') # Tiêu đề chung

    # Cấu hình ax1 (Camera Tracking)
    ax1.set_aspect("equal"); ax1.grid(True, alpha=0.3)
    ax1.set_title("Camera Tracking")
    ax1.add_patch(plt.Circle(target, 0.1*scale_draw, color="g", fill=False)) # Mục tiêu
    
    # Cấu hình ax2 (Full View) - như cũ
    ax2.set_xlim(min(y.min(), target[0])-1, max(y.max(), target[0])+1)
    ax2.set_ylim(min(z.min(), target[1])-1, max(z.max(), target[1])+1)
    ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)
    ax2.set_title("Full View")
    ax2.add_patch(plt.Circle(target, 0.1*scale_draw, color="g", fill=False)) # Mục tiêu

    # --- KHAI BÁO CÁC ĐỐI TƯỢNG VẼ CHO ax1 ---
    frame_line1,  = ax1.plot([], [], "k",    lw=lw_body)
    tether_line1, = ax1.plot([], [], "gray", lw=lw_pend)
    trail1,       = ax1.plot([], [], "b-",   lw=lw_trail, alpha=0.6)
    left_line1,   = ax1.plot([], [], "r", lw=lw_finger)
    right_line1,  = ax1.plot([], [], "r", lw=lw_finger)
    left_thrust_line1,  = ax1.plot([], [], color="orange", lw=lw_thrust)
    right_thrust_line1, = ax1.plot([], [], color="orange", lw=lw_thrust)

    # --- KHAI BÁO CÁC ĐỐI TƯỢNG VẼ CHO ax2 ---
    frame_line2,  = ax2.plot([], [], "k",    lw=lw_body)
    tether_line2, = ax2.plot([], [], "gray", lw=lw_pend)
    trail2,       = ax2.plot([], [], "b-",   lw=lw_trail, alpha=0.6)
    left_line2,   = ax2.plot([], [], "r", lw=lw_finger)
    right_line2,  = ax2.plot([], [], "r", lw=lw_finger)
    left_thrust_line2,  = ax2.plot([], [], color="orange", lw=lw_thrust)
    right_thrust_line2, = ax2.plot([], [], color="orange", lw=lw_thrust)

    def rotor_forces(u1, u2, arm):
        fR = 0.5 * (u1 + u2 / max(1e-9, arm))
        fL = 0.5 * (u1 - u2 / max(1e-9, arm))
        return max(0.0, fL), max(0.0, fR)

    def update(i):
        j = i if i < len(controls) else len(controls) - 1
        u1c, u2c, _ = controls[j]
        fL, fR = rotor_forces(u1c, u2c, l_q)

        yc, zc, phic, thetac = y[i], z[i], phi[i], theta[i]

        # --- CẬP NHẬT CHO ax1 (Camera Tracking) ---
        view_span = 6.0 # Tổng chiều rộng/cao của khung nhìn
        ax1.set_xlim(yc - view_span / 2, yc + view_span / 2)
        ax1.set_ylim(zc - view_span / 2, zc + view_span / 2)
        
        c, s = np.cos(phic), np.sin(phic)
        R_body = np.array([[ c,  s], [-s,  c]])
        T = np.array([[yc, yc], [zc, zc]])
        main = np.array([[-l_q_vis,  l_q_vis], [   0.0,       0.0]])
        body = R_body @ main + T
        frame_line1.set_data(body[0], body[1])

        ang = phic + thetac
        pend_w = np.array([[0.0,                 l_p_vis*np.sin(ang)],
                           [0.0,               - l_p_vis*np.cos(ang)]])
        pend_w = pend_w + np.array([[yc, yc], [zc, zc]])
        tether_line1.set_data(pend_w[0], pend_w[1])

        end_x, end_y = pend_w[0,1], pend_w[1,1]
        vx, vy = np.sin(ang), -np.cos(ang)
        nx, ny = -vy, vx
        dx, dy = vx * L_finger, vy * L_finger
        left_line1.set_data([end_x + nx*offset, end_x + nx*offset + dx],
                            [end_y + ny*offset, end_y + ny*offset + dy])
        right_line1.set_data([end_x - nx*offset, end_x - nx*offset + dx],
                             [end_y - ny*offset, end_y - ny*offset + dy])
        
        left_bar_local  = np.array([[-l_q_vis, -l_q_vis], [ thrust_base, thrust_base + thrust_scale * fL]])
        right_bar_local = np.array([[ l_q_vis,  l_q_vis], [ thrust_base, thrust_base + thrust_scale * fR]])
        left_bar  = R_body @ left_bar_local  + T
        right_bar = R_body @ right_bar_local + T
        left_thrust_line1.set_data(left_bar[0],  left_bar[1])
        right_thrust_line1.set_data(right_bar[0], right_bar[1])
        
        trail1.set_data(y[:i+1], z[:i+1])

        # --- CẬP NHẬT CHO ax2 (Full View) ---
        # ax2 không cần thay đổi xlim/ylim sau khi khởi tạo
        frame_line2.set_data(body[0], body[1]) # Sử dụng cùng body đã tính
        tether_line2.set_data(pend_w[0], pend_w[1]) # Sử dụng cùng pend_w đã tính
        left_line2.set_data([end_x + nx*offset, end_x + nx*offset + dx],
                            [end_y + ny*offset, end_y + ny*offset + dy])
        right_line2.set_data([end_x - nx*offset, end_x - nx*offset + dx],
                             [end_y - ny*offset, end_y - ny*offset + dy])
        left_thrust_line2.set_data(left_bar[0],  left_bar[1])
        right_thrust_line2.set_data(right_bar[0], right_bar[1])
        trail2.set_data(y[:i+1], z[:i+1])


        return (frame_line1, tether_line1, trail1, left_line1, right_line1, left_thrust_line1, right_thrust_line1,
                frame_line2, tether_line2, trail2, left_line2, right_line2, left_thrust_line2, right_thrust_line2)

    ani = FuncAnimation(fig, update, frames=len(states), interval=dt*1000, blit=False)
    plt.show()


# ==== Main ====
def main():
    controller = CascadePIDActuatedPendulum(target=(5.0, 5.0))
    states, controls = simulate(controller)
    animate(states, controls, target=(5.0,5.0))

if __name__ == "__main__":
    main()