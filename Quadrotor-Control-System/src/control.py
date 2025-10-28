import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd  ### THAY ĐỔI: Thêm pandas để đọc CSV

# ==== Parameters ====
# ### THAY ĐỔI: Cập nhật T và dt để khớp với quỹ đạo từ qp3.py
# qp3.py chạy trong 3.5 giây.
dt, T = 0.02, 3.5  
m_q, m_g, g = 0.5, 0.158, 9.81
I_xx, l_p, l_q = 0.15, 0.35, 0.2
J_q, J_g, L_g = 0.15, 0.0, 0.15

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
        self.i_term += e * dt
        d = (e - self.prev_e) / dt if dt > 1e-6 else 0.0
        self.prev_e = e
        u = self.Kp*e + self.Ki*self.i_term + self.Kd*d
        if self.out_min is not None and self.out_max is not None:
            u = np.clip(u, self.out_min, self.out_max)
        return float(u)

# ==== Controller ====
### THAY ĐỔI: Bộ điều khiển giờ đây nhận dữ liệu quỹ đạo
class CascadePIDActuatedPendulum:
    def __init__(self, trajectory_df):
        # Lưu trữ dữ liệu quỹ đạo tham chiếu
        self.t_ref = trajectory_df["t"].to_numpy()
        self.y_ref_traj = trajectory_df["x_q"].to_numpy()
        self.z_ref_traj = trajectory_df["z_q"].to_numpy()
        self.beta_ref_traj = trajectory_df["beta"].to_numpy() # (rad)

        # Các bộ PID (có thể bạn sẽ cần tuning lại gain cho bám quỹ đạo)
        self.pid_z   = PID(12.0, 1.5, 7.0,  out_min=-6.0, out_max=6.0)
        self.pid_y   = PID(1.0,  0.0, 0.35, out_min=-np.deg2rad(20), out_max=np.deg2rad(20))
        self.pid_phi = PID(65.0, 1.0, 18.0, out_min=U2_MIN, out_max=U2_MAX)
        self.pid_theta = PID(2.0, 0.0, 0.6, out_min=U3_MIN, out_max=U3_MAX)
        
        self.t = 0.0 # Thời gian mô phỏng nội bộ của controller

    def step(self, state):
        y, ydot, z, zdot, phi, phidot, theta, thetad = state

        # ### THAY ĐỔI: Tìm mục tiêu (ref) tại thời điểm t hiện tại
        # Dùng nội suy tuyến tính (linear interpolation) để tìm giá trị ref
        # `np.interp` sẽ tìm xem self.t đang ở đâu trong mảng self.t_ref
        # và trả về giá trị tương ứng từ mảng quỹ đạo.
        y_ref = np.interp(self.t, self.t_ref, self.y_ref_traj)
        z_ref = np.interp(self.t, self.t_ref, self.z_ref_traj)
        beta_ref = np.interp(self.t, self.t_ref, self.beta_ref_traj)

        # Tính toán PID dựa trên mục tiêu ĐỘNG (moving target)
        az_cmd = self.pid_z(z_ref, z, dt) 
        u1 = (m_q + m_g) * (g + az_cmd)
        u1 = np.clip(u1, U1_MIN, U1_MAX)
        
        phi_ref = self.pid_y(y_ref, y, dt)
        u2 = self.pid_phi(phi_ref, phi, dt)
        
        u3 = self.pid_theta(beta_ref, theta, dt) # Bám theo góc beta từ quỹ đạo
        u3 = np.clip(u3, U3_MIN, U3_MAX)
        
        self.t += dt # Cập nhật thời gian
        return u1, u2, u3

# ==== Simulation ====
def simulate(controller, T=T, dt=dt):
    N = int(T/dt)
    
    # ### THAY ĐỔI: Cập nhật trạng thái ban đầu để khớp với qp3.py
    # qp3.py bắt đầu từ x0=-2.0, z0=2.0, beta0=25.0 deg
    state = np.array([
        -2.0, 0.0,                # y, y_dot (khớp x0 từ qp3)
        2.0, 0.0,                 # z, z_dot (khớp z0 từ qp3)
        0.0, 0.0,                 # phi, phi_dot
        np.deg2rad(25.0), 0.0     # beta, beta_dot (khớp beta0_deg từ qp3)
    ]) 
    
    states, controls = [state], []
    # ### THAY ĐỔI: Tạo mảng thời gian để lưu
    times = [0.0]
    
    for _ in range(N):
        u1, u2, u3 = controller.step(state) # cap nhat dieu khien theo trang thai hien tai
        u = jnp.array([u1, u2, u3], dtype=jnp.float32)
        state = np.array(jax_dynamics_matrix(state, u))
        
        states.append(state)
        controls.append([u1, u2, u3])
        times.append(times[-1] + dt)
        
    return np.array(times), np.array(states), np.array(controls)

# ==== Visualization ====
def animate(t_sim, states, controls, trajectory_df, dt=dt):
    # ... (giữ nguyên phần tham số hiển thị) ...
    scale_draw   = 2
    l_q_vis      = l_q * scale_draw
    l_p_vis      = l_p * scale_draw
    L_finger     = 0.10 * scale_draw
    offset       = 0.05 * scale_draw
    lw_body      = 5 * scale_draw
    lw_pend      = 2 * scale_draw
    lw_finger    = 2 * scale_draw
    lw_trail     = 1 * scale_draw
    lw_thrust    = 2 * scale_draw
    thrust_scale = 0.04 * scale_draw
    thrust_base  = 0.08 * scale_draw

    y, z, phi, theta = states[:,0], states[:,2], states[:,4], states[:,6]
    
    # ### THAY ĐỔI: Lấy quỹ đạo tham chiếu để vẽ
    y_ref = trajectory_df["x_q"].to_numpy()
    z_ref = trajectory_df["z_q"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20*scale_draw/2, 10*scale_draw/2), dpi=120)
    fig.suptitle('Quadrotor Simulation: Camera Tracking (Left) vs. Full View (Right)')

    # --- Cấu hình ax1 (Camera Tracking) ---
    ax1.set_aspect("equal"); ax1.grid(True, alpha=0.3)
    ax1.set_title("Camera Tracking")
    # Vẽ quỹ đạo tham chiếu (màu xanh lá)
    ax1.plot(y_ref, z_ref, 'g--', lw=1, alpha=0.7, label='Reference Path')
    
    # --- Cấu hình ax2 (Full View) ---
    # Tính toán giới hạn dựa trên cả quỹ đạo mô phỏng (states) và quỹ đạo tham chiếu (ref)
    all_y = np.concatenate([y, y_ref])
    all_z = np.concatenate([z, z_ref])
    ax2.set_xlim(all_y.min()-1, all_y.max()+1)
    ax2.set_ylim(all_z.min()-1, all_z.max()+1)
    ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)
    ax2.set_title("Full View")
    # Vẽ quỹ đạo tham chiếu (màu xanh lá)
    ax2.plot(y_ref, z_ref, 'g--', lw=1, alpha=0.7, label='Reference Path')

    # --- (Giữ nguyên phần khai báo các đối tượng vẽ cho ax1 và ax2) ---
    frame_line1,  = ax1.plot([], [], "k",    lw=lw_body)
    tether_line1, = ax1.plot([], [], "gray", lw=lw_pend)
    trail1,       = ax1.plot([], [], "b-",   lw=lw_trail, alpha=0.6, label='Simulated Path')
    left_line1,   = ax1.plot([], [], "r", lw=lw_finger)
    right_line1,  = ax1.plot([], [], "r", lw=lw_finger)
    left_thrust_line1,  = ax1.plot([], [], color="orange", lw=lw_thrust)
    right_thrust_line1, = ax1.plot([], [], color="orange", lw=lw_thrust)

    frame_line2,  = ax2.plot([], [], "k",    lw=lw_body)
    tether_line2, = ax2.plot([], [], "gray", lw=lw_pend)
    trail2,       = ax2.plot([], [], "b-",   lw=lw_trail, alpha=0.6, label='Simulated Path')
    left_line2,   = ax2.plot([], [], "r", lw=lw_finger)
    right_line2,  = ax2.plot([], [], "r", lw=lw_finger)
    left_thrust_line2,  = ax2.plot([], [], color="orange", lw=lw_thrust)
    right_thrust_line2, = ax2.plot([], [], color="orange", lw=lw_thrust)
    
    # Thêm legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    def rotor_forces(u1, u2, arm):
        fR = 0.5 * (u1 + u2 / max(1e-9, arm))
        fL = 0.5 * (u1 - u2 / max(1e-9, arm))
        return max(0.0, fL), max(0.0, fR)

    def update(i):
        # ### THAY ĐỔI: Cẩn thận với độ dài
        # t_sim và states dài N+1, controls dài N
        j = i - 1 if i > 0 else 0
        if j >= len(controls): j = len(controls) - 1
            
        u1c, u2c, _ = controls[j]
        fL, fR = rotor_forces(u1c, u2c, l_q)

        yc, zc, phic, thetac = y[i], z[i], phi[i], theta[i]

        # --- CẬP NHẬT CHO ax1 (Camera Tracking) ---
        view_span = 4.0 # Thu nhỏ view lại
        ax1.set_xlim(yc - view_span / 2, yc + view_span / 2)
        ax1.set_ylim(zc - view_span / 2, zc + view_span / 2)
        
        # ... (Phần còn lại của hàm update giữ nguyên) ...
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
        frame_line2.set_data(body[0], body[1])
        tether_line2.set_data(pend_w[0], pend_w[1])
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
    # ### THAY ĐỔI: Đọc file CSV quỹ đạo
    # Đảm bảo đường dẫn này chính xác
    flat_csv_path = "/home/huynq/Project20251/Quadrotor-Control-System/src/minsnap_results/flat_outputs.csv"
    try:
        trajectory_df = pd.read_csv(flat_csv_path)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {flat_csv_path}")
        print("Vui lòng chạy file qp3.py trước để tạo file này.")
        return

    # Khởi tạo controller với quỹ đạo
    controller = CascadePIDActuatedPendulum(trajectory_df)
    
    # Chạy mô phỏng
    t_sim, states, controls = simulate(controller, T=T)
    
    # Chạy hoạt ảnh
    animate(t_sim, states, controls, trajectory_df, dt=dt)

if __name__ == "__main__":
    main()