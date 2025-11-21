import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN FILE CSV
# Bạn hãy dán đường dẫn file csv vừa tạo vào đây
# Dùng r"..." để tránh lỗi đường dẫn trên Windows
csv_path = r"C:\Users\2003h\OneDrive\Máy tính\quad2d_gripper\Quadrotor-Control-System\src\minsnap_results\simulation_log.csv"
# ==========================================

# Kiểm tra file có tồn tại không
if not os.path.exists(csv_path):
    print(f"Lỗi: Không tìm thấy file tại {csv_path}")
    print("Hãy kiểm tra lại đường dẫn hoặc chạy file control4.py để tạo file CSV trước.")
    exit()

# 1. Đọc dữ liệu từ CSV
df = pd.read_csv(csv_path)

# 2. Khởi tạo khung hình (Figure) với 3 đồ thị con (Subplots)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# --- Đồ thị 1: Vị trí (x, z) ---
ax1.plot(df['t'], df['x_q'], label='x_q (Mét)', color='blue', linewidth=2)
ax1.plot(df['t'], df['z_q'], label='z_q (Mét)', color='red', linewidth=2)
ax1.set_ylabel('Position (m)')
ax1.set_title('Vị trí Quadrotor (Position)')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Đồ thị 2: Góc (Theta, Beta) ---
ax2.plot(df['t'], df['theta'], label='Theta (Thân)', color='green')
ax2.plot(df['t'], df['beta'], label='Beta (Tay gắp)', color='orange')
ax2.set_ylabel('Angle (rad)')
ax2.set_title('Góc nghiêng (Angles)')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.6)

# --- Đồ thị 3: Tín hiệu điều khiển (u1, u3, tau) ---
ax3.plot(df['t'], df['u1'], label='u1 (Lực nâng)', linestyle='-')
ax3.plot(df['t'], df['u3'], label='u3 (Moment thân)', linestyle='--')
ax3.plot(df['t'], df['tau'], label='tau (Moment tay)', linestyle='-.')
ax3.set_ylabel('Control Inputs')
ax3.set_xlabel('Time (s)')
ax3.set_title('Tín hiệu điều khiển')
ax3.legend(loc='upper right')
ax3.grid(True, linestyle='--', alpha=0.6)

# Tự động căn chỉnh khoảng cách cho đẹp
plt.tight_layout()

# Hiển thị
plt.show()