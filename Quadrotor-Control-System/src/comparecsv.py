import pandas as pd
import matplotlib.pyplot as plt

# 1. Đọc 2 file CSV
# Thay đổi đường dẫn nếu file của bạn nằm ở thư mục khác
df_sim = pd.read_csv('C:\\Users\\2003h\\OneDrive\\Máy tính\\quad2d_gripper\\Quadrotor-Control-System\\src\\minsnap_results\\simulation_log.csv')
df_flat = pd.read_csv('C:\\Users\\2003h\\OneDrive\\Máy tính\\quad2d_gripper\\Quadrotor-Control-System\\src\\minsnap_results\\flat_outputs1.csv')

# 2. Tạo khung hình đồ thị
plt.figure(figsize=(10, 6))

# 3. Vẽ Beta từ file Simulation (Kết quả mô phỏng)
# Giả sử cột thời gian là 't' và giá trị là 'beta'
plt.plot(df_sim['t'], df_sim['beta'], 
         label='Beta (Simulation)', color='blue', linewidth=2)

# 4. Vẽ Beta từ file Flat Outputs (Giá trị mong muốn/tham chiếu)
# Dùng nét đứt (linestyle='--') để dễ phân biệt nếu chúng trùng nhau
plt.plot(df_flat['t'], df_flat['beta'], 
         label='Beta (Reference/Flat)', color='red', linestyle='--', linewidth=2)

# 5. Trang trí đồ thị
plt.xlabel('Thời gian (s)')
plt.ylabel('Góc Beta (rad)')
plt.title('So sánh góc Beta: Mô phỏng vs Tham chiếu')
plt.legend()     # Hiện chú thích
plt.grid(True)   # Hiện lưới

# 6. Hiển thị
plt.show()