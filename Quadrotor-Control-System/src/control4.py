# control1.py
# -*- coding: utf-8 -*-
"""
PD + Feed-forward controller cho quadrotor + gripper (mặt phẳng x–z),
tuân theo IDETC'13 (Thomas et al., 2013). Các feed-forward u1^d, u3^d, tau^d,
theta^d, thetadot^d lấy từ tranfer.recover_inputs_from_flat (ánh xạ (16)–(31)).

Điểm khác biệt so với bản cũ:
  (1) Sửa mô phỏng scope1: dùng u2 = u3 (KHÔNG cộng tau) để thỏa Eq. (31).
  (2) Thêm beta_sign để xử lý trường hợp plant dùng quy ước beta ngược dấu:
      - Map đo lường beta về quy ước của bài báo (CCW dương) để tính PD.
      - Chuyển ngược phần PD torque về quy ước plant khi phát lệnh.
      - Điều chỉnh đúng cặp (u3^d, tau^d) FF để vẫn đảm bảo u3^d - tau^d = Jq*theta_ddot^d
        dưới mọi lựa chọn beta_sign.

Tham khảo: AVIAN-INSPIRED GRASPING FOR QUADROTOR MICRO UAVS, Eq. (1),(16)–(31).
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
import os

# ===== Lấy ánh xạ phẳng -> điều khiển đã có sẵn =====
# File người dùng cung cấp là "tranfer.py"
import tranfer as tf  # :contentReference[oaicite:4]{index=4}


# ---------- Helpers ---------- tính đao hàm bằng sai phân hữu hạn mảng y theo t ----------
def finite_diff(t: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Central finite-difference of given order."""
    out = y.astype(float).copy()
    for _ in range(order):
        out = np.gradient(out, t, edge_order=2)
    return out


@dataclass
class Gains:
    # Vòng ngoài (x -> attitude)
    kpx: float = 1.2
    kdx: float = 0.6
    # Cao độ
    kpz: float = 10.0
    kdz: float = 5.5
    # Attitude (theta)
    kp_theta: float = 6.0
    kd_theta: float = 2.5
    # Cánh tay (beta)
    kp_beta: float = 4.0
    kd_beta: float = 1.2


class PDFFController:
    def __init__(
        self,
        flat_csv: str,
        params: Optional[Dict[str, float]] = None,
        gains: Optional[Gains] = None,
        beta_sign: int = +1,  # +1: cùng chiều CCW như bài báo; -1: plant dùng CW dương
    ):
        # Tham số động học (trùng tranfer.py)
        self.params: Dict[str, float] = dict(tf.PARAMS if params is None else params)
        self.gains = gains if gains is not None else Gains()
        self.s = +1 if beta_sign >= 0 else -1  # s ∈ {+1,-1}

        # Đọc quỹ đạo phẳng từ QP (t, x_q, z_q, beta) — do qp3 ghi ra
        # (qp3 đã ràng buộc β quét trái sau gắp theo đúng Fig. 6). :contentReference[oaicite:5]{index=5}
        df = pd.read_csv(flat_csv)
        for col in ("t", "x_q", "z_q", "beta"):
            if col not in df.columns:
                raise ValueError(f"Thiếu cột '{col}' trong {flat_csv}")
        self.t       = df["t"].to_numpy(dtype=float)
        self.x_qd    = df["x_q"].to_numpy(dtype=float)
        self.z_qd    = df["z_q"].to_numpy(dtype=float)
        self.beta_d  = df["beta"].to_numpy(dtype=float)  # rad (quy ước 'paper': CCW dương)

        # Đạo hàm mong muốn cho PD
        self.xdot_qd    = finite_diff(self.t, self.x_qd, 1)
        self.zdot_qd    = finite_diff(self.t, self.z_qd, 1)
        self.betadot_d  = finite_diff(self.t, self.beta_d, 1)

        # Feed‑forward từ ánh xạ phẳng→điều khiển (Eq. (16)–(31)). 
        ff = tf.recover_inputs_from_flat(self.t, self.x_qd, self.z_qd, self.beta_d, self.params)
        self.u1_d        = ff["u1"].astype(float)
        self.u3_d_paper  = ff["u3"].astype(float)      # u3 theo quy ước 'paper'
        self.tau_d_paper = ff["tau"].astype(float)     # tau theo quy ước 'paper'
        self.theta_d     = ff["theta"].astype(float)
        self.theta_dot_d = ff["theta_dot"].astype(float)

        # Điều chỉnh FF (u3, tau) về quy ước 'plant' theo beta_sign.
        # Phân tích: u3 = Jq*theta_ddot + tau   (Eq. (31))  ⇒
        #   tau_plant = (1/s)*tau_paper  (lật dấu nếu s=-1)
        #   u3_plant  = u3_paper + (tau_plant - tau_paper) = u3_paper + (s-1)*tau_paper
        self.tau_d = (1.0 / self.s) * self.tau_d_paper
        self.u3_d  = self.u3_d_paper + (self.s - 1.0) * self.tau_d_paper

        # dt
        self.dt = float(np.mean(np.diff(self.t)))

    def step(self, i: int, meas: Dict[str, float]):
        """
        Một bước điều khiển tại chỉ số i.
        meas cần có:
            x_q, z_q, xdot_q, zdot_q, theta, theta_dot, beta, beta_dot
        Trả về: u1_cmd, u3_cmd, tau_cmd (theo quy ước 'plant')
        """
        g = self.gains

        # Lấy desired tại bước i
        x_d, z_d   = self.x_qd[i], self.z_qd[i]
        xd_d, zd_d = self.xdot_qd[i], self.zdot_qd[i]
        th_d, thd_d = self.theta_d[i], self.theta_dot_d[i]
        beta_d, betad_d = self.beta_d[i], self.betadot_d[i]
        u1_ff, u3_ff, tau_ff = self.u1_d[i], self.u3_d[i], self.tau_d[i]

        # --- Map đo lường beta của plant -> quy ước 'paper' để tính sai lệch ---
        beta_m   = self.s * float(meas["beta"])
        betad_m  = self.s * float(meas["beta_dot"])

        # Sai lệch
        ex   = x_d - float(meas["x_q"])
        ez   = z_d - float(meas["z_q"])
        exd  = xd_d - float(meas["xdot_q"])
        ezd  = zd_d - float(meas["zdot_q"])
        eth  = th_d - float(meas["theta"])
        ethd = thd_d - float(meas["theta_dot"])
        eb   = beta_d - beta_m
        ebd  = betad_d - betad_m

        # --- Eq. (11): thrust PD + FF ---
        u1_c = g.kpz * ez + g.kdz * ezd + u1_ff

        # --- Eq. (13): lateral -> attitude command ---
        lat_cmd = g.kpx * ex + g.kdx * exd
        lat_cmd = float(np.clip(lat_cmd, -0.999, 0.999))  # an toàn cho asin
        theta_c = np.arcsin(lat_cmd) + th_d

        # --- Eq. (12): attitude moment PD + FF ---
        u3_pd = g.kp_theta * (theta_c - float(meas["theta"])) + g.kd_theta * ethd
        u3_c  = u3_pd + u3_ff

        # --- Cánh tay (β): PD + FF ---
        tau_pd_paper = g.kp_beta * eb + g.kd_beta * ebd  # torque theo quy ước 'paper'
        # Map về quy ước 'plant' (F' = F / s cho đổi biến q' = s q) với s ∈ {±1}
        tau_pd_plant = (1.0 / self.s) * tau_pd_paper
        tau_c = tau_pd_plant + tau_ff

        return float(u1_c), float(u3_c), float(tau_c)

    # ---------------- (Tuỳ chọn) harness mô phỏng với scope1.py ----------------
    def simulate_with_scope1(self, save_csv: Optional[str] = None, animate: bool = False):
        """
        Harness tối giản chạy mô phỏng phẳng với scope1 (nếu có).
        scope1 dùng state = [y, y_dot, z, z_dot, phi, phi_dot, beta, beta_dot],
        và động học: J_q*phi_ddot = (u2 - tau). ĐỂ ÁNH XẠ ĐÚNG (Eq. 31), PHẢI DÙNG u2 = u3.
        """
        try:
            from scope1 import jax_dynamics_matrix
        except Exception as e:
            raise RuntimeError("Không import được scope1.py; không thể mô phỏng.") from e

        # Khởi tạo tại tư thế mong muốn ban đầu
        y0   = self.x_qd[0]
        z0   = self.z_qd[0]
        phi0 = self.theta_d[0]
        beta0 = self.beta_d[0] * (1.0 / self.s)  # map beta desired sang quy ước plant cho state
        state = np.array([y0, 0.0, z0, 0.0, phi0, 0.0, beta0, 0.0], dtype=float)

        states = [state.copy()]
        cmds   = []  # (u1, u2, u3, tau)

        for i in range(len(self.t)-1):
            meas = dict(
                x_q=state[0],     xdot_q=state[1],
                z_q=state[2],     zdot_q=state[3],
                theta=state[4],   theta_dot=state[5],
                beta=state[6],    beta_dot=state[7],
            )
            u1, u3, tau = self.step(i, meas)

            # ÁNH XẠ ĐÚNG: scope1 dùng J_q*phi_ddot = (u2 - tau) ⇒ Đặt u2 = u3
            u2 = u3

            # Tích phân một bước
            control = np.array([u1, u2, tau], dtype=float)
            state = np.array(jax_dynamics_matrix(state, control, dt=self.dt), dtype=float)

            states.append(state.copy())
            cmds.append([u1, u2, u3, tau])

        states = np.array(states)
        cmds   = np.array(cmds)

        if save_csv:
            if not save_csv.lower().endswith('.csv'):
                # Nếu người dùng chỉ nhập tên thư mục (ví dụ: minsnap_results)
                folder_path = save_csv
                
                # Tạo thư mục nếu chưa có
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                # Tự động đặt tên file
                save_csv = os.path.join(folder_path, "ketqua.csv")
            else:
                # Nếu người dùng nhập cả tên file (ví dụ: results/log.csv)
                # Cần đảm bảo thư mục cha tồn tại
                folder_path = os.path.dirname(save_csv)
                if folder_path and not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            log = pd.DataFrame({
                "t": self.t[:len(cmds)],
                "u1": cmds[:,0], "u2": cmds[:,1], "u3": cmds[:,2], "tau": cmds[:,3],
                "x_q": states[:-1,0], "xdot_q": states[:-1,1],
                "z_q": states[:-1,2], "zdot_q": states[:-1,3],
                "theta": states[:-1,4], "theta_dot": states[:-1,5],
                "beta": states[:-1,6],  "beta_dot": states[:-1,7],
            })
            log.to_csv(save_csv, index=False)
            print(f"[OK] Saved sim log -> {save_csv}")

        if animate:
            try:
                from scope1 import animate as scope1_animate
                scope1_animate(states, cmds[:, :3], target=(self.x_qd[-1], self.z_qd[-1]), dt=self.dt)
            except Exception as e:
                print(f"Animation failed: {e}")

        return states, cmds


# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PD + FF quad controller (planar).")
    parser.add_argument("--flat_csv", type=str, default="flat_outputs.csv",
                        help="CSV planner: cột t,x_q,z_q,beta (beta: rad, quy ước 'paper').")
    parser.add_argument("--beta_sign", type=int, default=+1,
                        help="+1 nếu plant dùng CCW dương như bài báo; -1 nếu plant dùng CW dương.")
    parser.add_argument("--simulate", action="store_true", help="Chạy mô phỏng với scope1.py (nếu có).")
    parser.add_argument("--save_csv", type=str, default=None, help="Nếu set, lưu log mô phỏng ra CSV.")
    parser.add_argument("--animate", action="store_true", help="Hiển thị animation (scope1).")
    args = parser.parse_args()

    ctrl = PDFFController(flat_csv=args.flat_csv, beta_sign=args.beta_sign)

    if args.simulate:
        ctrl.simulate_with_scope1(save_csv=args.save_csv, animate=args.animate)
    else:
        print("Controller ready. Gọi PDFFController.step(i, meas) mỗi chu kỳ điều khiển.")
