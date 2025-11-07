# control4.py
# -*- coding: utf-8 -*-
"""
PD + Feed-forward controller (x–z plane) với **bảo vệ beta**:
- Soft-saturate beta^d trong [beta_min, beta_max] *trước* khi tính feed-forward
  để ánh xạ phẳng (recover_inputs_from_flat) nhất quán với giới hạn cơ khí.
- Thêm "bumper" chống đập stop cơ khí: nếu beta đo được gần cận và tau_c
  đẩy theo hướng xấu, cắt torque.
- Giữ đúng ánh xạ u3 = Jq * theta_ddot + tau  (Eq. (31)).

Tham khảo ánh xạ phẳng và cấu trúc điều khiển trong IDETC'13 (Thomas et al.).
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd

# ---- File ánh xạ phẳng của bạn (recover_inputs_from_flat) ----
# Yêu cầu: tf.recover_inputs_from_flat(t, x_qd, z_qd, beta_d, params) -> dict(u1,u3,tau,theta,theta_dot)
import tranfer as tf  # giữ nguyên như code hiện có

# ---------------- Helpers ----------------
def finite_diff(t: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    out = y.astype(float).copy()
    for _ in range(order):
        out = np.gradient(out, t, edge_order=2)
    return out

def soft_sat_beta(beta: np.ndarray, beta_min: float, beta_max: float, k: float = 30.0) -> np.ndarray:
    """
    "Saturator" C^∞ mượt cho beta.
    Map beta qua logistic để đảm bảo đạo hàm liên tục cao (quan trọng cho min‑snap & FF).
    """
    # tuyến tính hóa -> logistic trong [0,1]
    s = (beta - beta_min) / max(1e-9, (beta_max - beta_min))
    s_bar = 1.0 / (1.0 + np.exp(-k*(s - 0.5)))
    return beta_min + (beta_max - beta_min) * s_bar

@dataclass
class Gains:
    # Outer loop (x -> attitude)
    kpx: float = 1.2
    kdx: float = 0.6
    # Altitude
    kpz: float = 10.0
    kdz: float = 5.5
    # Attitude (theta)
    kp_theta: float = 6.0
    kd_theta: float = 2.5
    # Arm (beta)
    kp_beta: float = 4.0
    kd_beta: float = 1.2

@dataclass
class BetaLimits:
    beta_min_deg: float = 0.0
    beta_max_deg: float = 100.0
    soft_k: float       = 30.0
    stop_margin_deg: float = 1.0  # tạo bumper khi tiến gần cận
    tau_max: float      = None    # (tuỳ) clip |tau| tối đa, đơn vị theo mô hình của bạn

class PDFFController:
    def __init__(
        self,
        flat_csv: str,
        params: Optional[Dict[str, float]] = None,
        gains: Optional[Gains] = None,
        beta_limits: Optional[BetaLimits] = None,
        beta_sign: int = +1,  # +1: CCW dương (như paper), -1: plant đo beta CW dương
    ):
        self.params: Dict[str, float] = dict(tf.PARAMS if params is None else params)
        self.gains = gains if gains is not None else Gains()
        self.lims  = beta_limits if beta_limits is not None else BetaLimits()
        self.s     = +1 if beta_sign >= 0 else -1  # s ∈ {+1,-1}

        # --------- Đọc quỹ đạo phẳng từ planner ---------
        df = pd.read_csv(flat_csv)
        for col in ("t", "x_q", "z_q", "beta"):
            if col not in df.columns:
                raise ValueError(f"Thiếu cột '{col}' trong {flat_csv}")
        self.t      = df["t"].to_numpy(dtype=float)
        self.x_qd   = df["x_q"].to_numpy(dtype=float)
        self.z_qd   = df["z_q"].to_numpy(dtype=float)
        beta_raw    = df["beta"].to_numpy(dtype=float)  # rad (quy ước 'paper')

        # --------- Soft-saturate beta^d (TRƯỚC khi recover FF) ---------
        bmin = np.deg2rad(self.lims.beta_min_deg)
        bmax = np.deg2rad(self.lims.beta_max_deg)
        self.beta_d = soft_sat_beta(beta_raw, bmin, bmax, k=self.lims.soft_k)

        # Đạo hàm mong muốn
        self.xdot_qd   = finite_diff(self.t, self.x_qd, 1)
        self.zdot_qd   = finite_diff(self.t, self.z_qd, 1)
        self.betadot_d = finite_diff(self.t, self.beta_d, 1)

        # --------- Feed-forward từ ánh xạ phẳng (Eq. (16)–(31)) ---------
        ff = tf.recover_inputs_from_flat(self.t, self.x_qd, self.z_qd, self.beta_d, self.params)
        self.u1_d        = ff["u1"].astype(float)
        self.u3_d_paper  = ff["u3"].astype(float)
        self.tau_d_paper = ff["tau"].astype(float)
        self.theta_d     = ff["theta"].astype(float)
        self.theta_dot_d = ff["theta_dot"].astype(float)

        # Chuyển FF về quy ước plant theo beta_sign, đảm bảo u3 = Jq*theta_ddot + tau vẫn đúng
        self.tau_d = (1.0 / self.s) * self.tau_d_paper
        self.u3_d  = self.u3_d_paper + (self.s - 1.0) * self.tau_d_paper

        # dt mẫu thời gian
        self.dt = float(np.mean(np.diff(self.t)))

    def _apply_beta_bumper(self, tau_c: float, beta_meas: float) -> float:
        """
        Cắt torque khi beta tiến gần tới cận và torque đẩy theo hướng xấu.
        beta_meas: đo theo quy ước 'plant'.
        """
        bmin = np.deg2rad(self.lims.beta_min_deg) + np.deg2rad(self.lims.stop_margin_deg)
        bmax = np.deg2rad(self.lims.beta_max_deg) - np.deg2rad(self.lims.stop_margin_deg)
        if beta_meas >= bmax and tau_c > 0.0:
            tau_c = 0.0
        if beta_meas <= bmin and tau_c < 0.0:
            tau_c = 0.0
        if self.lims.tau_max is not None:
            tau_c = float(np.clip(tau_c, -abs(self.lims.tau_max), abs(self.lims.tau_max)))
        return tau_c

    def step(self, i: int, meas: Dict[str, float]):
        """
        Một bước điều khiển tại chỉ số i.
        meas gồm: x_q, z_q, xdot_q, zdot_q, theta, theta_dot, beta, beta_dot  (beta theo quy ước plant)
        Trả về: (u1_cmd, u3_cmd, tau_cmd) theo quy ước plant.
        """
        g = self.gains

        # Desired tại bước i
        x_d, z_d   = self.x_qd[i], self.z_qd[i]
        xd_d, zd_d = self.xdot_qd[i], self.zdot_qd[i]
        th_d, thd_d = self.theta_d[i], self.theta_dot_d[i]
        beta_d, betad_d = self.beta_d[i], self.betadot_d[i]
        u1_ff, u3_ff, tau_ff = self.u1_d[i], self.u3_d[i], self.tau_d[i]

        # Map beta đo về quy ước 'paper' để tính PD
        beta_m  = self.s * float(meas["beta"])
        betad_m = self.s * float(meas["beta_dot"])

        # Sai lệch
        ex   = x_d - float(meas["x_q"])
        ez   = z_d - float(meas["z_q"])
        exd  = xd_d - float(meas["xdot_q"])
        ezd  = zd_d - float(meas["zdot_q"])
        eth  = th_d - float(meas["theta"])
        ethd = thd_d - float(meas["theta_dot"])
        eb   = beta_d - beta_m
        ebd  = betad_d - betad_m

        # (11) thrust PD + FF
        u1_c = g.kpz * ez + g.kdz * ezd + u1_ff

        # (13) lateral -> attitude command
        lat_cmd = g.kpx * ex + g.kdx * exd
        lat_cmd = float(np.clip(lat_cmd, -0.999, 0.999))
        theta_c = np.arcsin(lat_cmd) + th_d

        # (12) attitude moment PD + FF
        u3_pd = g.kp_theta * (theta_c - float(meas["theta"])) + g.kd_theta * ethd
        u3_c  = u3_pd + u3_ff

        # Arm (beta): PD + FF; PD torque ở quy ước 'paper'
        tau_pd_paper = g.kp_beta * eb + g.kd_beta * ebd
        tau_pd_plant = (1.0 / self.s) * tau_pd_paper
        tau_c = tau_pd_plant + tau_ff

        # Bumper bảo vệ stop cơ khí
        tau_c = self._apply_beta_bumper(tau_c, float(meas["beta"]))

        return float(u1_c), float(u3_c), float(tau_c)

# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Controller4: PD + FF với bảo vệ beta.")
    parser.add_argument("--flat_csv", type=str, default="flat_outputs.csv", help="CSV từ QP (t,x_q,z_q,beta,...)")
    parser.add_argument("--beta_sign", type=int, default=+1, help="+1: plant cùng quy ước 'paper'; -1: đo ngược dấu")
    parser.add_argument("--beta_min_deg", type=float, default=0.0)
    parser.add_argument("--beta_max_deg", type=float, default=100.0)
    parser.add_argument("--soft_k", type=float, default=30.0)
    parser.add_argument("--stop_margin_deg", type=float, default=1.0)
    parser.add_argument("--tau_max", type=float, default=None)
    args = parser.parse_args()

    lims = BetaLimits(
        beta_min_deg=args.beta_min_deg,
        beta_max_deg=args.beta_max_deg,
        soft_k=args.soft_k,
        stop_margin_deg=args.stop_margin_deg,
        tau_max=args.tau_max,
    )

    ctrl = PDFFController(flat_csv=args.flat_csv, beta_limits=lims, beta_sign=args.beta_sign)
    print("Controller4 ready. Gọi PDFFController.step(i, meas) trong vòng lặp điều khiển.")
