# control4.py  — limit-aware PD/FF controller (planar quad + gripper)
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Feed‑forward mapping (Eq. (16)–(31)) bạn đã có trong tranfer.py
import tranfer as tf  # dùng đúng file 'tranfer.py' của bạn  :contentReference[oaicite:2]{index=2}

# ---------- utils ----------
def _diff(t: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    out = y.astype(float).copy()
    for _ in range(order):
        out = np.gradient(out, t, edge_order=2)
    return out

def _lowpass(prev: float, new: float, alpha: float) -> float:
    return alpha * prev + (1.0 - alpha) * new

@dataclass
class Gains:
    # x -> theta
    kpx: float = 1.6
    kdx: float = 0.8
    # z loop (PD, có thể thêm PI nếu cần)
    kpz: float = 12.0
    kdz: float = 6.0
    # attitude
    kp_theta: float = 7.5
    kd_theta: float = 3.0
    # arm beta
    kp_beta: float = 6.0
    kd_beta: float = 2.0

class PDFFController:
    def __init__(self,
                 flat_csv: str,
                 params: Optional[Dict[str, float]] = None,
                 gains: Optional[Gains] = None,
                 beta_sign: int = +1,
                 beta_limits_deg: Optional[Tuple[float, float]] = (0.0, 130.0),  # giới hạn β theo quy ước 'paper'
                 ref_smooth_tau: float = 0.05,     # [s] hằng số thời gian lọc βʳ
                 theta_c_rate_limit: float = np.deg2rad(400.0)  # [rad/s]
                 ):
        # tham số động học
        self.params = dict(tf.PARAMS if params is None else params)
        self.g = gains if gains is not None else Gains()
        # s = +1 (plant dùng CCW dương như paper) hoặc -1 (plant CW dương)
        self.s = +1 if beta_sign >= 0 else -1

        # đọc quỹ đạo phẳng
        df = pd.read_csv(flat_csv)
        for col in ("t", "x_q", "z_q", "beta"):
            if col not in df.columns:
                raise ValueError(f"Thiếu cột '{col}' trong {flat_csv}")
        self.t      = df["t"].to_numpy(float)
        self.x_qd   = df["x_q"].to_numpy(float)
        self.z_qd   = df["z_q"].to_numpy(float)
        self.beta_d = df["beta"].to_numpy(float)  # rad (quy ước 'paper')

        # đạo hàm
        self.xdot_qd   = _diff(self.t, self.x_qd, 1)
        self.zdot_qd   = _diff(self.t, self.z_qd, 1)
        self.betadot_d = _diff(self.t, self.beta_d, 1)

        # feed‑forward theo phụ lục (16)–(31)
        ff = tf.recover_inputs_from_flat(self.t, self.x_qd, self.z_qd, self.beta_d, self.params)
        self.u1_d_paper   = ff["u1"].astype(float)
        self.u3_d_paper   = ff["u3"].astype(float)
        self.tau_d_paper  = ff["tau"].astype(float)
        self.theta_d      = ff["theta"].astype(float)
        self.theta_dot_d  = ff["theta_dot"].astype(float)
        self.theta_ddot_d = ff["theta_ddot"].astype(float)

        # đổi (u3, tau) theo quy ước plant dựa trên beta_sign:
        # tau_plant = (1/s) tau_paper ;  u3_plant = u3_paper + (s-1) tau_paper  (để giữ u3 - tau = J_q θ̈)  :contentReference[oaicite:3]{index=3}
        self.tau_d = (1.0 / self.s) * self.tau_d_paper
        self.u3_d  = self.u3_d_paper + (self.s - 1.0) * self.tau_d_paper
        self.u1_d  = self.u1_d_paper

        self.dt = float(np.mean(np.diff(self.t)))

        # giới hạn β và lọc tham chiếu
        if beta_limits_deg is None:
            self.beta_lim = None
        else:
            bmin, bmax = beta_limits_deg
            self.beta_lim = (np.deg2rad(bmin), np.deg2rad(bmax))  # ở quy ước 'paper'
        self.alpha_ref = np.exp(-self.dt / max(ref_smooth_tau, 1e-3))
        self.beta_ref_prev  = float(self.beta_d[0])
        self.betad_ref_prev = float(self.betadot_d[0])

        self.theta_c_prev = self.theta_d[0]
        self.theta_c_rate_limit = float(theta_c_rate_limit)

    # ======== một bước điều khiển ========
    def step(self, i: int, meas: Dict[str, float]):
        g = self.g

        # desired + FF ở bước i
        x_d, z_d   = self.x_qd[i], self.z_qd[i]
        xd_d, zd_d = self.xdot_qd[i], self.zdot_qd[i]
        th_d, thd_d = self.theta_d[i], self.theta_dot_d[i]
        beta_d_raw, betad_d_raw = self.beta_d[i], self.betadot_d[i]
        u1_ff = self.u1_d[i]
        u3_ff = self.u3_d[i]
        tau_ff = self.tau_d[i]

        # --- LIMIT‑AWARE REFERENCE for β ---
        beta_ref = beta_d_raw
        if self.beta_lim is not None:
            bmin, bmax = self.beta_lim
            beta_clip = float(np.clip(beta_d_raw, bmin, bmax))
            beta_ref  = _lowpass(self.beta_ref_prev, beta_clip, self.alpha_ref)  # lọc mượt
        betad_ref = (beta_ref - self.beta_ref_prev) / self.dt
        self.beta_ref_prev  = beta_ref
        self.betad_ref_prev = betad_ref

        # --- Nếu đang đẩy vào stop → tắt τᵈ và bù u₃ᵈ để giữ u₃−τ = J_q θ̈ (Eq.31) ---
        if self.beta_lim is not None:
            bmin, bmax = self.beta_lim
            pushing_hi = (beta_d_raw > bmax) and (betad_ref > 0.0)
            pushing_lo = (beta_d_raw < bmin) and (betad_ref < 0.0)
            if pushing_hi or pushing_lo:
                u3_ff = u3_ff - tau_ff   # giữ u3^d - tau^d bất biến
                tau_ff = 0.0             # không đẩy thêm vào stop  :contentReference[oaicite:4]{index=4}

        # --- đo lường (map β plant -> quy ước paper để tính lỗi) ---
        x_m, xd_m   = float(meas["x_q"]),   float(meas["xdot_q"])
        z_m, zd_m   = float(meas["z_q"]),   float(meas["zdot_q"])
        th_m, thd_m = float(meas["theta"]), float(meas["theta_dot"])
        beta_m      = self.s * float(meas["beta"])
        betad_m     = self.s * float(meas["beta_dot"])

        # --- lỗi ---
        ex,  exd  = x_d - x_m,   xd_d - xd_m
        ez,  ezd  = z_d - z_m,   zd_d - zd_m
        eth, ethd = th_d - th_m, thd_d - thd_m
        eb,  ebd  = beta_ref - beta_m, betad_ref - betad_m

        # --- u1 (Eq.11) PD + FF ---
        u1_c = g.kpz * ez + g.kdz * ezd + u1_ff

        # --- θc từ lateral (Eq.13) + rate limit để mượt ---
        lat_cmd = float(np.clip(g.kpx * ex + g.kdx * exd, -0.95, 0.95))
        theta_c_des = np.arcsin(lat_cmd) + th_d
        dtheta_c = np.clip((theta_c_des - self.theta_c_prev) / self.dt,
                           -self.theta_c_rate_limit, self.theta_c_rate_limit)
        theta_c = self.theta_c_prev + dtheta_c * self.dt
        self.theta_c_prev = theta_c

        # --- u3 (Eq.12) PD + FF ---
        u3_pd = g.kp_theta * (theta_c - th_m) + g.kd_theta * ethd
        u3_c  = u3_pd + u3_ff

        # --- τ: PD + FF theo 'paper', rồi map về plant bằng 1/s ---
        tau_pd_paper = g.kp_beta * eb + g.kd_beta * ebd
        tau_pd_plant = (1.0 / self.s) * tau_pd_paper
        tau_c = tau_pd_plant + tau_ff

        return float(u1_c), float(u3_c), float(tau_c)

    # ======== mô phỏng với scope1 (nếu có) ========
    def simulate_with_scope1(self, save_csv: Optional[str] = None, animate: bool = False):
        """
        scope1 sử dụng J_q*phi_ddot = (u2 - tau) ⇒ để áp mô‑men thân u3, ta đặt **u2 = u3**.  :contentReference[oaicite:5]{index=5}
        Có hard‑stop theo góc thế giới (phi+beta) để β không vượt biên.
        """
        try:
            from scope1 import jax_dynamics_matrix
        except Exception as e:
            raise RuntimeError("Không import được scope1.py; không thể mô phỏng.") from e

        ANG_MAX = np.deg2rad(85.0)  # biên theo thế giới
        E_REST  = 0.0               # dập khi đập biên

        y0, z0   = float(self.x_qd[0]), float(self.z_qd[0])
        phi0     = float(self.theta_d[0])
        beta0_pl = float(self.beta_d[0]) * (1.0 / self.s)

        state = np.array([y0, 0.0, z0, 0.0, phi0, 0.0, beta0_pl, 0.0], dtype=float)
        states = [state.copy()]
        cmds   = []

        for i in range(len(self.t)-1):
            meas = dict(x_q=state[0], xdot_q=state[1],
                        z_q=state[2], zdot_q=state[3],
                        theta=state[4], theta_dot=state[5],
                        beta=state[6], beta_dot=state[7])
            u1, u3, tau = self.step(i, meas)

            u2 = u3  # rất quan trọng!

            control = np.array([u1, u2, tau], dtype=float)
            state = np.array(jax_dynamics_matrix(state, control, dt=self.dt), dtype=float)

            # hard stop φ+β
            phi, beta, bdot = float(state[4]), float(state[6]), float(state[7])
            ang = phi + beta
            if ang > ANG_MAX:
                state[6] = ANG_MAX - phi
                if bdot > 0.0: state[7] = -E_REST * bdot
            elif ang < -ANG_MAX:
                state[6] = -ANG_MAX - phi
                if bdot < 0.0: state[7] = -E_REST * bdot

            states.append(state.copy())
            cmds.append([u1, u2, u3, tau])

        states, cmds = np.array(states), np.array(cmds)

        if save_csv:
            log = pd.DataFrame({
                "t": self.t[:len(cmds)],
                "u1": cmds[:,0], "u2": cmds[:,1], "u3": cmds[:,2], "tau": cmds[:,3],
                "x_q": states[:-1,0], "xdot_q": states[:-1,1],
                "z_q": states[:-1,2], "zdot_q": states[:-1,3],
                "theta": states[:-1,4], "theta_dot": states[:-1,5],
                "beta": states[:-1,6], "beta_dot": states[:-1,7],
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

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Limit-aware PD/FF controller for quad+gripper (planar).")
    p.add_argument("--flat_csv", type=str, default="flat_outputs.csv",
                   help="CSV chứa t,x_q,z_q,beta (beta: rad, quy ước 'paper').")
    p.add_argument("--beta_sign", type=int, default=+1,
                   help="+1 nếu plant dùng CCW dương; -1 nếu plant dùng CW dương.")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--save_csv", type=str, default=None)
    p.add_argument("--animate", action="store_true")
    args = p.parse_args()

    ctrl = PDFFController(flat_csv=args.flat_csv, beta_sign=args.beta_sign)
    if args.simulate:
        ctrl.simulate_with_scope1(save_csv=args.save_csv, animate=args.animate)
    else:
        print("Ready. Call PDFFController.step(i, meas) mỗi chu kỳ.")
