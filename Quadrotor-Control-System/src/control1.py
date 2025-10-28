# controller_pd_ff.py
# -*- coding: utf-8 -*-
"""
PD + Feed-forward controller for quadrotor-with-arm (planar) tracking a min-snap trajectory,
faithful to the IDETC'13 paper:
  u1_c = kpz (z_q^d - z_q) + kdz (zdot_q^d - zdot_q) + u1^d
  theta_c = asin( kpx (x_q^d - x_q) + kdx (xdot_q^d - xdot_q) ) + theta^d
  u3_c = kpθ (theta_c - θ) + kdθ (thetadot^d - θdot) + u3^d
and the gripper torque is commanded as:
  tau_c = kβ (β^d - β) + kdβ (βdot^d - βdot) + τ^d
Feed-forward terms (u1^d, u3^d, τ^d, θdot^d, θddot^d, x_g, z_g) come from tranfer.recover_inputs_from_flat,
which implements the Appendix (16)–(31) mapping.

If scope1.py is available, a simple simulation harness is provided (optional).
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd

# === Use your existing transform ===
import tranfer as tf  # <-- file name you provided: tranfer.py (not "transfer")

# ---------- Small helpers ----------
def finite_diff(t: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Central finite-difference of given order."""
    out = y.astype(float).copy()
    for _ in range(order):
        out = np.gradient(out, t, edge_order=2)
    return out

def theta_from_flat(t: np.ndarray, x_q: np.ndarray, z_q: np.ndarray, beta: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Compute θ^d from flat outputs using b3 = normalize(r¨_s + g e3) (planar x–z),
    θ^d = atan2(b3_x, b3_z). Matches (16)–(18) in the paper.
    """
    m_q, m_g, L_g, g = params["m_q"], params["m_g"], params["L_g"], params["g"]
    m_s = m_q + m_g

    x_g = x_q + L_g * np.cos(beta)         # Eq. (1)
    z_g = z_q - L_g * np.sin(beta)

    x_s = (m_q * x_q + m_g * x_g) / m_s    # Eq. (15)
    z_s = (m_q * z_q + m_g * z_g) / m_s

    x_s_dd = finite_diff(t, x_s, 2)
    z_s_dd = finite_diff(t, z_s, 2)

    # v = r¨_s + g e3  in 2D -> [x_s_dd, z_s_dd + g]
    vx = x_s_dd
    vz = z_s_dd + g
    return np.arctan2(vx, vz)              # Eq. (18) -> θ from b3

@dataclass
class Gains:
    # Outer-loop (position -> attitude)
    kpx: float = 1.2
    kdx: float = 0.6
    # Altitude loop
    kpz: float = 10.0
    kdz: float = 5.5
    # Inner attitude loop
    kp_theta: float = 6.0
    kd_theta: float = 2.5
    # Arm (β) loop
    kp_beta: float = 4.0
    kd_beta: float = 1.2

class PDFFController:
    def __init__(self, flat_csv: str, params: Optional[Dict[str, float]] = None, gains: Optional[Gains] = None):
        self.params: Dict[str, float] = dict(tf.PARAMS if params is None else params)
        self.gains = gains if gains is not None else Gains()

        # Load flat outputs from planner (t, x_q, z_q, beta) — your CSV
        df = pd.read_csv(flat_csv)
        for col in ("t", "x_q", "z_q", "beta"):
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {flat_csv}")
        self.t    = df["t"].to_numpy(dtype=float)
        self.x_qd = df["x_q"].to_numpy(dtype=float)
        self.z_qd = df["z_q"].to_numpy(dtype=float)
        self.beta_d = df["beta"].to_numpy(dtype=float)  # rad

        # Desired derivatives for PD
        self.xdot_qd = finite_diff(self.t, self.x_qd, 1)
        self.zdot_qd = finite_diff(self.t, self.z_qd, 1)
        self.betadot_d = finite_diff(self.t, self.beta_d, 1)

        # Feed-forward terms (u1^d, u3^d, tau^d, thetadot^d, thetaddot^d, ...)
        ff = tf.recover_inputs_from_flat(self.t, self.x_qd, self.z_qd, self.beta_d, self.params)
        self.u1_d   = ff["u1"].astype(float)
        self.u3_d   = ff["u3"].astype(float)
        self.tau_d  = ff["tau"].astype(float)
        self.theta_dot_d = ff["theta_dot"].astype(float)

        # θ^d from flat mapping (b3 direction)
        self.theta_d = theta_from_flat(self.t, self.x_qd, self.z_qd, self.beta_d, self.params)

        # Precompute dt
        self.dt = float(np.mean(np.diff(self.t)))

    def step(self, i: int, meas: Dict[str, float]):
        """
        One control step at time index i.
        meas must provide:
            x_q, z_q, xdot_q, zdot_q, theta, theta_dot, beta, beta_dot
        Returns: u1_cmd, u3_cmd, tau_cmd
        """
        g = self.gains

        # Desired at step i
        x_d, z_d   = self.x_qd[i], self.z_qd[i]
        xd_d, zd_d = self.xdot_qd[i], self.zdot_qd[i]
        th_d, thd_d = self.theta_d[i], self.theta_dot_d[i]
        beta_d, betad_d = self.beta_d[i], self.betadot_d[i]
        u1_ff, u3_ff, tau_ff = self.u1_d[i], self.u3_d[i], self.tau_d[i]

        # Errors
        ex   = x_d - float(meas["x_q"])
        ez   = z_d - float(meas["z_q"])
        exd  = xd_d - float(meas["xdot_q"])
        ezd  = zd_d - float(meas["zdot_q"])
        eth  = th_d - float(meas["theta"])
        ethd = thd_d - float(meas["theta_dot"])
        eb   = beta_d - float(meas["beta"])
        ebd  = betad_d - float(meas["beta_dot"])

        # --- Eq. (11): thrust command with PD + FF ---
        u1_c = g.kpz * ez + g.kdz * ezd + u1_ff

        # --- Eq. (13): lateral position -> attitude command ---
        lat_cmd = g.kpx * ex + g.kdx * exd
        lat_cmd = float(np.clip(lat_cmd, -0.999, 0.999))  # safe asin
        theta_c = np.arcsin(lat_cmd) + th_d

        # --- Eq. (12): moment command with PD + FF ---
        u3_c = g.kp_theta * (theta_c - float(meas["theta"])) + g.kd_theta * ethd + u3_ff

        # --- Arm torque: PD + FF around β (not explicitly written in paper, but standard) ---
        tau_c = g.kp_beta * eb + g.kd_beta * ebd + tau_ff

        return float(u1_c), float(u3_c), float(tau_c)

    # ---------------- Optional: simple simulation using scope1.py ----------------
    def simulate_with_scope1(self, save_csv: Optional[str] = None, animate: bool = False):
        """
        Minimal harness to run a planar sim with scope1's dynamics.
        Mapping: scope1 uses state = [y, y_dot, z, z_dot, phi, phi_dot, beta, beta_dot]
                 Our x_q <-> scope1.y ; theta <-> phi.
                 scope1 uses equation J_q*phi_ddot = (u2 - tau). To apply body moment u3, send u2 = u3 + tau.
        """
        try:
            from scope1 import jax_dynamics_matrix
        except Exception as e:
            raise RuntimeError("scope1.py not importable; cannot simulate.") from e

        # Initial state at t0: place the robot on the desired initial pose
        y0   = self.x_qd[0]
        z0   = self.z_qd[0]
        phi0 = self.theta_d[0]       # θ^d(t0)
        beta0 = self.beta_d[0]
        state = np.array([y0, 0.0, z0, 0.0, phi0, 0.0, beta0, 0.0], dtype=float)

        states = [state.copy()]
        cmds   = []  # (u1, u2, u3, tau)

        for i in range(len(self.t)-1):
            # Build a measurement dict from current state (scope1.y -> x_q)
            meas = dict(
                x_q=state[0],     xdot_q=state[1],
                z_q=state[2],     zdot_q=state[3],
                theta=state[4],   theta_dot=state[5],
                beta=state[6],    beta_dot=state[7],
            )
            u1, u3, tau = self.step(i, meas)

            # IMPORTANT for scope1: use u2 = u3 + tau so that (u2 - tau) = u3
            u2 = u3 + tau

            # Integrate one step
            control = np.array([u1, u2, tau], dtype=float)
            state = np.array(jax_dynamics_matrix(state, control, dt=self.dt), dtype=float)

            states.append(state.copy())
            cmds.append([u1, u2, u3, tau])

        states = np.array(states)
        cmds   = np.array(cmds)

        if save_csv:
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
            # Optionally re-use scope1.animate by packaging states/controls appropriately
            try:
                from scope1 import animate as scope1_animate
                scope1_animate(states, cmds[:, :3], target=(self.x_qd[-1], self.z_qd[-1]), dt=self.dt)
            except Exception as e:
                print(f"Animation failed: {e}")

        return states, cmds


# ----------------------------- CLI usage -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PD + FF quad controller (planar).")
    parser.add_argument("--flat_csv", type=str, default="flat_outputs.csv",
                        help="Planner output CSV with columns: t,x_q,z_q,beta (beta in rad).")
    parser.add_argument("--simulate", action="store_true", help="Run a quick planar sim using scope1.py.")
    parser.add_argument("--save_csv", type=str, default=None, help="If set, save sim log to this CSV.")
    parser.add_argument("--animate", action="store_true", help="Show animation (scope1).")
    args = parser.parse_args()

    ctrl = PDFFController(flat_csv=args.flat_csv)

    if args.simulate:
        ctrl.simulate_with_scope1(save_csv=args.save_csv, animate=args.animate)
    else:
        print("Controller ready. Call PDFFController.step(i, meas) each control tick.")
