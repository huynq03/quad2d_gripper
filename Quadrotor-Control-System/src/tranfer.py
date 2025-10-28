# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict
# quad params
PARAMS = dict(
    m_q = 0.500,     # kg
    m_g = 0.158,     # kg
    J_q = 1.2e-2,    # kg*m^2
    J_g = 1.0e-3,    # kg*m^2
    L_g = 0.105,     # m
    g   = 9.81,      # m/s^2
)

# ==============================
# HÀM PHỤ
# ==============================
def _finite_diff(t: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    out = y.astype(float).copy()
    for _ in range(order):
        out = np.gradient(out, t, edge_order=2)
    return out

def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

# ==============================
# CHUYỂN (x_q, z_q, beta) -> (u1, u3, tau)
# ==============================
def recover_inputs_from_flat(
    t: np.ndarray,
    x_q: np.ndarray,
    z_q: np.ndarray,
    beta: np.ndarray,
    params: Dict[str, float],
):
    m_q = float(params["m_q"])
    m_g = float(params["m_g"])
    J_q = float(params["J_q"])
    J_g = float(params["J_g"])
    L_g = float(params["L_g"])
    g   = float(params["g"])
    m_s = m_q + m_g

    # Vị trí gripper (Eq. (1))
    x_g = x_q + L_g * np.cos(beta)
    z_g = z_q - L_g * np.sin(beta)

    # Tâm khối hệ (Eq. (15))
    x_s = (m_q * x_q + m_g * x_g) / m_s
    z_s = (m_q * z_q + m_g * z_g) / m_s

    # Đạo hàm
    x_s_dd  = _finite_diff(t, x_s, 2)
    z_s_dd  = _finite_diff(t, z_s, 2)
    x_s_ddd = _finite_diff(t, x_s, 3)
    z_s_ddd = _finite_diff(t, z_s, 3)
    x_s_4   = _finite_diff(t, x_s, 4)
    z_s_4   = _finite_diff(t, z_s, 4)

    # Nhúng phẳng x–z vào 3D
    a_s = np.stack([x_s_dd,  np.zeros_like(x_s_dd),  z_s_dd ], axis=-1)
    j_s = np.stack([x_s_ddd, np.zeros_like(x_s_ddd), z_s_ddd], axis=-1)
    s4  = np.stack([x_s_4,   np.zeros_like(x_s_4),   z_s_4  ], axis=-1)
    e3  = np.array([0.0, 0.0, 1.0])

    # u1, b3 (Eqs. (16)–(18))
    v  = a_s + g * e3
    u1 = (m_s * np.linalg.norm(v, axis=-1)).astype(float)
    b3 = _normalize(v)

    theta = np.arctan2(b3[:, 0], b3[:, 2])
    
    # b2 = e2; b1 = b2 × b3
    b2 = np.array([0.0, 1.0, 0.0])
    b1 = np.cross(b2.reshape(1, 3), b3)

    # u1_dot, theta_dot, theta_ddot (Eqs. (20),(22),(25))
    u1_dot = np.einsum('ij,ij->i', b3, m_s * j_s)
    theta_dot = (m_s / np.maximum(u1, 1e-9)) * np.einsum('ij,ij->i', b1, j_s)
    theta_ddot = (
        (m_s * np.einsum('ij,ij->i', b1, s4)) - 2.0 * u1_dot * theta_dot
    ) / np.maximum(u1, 1e-9)

    # Torque gripper tau (Eq. (30))
    x_g_dd  = _finite_diff(t, x_g, 2)
    z_g_dd  = _finite_diff(t, z_g, 2)
    beta_dd = _finite_diff(t, beta, 2)
    tau = J_g * beta_dd - L_g * m_g * (x_g_dd * np.sin(beta) + (z_g_dd + g) * np.cos(beta))

    # u3 (Eq. (31))
    u3 = J_q * theta_ddot + tau

    return {
        "u1": u1,
        "u3": u3,
        "tau": tau,
        "theta": theta,
        "theta_dot": theta_dot,
        "theta_ddot": theta_ddot,
        "x_g": x_g,
        "z_g": z_g,
    }

# ==============================
# MAIN: đọc flat_outputs.csv và xuất inputs_from_flat.csv
# ==============================
if __name__ == "__main__":
    flat_csv = "/home/huynq/Project20251/Quadrotor-Control-System/src/minsnap_results/flat_outputs.csv"      # Đường dẫn đầu vào
    out_csv  = "/home/huynq/Project20251/Quadrotor-Control-System/src/minsnap_results/inputs_from_flat.csv"  # Đường dẫn đầu ra

    # Đọc dữ liệu phẳng (beta phải là radian)
    df = pd.read_csv(flat_csv)
    for col in ("t", "x_q", "z_q", "beta"):
        if col not in df.columns:
            raise ValueError(f"Thiếu cột '{col}' trong {flat_csv}")
    t    = df["t"].to_numpy()
    x_q  = df["x_q"].to_numpy()
    z_q  = df["z_q"].to_numpy()
    beta = df["beta"].to_numpy()   # rad

    res = recover_inputs_from_flat(t, x_q, z_q, beta, PARAMS)

    out_df = pd.DataFrame({
        "t": t,
        "u1": res["u1"],
        "u3": res["u3"],
        "tau": res["tau"],
        "theta": res["theta"], 
        "theta_dot": res["theta_dot"],
        "theta_ddot": res["theta_ddot"],
        "x_g": res["x_g"],
        "z_g": res["z_g"],
    })
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] Đã lưu u1/u3/tau/theta -> {out_csv}")  
