# qp4.py
# -*- coding: utf-8 -*-
"""
QP min-snap cho (x_q, z_q, beta) với **giới hạn góc beta hai phía**
để ngăn gripper quay quá giới hạn cơ khí ở giai đoạn sau khi gắp.
- Bổ sung cận trên:  beta <= beta_max  (hinge mềm trên lưới thời gian)
- Giữ nguyên cận dưới: beta >= beta_min
- Kẹp (clamp) các post-pick keyframes vào [beta_min, beta_max]
- (Tùy chọn) ràng buộc |beta_dot| và |beta_ddot| tối đa (hinge mềm)

LƯU Ý QUAN TRỌNG:
- QP của x và z **không đổi**, nên quỹ đạo bay của quad hầu như không thay đổi.
- Chỉ sửa quỹ đạo beta. Theo ánh xạ phẳng (IDETC'13), điều này chỉ
  ảnh hưởng feed-forward (theta, u3, tau) và vẫn đảm bảo khả năng bám x,z.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import argparse
from typing import List, Tuple

# ----------------------- Cấu hình mặc định -----------------------
# Thời điểm gắp và tổng thời gian
t_pick_default  = 2.0
t_final_default = 3.5

# Waypoints (x,z) (đơn giản, bạn có thể đổi qua CLI)
x0, z0         = -2.0,  2.0
x_pick, z_pick =  0.50, 1.35
xF,  zF        =  1.55, 1.55

# Beta đầu-cuối và ràng buộc
beta0_deg     = 20.0
beta_pick_deg = 90.0
betaF_deg     = 90.0
beta_min_deg  = 0.0           # cận dưới
beta_max_deg  = 100.0         # >>> THAY VỚI GIỚI HẠN CƠ KHÍ <<<

# Trọng số cho các ràng buộc mềm
rho_pre_los    = 2e6    # pre-pick LOS
rho_pre_shape  = 8e6    # pre-pick keyframes
rho_post       = 1e7    # post-pick shaping
rho_hinge_low  = 1e8    # phạt beta < beta_min
rho_hinge_up   = 1e8    # phạt beta > beta_max
rho_beta_rate  = 1e6    # (tùy) phạt vượt |beta_dot|
rho_beta_acc   = 5e5    # (tùy) phạt vượt |beta_ddot|

# (Tùy chọn) giới hạn tốc độ/gia tốc beta để giảm nhọn mô-men sau gắp
enable_rate_acc_limits   = True
beta_dot_max_deg_s       = 260.0   # ví dụ, ~4.5 rad/s
beta_ddot_max_deg_s2     = 5000.0  # ví dụ, ~87 rad/s^2

# Độ bậc đa thức và số đoạn (2 đoạn: [0,t_pick], [t_pick,t_final])
poly_degree = 7

# ----------------------- Các hàm tiện ích QP -----------------------
def poly_deriv_coeff(k: int, r: int) -> float:
    if k < r: 
        return 0.0
    c = 1.0
    for i in range(r):
        c *= (k - i)
    return c

def basis_row(t: float, degree: int, r: int) -> np.ndarray:
    row = np.zeros(degree + 1)
    for k in range(degree + 1):
        coeff = poly_deriv_coeff(k, r)
        row[k] = 0.0 if coeff == 0.0 else coeff * (t ** (k - r))
    return row

def segment_cost_Q(degree: int, T: float, snap_order: int = 4) -> np.ndarray:
    n = degree
    Q = np.zeros((n + 1, n + 1))
    for i in range(snap_order, n + 1):
        for j in range(snap_order, n + 1):
            ci = cj = 1.0
            for a in range(snap_order):
                ci *= (i - a)
            for b in range(snap_order):
                cj *= (j - b)
            power = i + j - 2 * snap_order + 1
            Q[i, j] = ci * cj * (T ** power) / power
    return Q

@dataclass
class Segment:
    T: float
    degree: int = poly_degree

@dataclass
class TrajectorySpec1D:
    segments: List[Segment]
    start_pos: float
    end_pos: float
    start_derivs: dict
    end_derivs: dict
    interior_positions: List[float]

def build_eq_qp_1d(spec: TrajectorySpec1D, snap_order=4, continuity_order=3):
    segs  = spec.segments
    n     = segs[0].degree
    assert len(segs) == 2, "Helper này giả định đúng 2 đoạn"
    N     = (n + 1) * 2

    H = np.zeros((N, N))
    H[0:n+1, 0:n+1] = segment_cost_Q(n, segs[0].T, snap_order=snap_order)
    H[n+1:, n+1:]   = segment_cost_Q(n, segs[1].T, snap_order=snap_order)
    f = np.zeros(N)

    Aeq_rows = []
    beq = []

    def add(si: int, tau: float, r: int, val: float):
        row = np.zeros(N); i0 = si * (n + 1)
        row[i0:i0+n+1] = basis_row(tau, n, r)
        Aeq_rows.append(row); beq.append(val)

    # start
    add(0, 0.0, 0, spec.start_pos)
    for r, v in spec.start_derivs.items():
        add(0, 0.0, r, v)

    # interior position + C^3 continuity ở t = T0
    T0 = segs[0].T
    add(0, T0, 0, spec.interior_positions[0])
    for r in range(continuity_order + 1):
        row = np.zeros(N)
        row[0:n+1]          = basis_row(T0, n, r)
        row[(n+1):2*(n+1)] -= basis_row(0.0, n, r)
        Aeq_rows.append(row); beq.append(0.0)

    # end
    add(1, segs[1].T, 0, spec.end_pos)
    for r, v in spec.end_derivs.items():
        add(1, segs[1].T, r, v)

    Aeq = np.vstack(Aeq_rows)
    beq = np.array(beq, dtype=float)
    return H, f, Aeq, beq

def solve_qp_equality(H: np.ndarray, f: np.ndarray, Aeq: np.ndarray, beq: np.ndarray, reg: float = 1e-10) -> np.ndarray:
    N = H.shape[0]
    Hreg = H + reg * np.eye(N)
    KKT  = np.block([[Hreg, Aeq.T], [Aeq, np.zeros((Aeq.shape[0], Aeq.shape[0]))]])
    rhs  = np.concatenate([-f, beq])
    sol  = np.linalg.solve(KKT, rhs)
    c    = sol[:N]
    return c

def add_soft_samples_to_objective(H, f, rows, values, weights):
    if len(rows) == 0:
        return H, f
    R = np.vstack(rows)
    y = np.array(values, dtype=float)
    W = np.diag(np.sqrt(np.asarray(weights, dtype=float)))
    RtWR = (W @ R).T @ (W @ R)
    RtWy = (W @ R).T @ (W @ y)
    H2 = H + 2.0 * RtWR
    f2 = f - 2.0 * RtWy
    return H2, f2

# ----------------------- Evaluate helpers -----------------------
def evaluate_piecewise(coeffs: np.ndarray, segs: List[Segment], t_query: np.ndarray, degree: int = poly_degree) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = degree
    cpseg = n + 1
    starts = [0.0, segs[0].T]
    y   = np.zeros_like(t_query, dtype=float)
    yd  = np.zeros_like(t_query, dtype=float)
    ydd = np.zeros_like(t_query, dtype=float)
    yddd= np.zeros_like(t_query, dtype=float)
    for i, t in enumerate(t_query):
        sidx = 0 if t <= starts[1] else 1
        tau  = t - starts[sidx]
        a    = coeffs[sidx*cpseg:(sidx+1)*cpseg]
        y[i]    = basis_row(tau, n, 0).dot(a)
        yd[i]   = basis_row(tau, n, 1).dot(a)
        ydd[i]  = basis_row(tau, n, 2).dot(a)
        yddd[i] = basis_row(tau, n, 3).dot(a)
    return y, yd, ydd, yddd

# ----------------------- Builder chính -----------------------
def build_flat_trajectories(t_pick: float, t_final: float,
                            beta_min_deg_local: float, beta_max_deg_local: float,
                            out_csv: str = "flat_outputs.csv",
                            plot: bool = False):
    n = poly_degree
    cpseg = n + 1

    segs = [Segment(T=t_pick, degree=n), Segment(T=t_final - t_pick, degree=n)]

    # ---------- x,z: QP min-snap (yêu cầu C^3) ----------
    x_spec = TrajectorySpec1D(
        segs, start_pos=x0, end_pos=xF,
        start_derivs={1:0.0,2:0.0,3:0.0},
        end_derivs  ={1:0.0,2:0.0,3:0.0},
        interior_positions=[x_pick]
    )
    z_spec = TrajectorySpec1D(
        segs, start_pos=z0, end_pos=zF,
        start_derivs={1:0.0,2:0.0,3:0.0},
        end_derivs  ={1:0.0,2:0.0,3:0.0},
        interior_positions=[z_pick]
    )

    Hx, fx, Ax, bx = build_eq_qp_1d(x_spec)
    Hz, fz, Az, bz = build_eq_qp_1d(z_spec)
    cx = solve_qp_equality(Hx, fx, Ax, bx)
    cz = solve_qp_equality(Hz, fz, Az, bz)

    # ---------- beta: QP + shaping + hinge hai phía ----------
    beta0     = np.deg2rad(beta0_deg)
    beta_pick = np.deg2rad(beta_pick_deg)
    betaF     = np.deg2rad(betaF_deg)

    beta_spec = TrajectorySpec1D(
        segs, start_pos=beta0, end_pos=betaF,
        start_derivs={1:0.0,2:0.0,3:0.0},
        end_derivs  ={1:0.0,2:0.0,3:0.0},
        interior_positions=[beta_pick]
    )
    Hb, fb, Ab, bb = build_eq_qp_1d(beta_spec)

    beta_min = np.deg2rad(beta_min_deg_local)
    beta_max = np.deg2rad(beta_max_deg_local)

    # ----- (1) Pre-pick LOS: beta ~= atan2(z_q - z_pick, x_pick - x_q) -----
    rows = []; vals = []; wts = []
    t_pre = np.linspace(0.20, t_pick - 0.004, 60)
    for tau in t_pre:
        x_tau = basis_row(tau, n, 0).dot(cx[0:cpseg])
        z_tau = basis_row(tau, n, 0).dot(cz[0:cpseg])
        los   = np.arctan2(z_tau - z_pick, x_pick - x_tau)  # dấu đúng quy ước 'paper'
        row   = np.zeros(2 * (n + 1)); row[0:cpseg] = basis_row(tau, n, 0)
        rows.append(row); vals.append(los); wts.append(rho_pre_los)

    # ----- (2) Pre-pick shaping keyframes -----
    pre_abs = np.array([t_pick - 1.20, t_pick - 0.90, t_pick - 0.60, t_pick - 0.30, t_pick - 0.08])
    pre_deg = np.array([28, 40, 32, 18, 12])
    for t_abs, val_deg in zip(pre_abs, pre_deg):
        tau = t_abs
        if tau <= 0.0:
            continue
        row = np.zeros(2 * (n + 1)); row[0:cpseg] = basis_row(tau, n, 0)
        rows.append(row); vals.append(np.deg2rad(val_deg)); wts.append(rho_pre_shape)

    # ----- (3) Post-pick shaping (KẸP vào [beta_min, beta_max]) -----
    post_abs = np.array([t_pick + 0.02, t_pick + 0.08, t_pick + 0.14, t_pick + 0.20, t_pick + 0.26,
                         t_pick + 0.34, t_pick + 0.50, t_pick + 0.80, t_pick + 1.20, t_pick + 1.50])
    post_deg = np.array([95, 105, 112, 120, 124, 125, 120, 105, 92, 90])
    post_deg = np.clip(post_deg, beta_min_deg_local, beta_max_deg_local)  # CLAMP

    for t_abs, val_deg in zip(post_abs, post_deg):
        tau = t_abs - t_pick
        row = np.zeros(2 * (n + 1)); row[cpseg:2*cpseg] = basis_row(tau, n, 0)
        rows.append(row); vals.append(np.deg2rad(val_deg)); wts.append(rho_post)

    # Gộp vào objective mềm và giải lần 1
    Hb_soft, fb_soft = add_soft_samples_to_objective(Hb, fb, rows, vals, wts)
    cb = solve_qp_equality(Hb_soft, fb_soft, Ab, bb)

    # ----- Helper eval beta và đạo hàm trên lưới -----
    def eval_beta_and_derivs(coeffs: np.ndarray, t: np.ndarray, r: int):
        out = np.zeros_like(t, dtype=float)
        for i, tt in enumerate(t):
            if tt <= t_pick:
                tau = tt
                segc = coeffs[0:cpseg]
            else:
                tau = tt - t_pick
                segc = coeffs[cpseg:2*cpseg]
            out[i] = basis_row(tau, n, r).dot(segc)
        return out

    # ----- (4) Hinge hai phía cho beta (>= beta_min, <= beta_max) -----
    grid = np.linspace(0.0, t_final, 1600)
    for _ in range(8):
        b_val = eval_beta_and_derivs(cb, grid, r=0)

        viol_low  = np.where(b_val < beta_min - 1e-9)[0]
        viol_high = np.where(b_val > beta_max + 1e-9)[0]
        if len(viol_low) + len(viol_high) == 0:
            break

        rows_h = []; vals_h = []; wts_h = []
        # lower
        for idx in viol_low[::2]:
            tt = grid[idx]
            row = np.zeros(2*(n+1))
            if tt <= t_pick:
                row[0:cpseg] = basis_row(tt, n, 0)
            else:
                row[cpseg:2*cpseg] = basis_row(tt - t_pick, n, 0)
            rows_h.append(row); vals_h.append(beta_min); wts_h.append(rho_hinge_low)

        # upper
        for idx in viol_high[::2]:
            tt = grid[idx]
            row = np.zeros(2*(n+1))
            if tt <= t_pick:
                row[0:cpseg] = basis_row(tt, n, 0)
            else:
                row[cpseg:2*cpseg] = basis_row(tt - t_pick, n, 0)
            rows_h.append(row); vals_h.append(beta_max); wts_h.append(rho_hinge_up)

        Hb_soft, fb_soft = add_soft_samples_to_objective(Hb_soft, fb_soft, rows_h, vals_h, wts_h)
        cb = solve_qp_equality(Hb_soft, fb_soft, Ab, bb)

    # ----- (5) (Tùy chọn) Hinge cho tốc độ & gia tốc beta -----
    if enable_rate_acc_limits:
        beta_dot_max  = np.deg2rad(beta_dot_max_deg_s)
        beta_ddot_max = np.deg2rad(beta_ddot_max_deg_s2)
        for _ in range(4):
            bdot  = eval_beta_and_derivs(cb, grid, r=1)
            bddot = eval_beta_and_derivs(cb, grid, r=2)

            rows_h = []; vals_h = []; wts_h = []

            # |beta_dot| <= beta_dot_max
            viol_up = np.where(bdot >  beta_dot_max + 1e-9)[0]
            viol_lo = np.where(bdot < -beta_dot_max - 1e-9)[0]
            for idx in viol_up[::2]:
                tt = grid[idx]
                row = np.zeros(2*(n+1))
                if tt <= t_pick: row[0:cpseg] = basis_row(tt, n, 1)
                else:            row[cpseg:2*cpseg] = basis_row(tt - t_pick, n, 1)
                rows_h.append(row); vals_h.append(beta_dot_max); wts_h.append(rho_beta_rate)
            for idx in viol_lo[::2]:
                tt = grid[idx]
                row = np.zeros(2*(n+1))
                if tt <= t_pick: row[0:cpseg] = basis_row(tt, n, 1)
                else:            row[cpseg:2*cpseg] = basis_row(tt - t_pick, n, 1)
                rows_h.append(row); vals_h.append(-beta_dot_max); wts_h.append(rho_beta_rate)

            # |beta_ddot| <= beta_ddot_max
            viol_up = np.where(bddot >  beta_ddot_max + 1e-9)[0]
            viol_lo = np.where(bddot < -beta_ddot_max - 1e-9)[0]
            for idx in viol_up[::2]:
                tt = grid[idx]
                row = np.zeros(2*(n+1))
                if tt <= t_pick: row[0:cpseg] = basis_row(tt, n, 2)
                else:            row[cpseg:2*cpseg] = basis_row(tt - t_pick, n, 2)
                rows_h.append(row); vals_h.append(beta_ddot_max); wts_h.append(rho_beta_acc)
            for idx in viol_lo[::2]:
                tt = grid[idx]
                row = np.zeros(2*(n+1))
                if tt <= t_pick: row[0:cpseg] = basis_row(tt, n, 2)
                else:            row[cpseg:2*cpseg] = basis_row(tt - t_pick, n, 2)
                rows_h.append(row); vals_h.append(-beta_ddot_max); wts_h.append(rho_beta_acc)

            if len(rows_h) == 0:
                break
            Hb_soft, fb_soft = add_soft_samples_to_objective(Hb_soft, fb_soft, rows_h, vals_h, wts_h)
            cb = solve_qp_equality(Hb_soft, fb_soft, Ab, bb)

    # ---------- Evaluate & Lưu CSV ----------
    t = np.linspace(0.0, t_final, 800)
    x, xd, xdd, _ = evaluate_piecewise(cx, segs, t)
    z, zd, zdd, _ = evaluate_piecewise(cz, segs, t)
    b, bd, bdd, _ = evaluate_piecewise(cb, segs, t)

    df = pd.DataFrame({
        "t": t,
        "x_q": x,
        "z_q": z,
        "beta": b,
        "xd_q": xd,
        "zd_q": zd,
        "betad": bd
    })
    df.to_csv(out_csv, index=False)
    print(f"[qp4] Saved flat outputs -> {out_csv}  (beta in [deg]: min={np.rad2deg(b).min():.1f}, max={np.rad2deg(b).max():.1f})")

    if plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.plot(t, np.rad2deg(b), label=r"$\beta^d$")
        ax1.axhline(beta_min_deg_local, linestyle="--")
        ax1.axhline(beta_max_deg_local, linestyle="--")
        ax1.axvline(t_pick, linestyle=":")
        ax1.set_xlabel("t (s)"); ax1.set_ylabel("deg"); ax1.set_title("beta(t) with bounds"); ax1.grid(True, alpha=0.5); ax1.legend()

        ax2.plot(t, x, label="x_q^d")
        ax2.plot(t, z, label="z_q^d")
        ax2.axvline(t_pick, linestyle=":")
        ax2.set_xlabel("t (s)"); ax2.set_ylabel("m"); ax2.set_title("quad position"); ax2.grid(True, alpha=0.5); ax2.legend()
        fig.tight_layout(); plt.show()

    return df

# ----------------------- CLI -----------------------
def main():
    parser = argparse.ArgumentParser(description="QP4: Min-snap với giới hạn beta hai phía.")
    parser.add_argument("--t_pick", type=float, default=t_pick_default, help="Thời điểm gắp (s).")
    parser.add_argument("--t_final", type=float, default=t_final_default, help="Thời gian kết thúc (s).")
    parser.add_argument("--beta_min_deg", type=float, default=beta_min_deg, help="Cận dưới beta (deg).")
    parser.add_argument("--beta_max_deg", type=float, default=beta_max_deg, help="Cận trên beta (deg).")
    parser.add_argument("--out_csv", type=str, default="flat_outputs.csv", help="Đường dẫn CSV xuất ra.")
    parser.add_argument("--plot", action="store_true", help="Hiển thị biểu đồ.")
    args = parser.parse_args()

    build_flat_trajectories(args.t_pick, args.t_final, args.beta_min_deg, args.beta_max_deg, out_csv=args.out_csv, plot=args.plot)

if __name__ == "__main__":
    main()
