# -*- coding: utf-8 -*-
"""
Minimum-snap QP (deg 7), 2 đoạn (start->pickup->end), 3 trục (x, z, beta).
- Cost: (1/2) c^T Q c với Q từ ∫(d^4/dt^4)^2
- Constraints: đẳng thức (biên đầu/cuối, C^3 tại pickup), + các anchor cho beta trước pickup, + beta(Δ1)=90°
- Solve: hệ KKT, dùng numpy.linalg.lstsq (ổn định khi Q bán xác định)
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple  # <-- dùng typing cũ cho 3.7/3.8/3.9

# =================== CẤU HÌNH (THAY TÙY Ý) ===================
n = 7                                  # bậc đa thức (>=7)
durations = [2.0, 1.5]                 # Δ1 (start->pickup), Δ2 (pickup->end) [s]
Delta1, Delta2 = float(durations[0]), float(durations[1])

# Biên đầu/cuối cho [x, z, beta] (beta nhập theo độ cho dễ nhìn)
deg = np.deg2rad
y0 = np.array([0.0, 0.0, deg(20.0)])   # beta0 ≈ 20°
yT = np.array([1.0, 0.2, deg(90.0)])   # beta cuối ≈ 90°
beta_pickup: Optional[float] = deg(90.0)  # ép beta(Δ1) = 90°
# Các anchor cho beta TRƯỚC pickup (t < Δ1), theo độ:
beta_anchors_pre_pickup: List[Tuple[float, float]] = [
    (0.5,  25.0),
    (1.0,  40.0),
    (1.5,  15.0),
    (1.9,  -5.0),
]
# =============================================================

axes = 3       # x, z, beta
segments = 2

def falling(k: int, r: int) -> float:
    if r == 0: return 1.0
    if k < r:  return 0.0
    out = 1.0
    for m in range(r):
        out *= (k - m)
    return out

def deriv_row(n: int, r: int, tau: float) -> np.ndarray:
    row = np.zeros(n+1)
    for k in range(r, n+1):
        row[k] = falling(k, r) * (tau ** (k - r))
    return row

def Q_block_min_snap(n: int, Delta: float) -> np.ndarray:
    Q = np.zeros((n+1, n+1))
    for i in range(4, n+1):
        ai = falling(i, 4)
        for j in range(4, n+1):
            aj = falling(j, 4)
            power = i + j - 7  # ∫ t^{i+j-8} dt = t^{i+j-7}/(i+j-7)
            Q[i, j] = 2.0 * ai * aj * (Delta ** power) / power
    return Q

def idx(axis: int, seg: int, n: int) -> int:
    return ((axis * segments) + seg) * (n + 1)

def assemble_Q(n: int, durations) -> np.ndarray:
    nvar = axes * segments * (n + 1)
    Q = np.zeros((nvar, nvar))
    for axis in range(axes):
        for seg, Delta in enumerate(durations):
            b = idx(axis, seg, n)
            Q[b:b+n+1, b:b+n+1] = Q_block_min_snap(n, float(Delta))
    return Q

def assemble_constraints(n: int,
                         y0: np.ndarray,
                         yT: np.ndarray,
                         Delta1: float,
                         Delta2: float,
                         beta_pickup: Optional[float],
                         beta_anchors_pre_pickup: List[Tuple[float, float]]
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo A, b cho các ràng buộc đẳng thức:
      - Start (seg0, tau=0): pos=y0; vel=acc=jerk=0
      - End   (seg1, tau=Δ2): pos=yT; vel=acc=jerk=0
      - C^3 continuity tại pickup (seg0 tau=Δ1 = seg1 tau=0) cho cả 3 trục
      - Optional: beta(Δ1)=beta_pickup
      - Optional: các anchor cho beta trước pickup: beta(t_i)=value_i (deg)
    """
    nvar = axes * segments * (n + 1)
    Arows, brows = [], []

    # Biên đầu
    for axis in range(axes):
        base = idx(axis, 0, n)
        row = np.zeros(nvar); row[base:base+n+1] = deriv_row(n, 0, 0.0)
        Arows.append(row); brows.append(float(y0[axis]))
        for r in (1, 2, 3):
            row = np.zeros(nvar); row[base:base+n+1] = deriv_row(n, r, 0.0)
            Arows.append(row); brows.append(0.0)

    # Biên cuối
    for axis in range(axes):
        base = idx(axis, 1, n)
        row = np.zeros(nvar); row[base:base+n+1] = deriv_row(n, 0, Delta2)
        Arows.append(row); brows.append(float(yT[axis]))
        for r in (1, 2, 3):
            row = np.zeros(nvar); row[base:base+n+1] = deriv_row(n, r, Delta2)
            Arows.append(row); brows.append(0.0)

    # C^3 tại pickup
    for axis in range(axes):
        b0 = idx(axis, 0, n)
        b1 = idx(axis, 1, n)
        for r in (0, 1, 2, 3):
            row = np.zeros(nvar)
            row[b0:b0+n+1] = deriv_row(n, r, Delta1)
            row[b1:b1+n+1] -= deriv_row(n, r, 0.0)
            Arows.append(row); brows.append(0.0)

    # beta(Δ1)
    if beta_pickup is not None:
        row = np.zeros(nvar)
        b_beta0 = idx(2, 0, n)
        row[b_beta0:b_beta0+n+1] = deriv_row(n, 0, Delta1)
        Arows.append(row); brows.append(float(beta_pickup))

    # Anchors cho beta trước pickup
    for (tmark, beta_deg_val) in beta_anchors_pre_pickup:
        if 0.0 < tmark < Delta1:
            row = np.zeros(nvar)
            b_beta0 = idx(2, 0, n)
            row[b_beta0:b_beta0+n+1] = deriv_row(n, 0, float(tmark))
            Arows.append(row); brows.append(float(np.deg2rad(beta_deg_val)))

    A = np.vstack(Arows)
    b = np.array(brows)
    return A, b

def solve_kkt(Q: np.ndarray, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Giải hệ KKT:
      [ Q  A^T ] [c]   [0]
      [ A   0  ] [λ] = [b]
    """
    nvar = Q.shape[0]
    m = A.shape[0]
    KKT = np.block([[Q, A.T],
                    [A, np.zeros((m, m))]])
    rhs = np.concatenate([np.zeros(nvar), b])
    sol, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
    c = sol[:nvar]
    lam = sol[nvar:]
    return c, lam

def coeffs_of(c: np.ndarray, axis: int, seg: int, n: int) -> np.ndarray:
    b = idx(axis, seg, n)
    return c[b:b+n+1]

def eval_poly(coeffs: np.ndarray, tau: float) -> float:
    s, p = 0.0, 1.0
    for ak in coeffs:
        s += ak * p
        p *= tau
    return s

def eval_piecewise(c: np.ndarray, axis: int, t: float) -> float:
    if t <= Delta1:
        return eval_poly(coeffs_of(c, axis, 0, n), t)
    else:
        return eval_poly(coeffs_of(c, axis, 1, n), t-Delta1)

def main():
    Q = assemble_Q(n, durations)
    A, b = assemble_constraints(n, y0, yT, Delta1, Delta2, beta_pickup, beta_anchors_pre_pickup)

    c, lam = solve_kkt(Q, A, b)

    res_stationarity = np.linalg.norm(Q @ c + A.T @ lam)
    res_primal = np.linalg.norm(A @ c - b)
    print("Solved. Vars:", Q.shape[0], "| Constraints:", A.shape[0])
    print("KKT residuals: ||Qc+A^Tλ||=%.3e, ||Ac-b||=%.3e" % (res_stationarity, res_primal))

    # Sample
    T_total = Delta1 + Delta2
    T = np.linspace(0.0, T_total, 351)
    x = np.array([eval_piecewise(c, 0, t) for t in T])
    z = np.array([eval_piecewise(c, 1, t) for t in T])
    beta = np.array([eval_piecewise(c, 2, t) for t in T])

    outdir = Path("./minsnap_results")
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "minsnap_qp_solution.npz",
             n=n, durations=np.array(durations),
             y0=y0, yT=yT,
             beta_pickup=np.array([beta_pickup]) if beta_pickup is not None else np.array([]),
             beta_anchors_pre_pickup=np.array(beta_anchors_pre_pickup, dtype=float),
             Q=Q, A=A, b=b, c=c, lam=lam,
             T=T, x=x, z=z, beta=beta,
             res_stationarity=res_stationarity, res_primal=res_primal)
    print("Saved:", (outdir / "minsnap_qp_solution.npz").resolve())

    # Plot (nếu có matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def plot_series(T, Y, title, ylabel, fname):
            plt.figure()
            plt.plot(T, Y)
            plt.title(title)
            plt.xlabel("t [s]"); plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(outdir / fname); plt.close()

        plot_series(T, x,    "x(t) - minimum snap",   "x", "x_traj.png")
        plot_series(T, z,    "z(t) - minimum snap",   "z", "z_traj.png")
        plot_series(T, np.rad2deg(beta), "beta(t) [deg] - with anchors", "beta [deg]", "beta_traj_deg.png")
        print("Saved plots to:", outdir.resolve())
    except Exception as e:
        print("Plotting skipped:", e)

if __name__ == "__main__":
    main()
