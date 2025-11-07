# qp3 (1).py  — TẠO QUỸ ĐẠO MIN-SNAP CHO x,z VÀ β
# (đã sửa: LOS đúng dấu + thêm pre-pick shaping để có "uốn lượn" như Fig.6)

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ==============================
# Parameters you likely want to edit
# ==============================

# Times
t_pick = 2.0
t_final = 3.5
# Waypoints (approximate Fig. 5)
x0, z0         = -2.0,  2.0
x_pick, z_pick =  0.50, 1.35
xF, zF         =  1.55, 1.55

# Beta boundary & shaping
beta0_deg      = 20.0           # ~20° ở t=0 như Fig.6
beta_pick_deg  = 90.0           # vertical tại lúc gắp
betaF_deg      = 90.0           # về ~90° ở cuối
beta_min_deg   = 0.0            # cho phép tiến sát 0° ngay trước gắp

# Trọng số
rho_pre_los   = 2e6   # LOS trước gắp
rho_pre_shape = 8e6   # keyframe "uốn lượn" trước gắp
rho_post      = 1e7   # shaping sau gắp
rho_hinge     = 1e8   # phạt vi phạm β >= beta_min

# Physical constants for theta mapping (for plot reference only)
g  = 9.81
mq = 0.500
mg = 0.158
ms = mq + mg
Lg = 0.105

# ==============================
# Core utilities
# ==============================

def poly_deriv_coeff(k, r):
    if k < r: return 0.0
    c=1.0
    for i in range(r): c *= (k-i)
    return c

def basis_row(t, degree, r):
    row = np.zeros(degree+1)
    for k in range(degree+1):
        coeff = poly_deriv_coeff(k, r)
        row[k] = 0.0 if coeff == 0.0 else coeff * (t**(k-r))
    return row

def segment_cost_Q(degree, T, snap_order=4):
    n=degree; Q=np.zeros((n+1,n+1))
    for i in range(snap_order, n+1):
        for j in range(snap_order, n+1):
            ci=cj=1.0
            for a in range(snap_order): ci *= (i-a)
            for b in range(snap_order): cj *= (j-b)
            power = i + j - 2*snap_order + 1
            Q[i, j] = ci * cj * (T**power) / power
    return Q

@dataclass
class Segment:
    T: float
    degree: int = 7

@dataclass
class TrajectorySpec1D:
    segments: list
    start_pos: float
    end_pos: float
    start_derivs: dict
    end_derivs: dict
    interior_positions: list

# ==============================
# QP assembly helpers
# ==============================

def build_eq_qp_1d(spec: TrajectorySpec1D, snap_order=4, continuity_order=3):
    segs  = spec.segments
    n     = segs[0].degree
    assert len(segs) == 2, "This helper expects exactly 2 segments"
    N     = (n+1)*2
    H = np.zeros((N, N))
    H[0:n+1, 0:n+1] = segment_cost_Q(n, segs[0].T, snap_order=snap_order)
    H[n+1:, n+1:]   = segment_cost_Q(n, segs[1].T, snap_order=snap_order)
    f = np.zeros(N)

    Aeq_rows = []; beq = []
    def add(si, tau, r, val):
        row = np.zeros(N); i0 = si*(n+1)
        row[i0:i0+n+1] = basis_row(tau, n, r)
        Aeq_rows.append(row); beq.append(val)

    # start
    add(0, 0.0, 0, spec.start_pos)
    for r, v in spec.start_derivs.items(): add(0, 0.0, r, v)
    # interior (pin position at pickup) + C^3 continuity
    Tk = segs[0].T
    add(0, Tk, 0, spec.interior_positions[0])
    for r in range(continuity_order+1):
        row = np.zeros(N)
        row[0:n+1]          = basis_row(Tk, n, r)
        row[(n+1):2*(n+1)] -= basis_row(0.0, n, r)
        Aeq_rows.append(row); beq.append(0.0)
    # end
    add(1, segs[1].T, 0, spec.end_pos)
    for r, v in spec.end_derivs.items(): add(1, segs[1].T, r, v)

    Aeq = np.vstack(Aeq_rows); beq = np.array(beq)
    return H, f, Aeq, beq

def solve_qp_equality(H, f, Aeq, beq, reg=1e-10):
    N = H.shape[0]
    Hreg = H + reg*np.eye(N)
    KKT  = np.block([[Hreg, Aeq.T], [Aeq, np.zeros((Aeq.shape[0], Aeq.shape[0]))]])
    rhs  = np.concatenate([-f, beq])
    sol  = np.linalg.solve(KKT, rhs)
    c    = sol[:N]
    return c

def add_soft_samples_to_objective(H, f, rows, values, weights):
    if len(rows) == 0:
        return H, f
    R = np.vstack(rows)
    y = np.array(values)
    W = np.diag(np.sqrt(np.asarray(weights)))
    RtWR = (W @ R).T @ (W @ R)
    RtWy = (W @ R).T @ (W @ y)
    H2 = H + 2.0 * RtWR
    f2 = f - 2.0 * RtWy
    return H2, f2

# ==============================
# Evaluate utilities
# ==============================

def evaluate_piecewise(coeffs, segs, t_query, degree=7):
    n=degree; cpseg=n+1
    starts=[0.0, segs[0].T]
    y=np.zeros_like(t_query); yd=np.zeros_like(t_query); ydd=np.zeros_like(t_query); yddd=np.zeros_like(t_query)
    for i,t in enumerate(t_query):
        sidx=0 if t<=starts[1] else 1; tau=t-starts[sidx]
        a=coeffs[sidx*cpseg:(sidx+1)*cpseg]
        y[i]=basis_row(tau,n,0).dot(a)
        yd[i]=basis_row(tau,n,1).dot(a)
        ydd[i]=basis_row(tau,n,2).dot(a)
        yddd[i]=basis_row(tau,n,3).dot(a)
    return y, yd, ydd, yddd

# ==============================
# Build and solve QPs for x and z
# ==============================

segs = [Segment(T=t_pick, degree=7), Segment(T=t_final - t_pick, degree=7)]
x_spec = TrajectorySpec1D(segs, start_pos=x0, end_pos=xF, start_derivs={1:0.0,2:0.0,3:0.0},
                          end_derivs={1:0.0,2:0.0,3:0.0}, interior_positions=[x_pick])
z_spec = TrajectorySpec1D(segs, start_pos=z0, end_pos=zF, start_derivs={1:0.0,2:0.0,3:0.0},
                          end_derivs={1:0.0,2:0.0,3:0.0}, interior_positions=[z_pick])

Hx, fx, Ax, bx = build_eq_qp_1d(x_spec)
Hz, fz, Az, bz = build_eq_qp_1d(z_spec)
cx = solve_qp_equality(Hx, fx, Ax, bx)
cz = solve_qp_equality(Hz, fz, Az, bz)

# ==============================
# Build QP for beta
# ==============================

beta0 = np.deg2rad(beta0_deg); beta_pick = np.deg2rad(beta_pick_deg); betaF = np.deg2rad(betaF_deg)
beta_spec = TrajectorySpec1D(segs, start_pos=beta0, end_pos=betaF,
                             start_derivs={1:0.0,2:0.0,3:0.0},
                             end_derivs={1:0.0,2:0.0,3:0.0},
                             interior_positions=[beta_pick])

Hb, fb, Ab, bb = build_eq_qp_1d(beta_spec)

beta_min = np.deg2rad(beta_min_deg)
rows = []; vals = []; wts = []

n = 7; cpseg = n+1

# ---- (1) Pre-pick LOS soft constraints (ĐÃ SỬA DẤU) ----
#   β_LOS = atan2( z_q(t) - z_pick , x_pick - x_q(t) )
#   => dương khi mục tiêu nằm BÊN DƯỚI quad (đúng quy ước β của bài báo).
t_pre = np.linspace(0.20, t_pick-0.004, 60)
for tau in t_pre:
    x_tau = basis_row(tau, n, 0).dot(cx[0:cpseg])
    z_tau = basis_row(tau, n, 0).dot(cz[0:cpseg])
    los = np.arctan2(z_tau - z_pick, x_pick - x_tau)  # <<< SỬA Ở ĐÂY
    row = np.zeros(2*(n+1)); row[0:cpseg] = basis_row(tau, n, 0)
    rows.append(row); vals.append(los); wts.append(rho_pre_los)

# ---- (2) Pre-pick SHAPING keyframes (để "uốn lượn" như Fig.6) ----
# thời điểm tương đối so với t_pick và giá trị β mục tiêu (độ)
pre_abs = np.array([t_pick-1.20, t_pick-0.90, t_pick-0.60, t_pick-0.30, t_pick-0.08])
pre_deg = np.array([        28,          40,          32,          18,          12])
for t_abs, val_deg in zip(pre_abs, pre_deg):
    tau = t_abs
    if tau <= 0.0: 
        continue
    row = np.zeros(2*(n+1)); row[0:cpseg] = basis_row(tau, n, 0)
    rows.append(row); vals.append(np.deg2rad(val_deg)); wts.append(rho_pre_shape)

# ---- (3) Post-pick shaping (giữ như trước) ----
post_abs = np.array([t_pick+0.02, t_pick+0.08, t_pick+0.14, t_pick+0.20, t_pick+0.26, t_pick+0.34, t_pick+0.50, t_pick+0.80, t_pick+1.20, t_pick+1.50])
post_deg = np.array([ 95, 105, 112, 120, 124, 125, 120, 105,  92,  90])
for t_abs, val_deg in zip(post_abs, post_deg):
    tau = t_abs - t_pick
    row = np.zeros(2*(n+1)); row[cpseg:2*cpseg] = basis_row(tau, n, 0)
    rows.append(row); vals.append(np.deg2rad(val_deg)); wts.append(rho_post)

# Gộp các term mềm vào (H, f) và giải QP cho β
Hb_soft, fb_soft = add_soft_samples_to_objective(Hb, fb, rows, vals, wts)
cb = solve_qp_equality(Hb_soft, fb_soft, Ab, bb)

# ---- (4) Hinge β >= beta_min trên lưới dày (giữ như trước) ----
def eval_beta(coeffs, t):
    out = np.zeros_like(t)
    for i, tt in enumerate(t):
        if tt <= t_pick:
            tau = tt
            out[i] = basis_row(tau, n, 0).dot(coeffs[0:cpseg])
        else:
            tau = tt - t_pick
            out[i] = basis_row(tau, n, 0).dot(coeffs[cpseg:2*cpseg])
    return out

grid = np.linspace(0.0, t_final, 1600)
for _ in range(8):
    b_val = eval_beta(cb, grid)
    viol = np.where(b_val < beta_min - 1e-9)[0]
    if len(viol) == 0:
        break
    rows_h = []; vals_h = []; wts_h = []
    for idx in viol[::2]:
        tt = grid[idx]
        row = np.zeros(2*(n+1))
        if tt <= t_pick:
            row[0:cpseg] = basis_row(tt, n, 0)
        else:
            row[cpseg:2*cpseg] = basis_row(tt - t_pick, n, 0)
        rows_h.append(row); vals_h.append(beta_min); wts_h.append(rho_hinge)
    Hb_soft, fb_soft = add_soft_samples_to_objective(Hb_soft, fb_soft, rows_h, vals_h, wts_h)
    cb = solve_qp_equality(Hb_soft, fb_soft, Ab, bb)

# ==============================
# Evaluate trajectories and plot
# ==============================

t = np.linspace(0.0, t_final, 800)
x, xd, xdd, _ = evaluate_piecewise(cx, segs, t)
z, zd, zdd, _ = evaluate_piecewise(cz, segs, t)
beta, betad, betadd, _ = evaluate_piecewise(cb, segs, t)

# theta (tham khảo) theo ánh xạ phẳng (Eq. (18))
xdd_g = xdd - Lg*(betadd*np.sin(beta) + (betad**2)*np.cos(beta))
zdd_g = zdd - Lg*(betadd*np.cos(beta) - (betad**2)*np.sin(beta))
xdd_s = (mq*xdd + mg*xdd_g)/ms
zdd_s = (mq*zdd + mg*zdd_g)/ms
b3x, b3z = xdd_s, zdd_s + g
norm = np.sqrt(b3x**2 + b3z**2); b3x/=norm; b3z/=norm
theta = np.arctan2(b3x, b3z)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.plot(t, np.rad2deg(beta), label=r'$\beta^d$')
ax1.plot(t, np.rad2deg(theta), label=r'$\theta^d$')
ax1.axvline(x=t_pick, linestyle='--')
ax1.set_xlabel('t (s)'); ax1.set_ylabel('angle (deg)')
ax1.set_title('β (pre-pick bump) and θ'); ax1.legend(); ax1.grid(True, alpha=0.5)
ax2.plot(t, x, label=r'$x_q^d$'); ax2.plot(t, z, label=r'$z_q^d$')
ax2.axvline(x=t_pick, linestyle='--')
ax2.set_xlabel('t (s)'); ax2.set_ylabel('position (m)')
ax2.set_title('Desired quadrotor position trajectories'); ax2.legend(); ax2.grid(True, alpha=0.5)
fig.tight_layout(); plt.show()

# ==============================
# Lưu file cho tranfer.py
# ==============================
import pandas as pd
flat_out_path = "C:/Users/2003h/OneDrive/Máy tính/quad2d_gripper/Quadrotor-Control-System/src/minsnap_results/flat_outputs1.csv"
df_flat = pd.DataFrame({
    "t": t,
    "x_q": x,
    "z_q": z,
    "beta": beta,     # rad
    "xd_q": xd,
    "zd_q": zd,
    "betad": betad    # rad/s
})
df_flat.to_csv(flat_out_path, index=False)
print(f"Saved flat outputs -> {flat_out_path}")
