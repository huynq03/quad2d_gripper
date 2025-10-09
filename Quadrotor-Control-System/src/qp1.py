
"""
Minimum-snap QP for flat outputs (x_q, z_q, beta) with the "point-at-target" constraint for beta,
following Thomas et al., "Avian-Inspired Grasping for Quadrotor Micro UAVs" (IDETC 2013).

Key paper facts used here
-------------------------
- Flat outputs: y = [x_q, z_q, beta]^T (Eq. (9)).
- Cost: minimize integral of squared snap for each output (Eq. (10)).
- Trajectory must be C^3 (continuity up to jerk) because inputs depend on snap.
- Constraints used in the paper (text under Fig. 5 & Fig. 6): start/finish positions with zero vel/acc/jerk;
  at pickup, the gripper is vertical; **before pickup, beta is constrained so that the gripper points
  directly at the target**. This is the missing ingredient that makes beta rise very steeply near t_pick.

This script implements the above. For beta we add **soft constraints** at many pre-pick sample times to make
beta(t) follow the line-of-sight to the target: beta_LOS(t) = atan2(z_T - z_q(t), x_T - x_q(t)).
Optionally, we add a few post-pickup "shape" samples to cap the peak (~125 deg) and settle to ~90 deg.

How to tune
-----------
- Change waypoints x_pick, z_pick, xF, zF and the time t_pick, tF.
- Change the target (x_T, z_T). By default x_T=x_pick, z_T=z_pick.
- Adjust rho_pre (strength of "point-at-target" before pickup). Larger -> closer tracking -> steeper swing at t_pick.
- Adjust rho_post and post_targets to cap the peak near 120–130 deg and settle to 90 deg.

This file is self-contained (numpy + matplotlib only).
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ---------------- Utilities ----------------

def poly_deriv_coeff(k, r):
    if k < r: return 0.0
    c = 1.0
    for i in range(r): c *= (k - i)
    return c

def basis_row(t, degree, r):
    row = np.zeros(degree + 1)
    for k in range(degree + 1):
        coeff = poly_deriv_coeff(k, r)
        row[k] = 0.0 if coeff == 0.0 else coeff * (t ** (k - r))
    return row

def segment_cost_Q(degree, T, snap_order=4):
    n = degree
    Q = np.zeros((n+1, n+1))
    for i in range(snap_order, n+1):
        for j in range(snap_order, n+1):
            ci = 1.0
            for a in range(snap_order): ci *= (i - a)
            cj = 1.0
            for b in range(snap_order): cj *= (j - b)
            power = i + j - 2*snap_order + 1
            Q[i, j] = ci * cj * (T ** power) / power
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
    interior_positions: list  # one per interior boundary

def solve_min_snap_eq_only(spec: TrajectorySpec1D, continuity_order=3):
    """Equality-only minimum-snap for a 2-segment, degree-7 piecewise polynomial."""
    segs = spec.segments
    n = segs[0].degree
    S = len(segs)
    assert S == 2, "This helper expects 2 segments"
    N = (n+1)*S

    # Hessian (block-diag)
    H = np.zeros((N, N))
    H[0:n+1, 0:n+1] = segment_cost_Q(n, segs[0].T)
    H[n+1:, n+1:]   = segment_cost_Q(n, segs[1].T)

    # Build Aeq, beq
    A, b = [], []
    def add(si, tau, r, val):
        row = np.zeros(N); i0 = si*(n+1)
        row[i0:i0+n+1] = basis_row(tau, n, r)
        A.append(row); b.append(val)

    # start
    add(0, 0.0, 0, spec.start_pos)
    for r, v in spec.start_derivs.items(): add(0, 0.0, r, v)

    # interior boundary (position + C^3 continuity)
    Tk = segs[0].T
    add(0, Tk, 0, spec.interior_positions[0])
    for r in range(continuity_order+1):
        row = np.zeros(N)
        row[0:n+1]          = basis_row(Tk, n, r)
        row[(n+1):2*(n+1)] -= basis_row(0.0, n, r)
        A.append(row); b.append(0.0)

    # end
    add(1, segs[1].T, 0, spec.end_pos)
    for r, v in spec.end_derivs.items(): add(1, segs[1].T, r, v)

    A = np.vstack(A); b = np.array(b)
    # Solve KKT
    reg = 1e-8
    Hreg = H + reg*np.eye(N)
    KKT = np.block([[Hreg, A.T],[A, np.zeros((A.shape[0], A.shape[0]))]])
    rhs = np.concatenate([np.zeros(N), b])
    sol = np.linalg.solve(KKT, rhs)
    return sol[:N], H

def solve_beta_with_soft_constraints(segs, H_snap, Aeq, beq, x_coeffs, z_coeffs,
                                     xT, zT, t_pick, rho_pre=2e6,
                                     post_targets=None, rho_post=1e6):
    """
    Minimize 0.5 c^T (H_snap + 2*Phi^T W Phi) c - (2 Phi^T W y) c
    subject to Aeq c = beq.
    - Pre-pick Phi/y from LOS samples to (xT,zT).
    - Post-pick Phi/y from shaping samples (time,value) in 'post_targets' (absolute times).
    """
    n = segs[0].degree; N = (n+1)*2
    # Collect equality rows from Aeq, beq (already built for "structure": start, interior, end)
    # Here we rebuild Aeq, beq to match: start at 22 deg; interior pos at 90 deg; end at 90 deg; and C^3 continuity.
    A, b = [], []
    def add(si, tau, r, val):
        row = np.zeros(N); i0 = si*(n+1); row[i0:i0+n+1] = basis_row(tau, n, r)
        A.append(row); b.append(val)
    # start & end derivatives zero
    add(0, 0.0, 0, np.deg2rad(22.0)); [add(0,0.0,r,0.0) for r in [1,2,3]]
    add(0, t_pick, 0, np.deg2rad(90.0))
    for r in [0,1,2,3]:
        row = np.zeros(N); row[0:n+1] = basis_row(t_pick, n, r); row[n+1:2*(n+1)] -= basis_row(0.0, n, r)
        A.append(row); b.append(0.0)
    add(1, segs[1].T, 0, np.deg2rad(90.0)); [add(1, segs[1].T, r, 0.0) for r in [1,2,3]]
    Aeq = np.vstack(A); beq = np.array(b)

    # Build Phi, y, weights
    Phi_rows, y_vals, w = [], [], []

    # Pre-pick dense LOS samples
    t_samples = np.linspace(0.2, t_pick - 0.003, 60)
    cpseg = n+1
    for tau in t_samples:
        # evaluate x_q, z_q on segment 0
        x_tau = basis_row(tau, n, 0).dot(x_coeffs[0:cpseg])
        z_tau = basis_row(tau, n, 0).dot(z_coeffs[0:cpseg])
        los   = np.arctan2(zT - z_tau, xT - x_tau)
        row = np.zeros(N); row[0:cpseg] = basis_row(tau, n, 0)
        Phi_rows.append(row); y_vals.append(los); w.append(rho_pre)

    # Post-pick shaping samples
    if post_targets is None:
        post_targets = [(t_pick+0.1, np.deg2rad(110.0)),
                        (t_pick+0.3, np.deg2rad(125.0)),
                        (t_pick+0.5, np.deg2rad(120.0)),
                        (t_pick+0.8, np.deg2rad(105.0)),
                        (t_pick+1.2, np.deg2rad(92.0)),
                        (t_pick+1.5, np.deg2rad(90.0))]
    for t_abs, val in post_targets:
        tau = t_abs - t_pick  # local time within segment 1
        row = np.zeros(N); row[cpseg:2*cpseg] = basis_row(tau, n, 0)
        Phi_rows.append(row); y_vals.append(val); w.append(rho_post)

    Phi = np.stack(Phi_rows); y = np.array(y_vals); w = np.array(w)
    W  = np.diag(np.sqrt(w))
    Phi_w = W @ Phi; y_w = W @ y

    H_aug = H_snap + 2*(Phi_w.T @ Phi_w)
    f     = -2*(Phi_w.T @ y_w)

    # Solve KKT for general quadratic with linear term
    reg = 1e-8
    Hreg = H_aug + reg*np.eye(N)
    KKT  = np.block([[Hreg, Aeq.T],[Aeq, np.zeros((Aeq.shape[0], Aeq.shape[0]))]])
    rhs  = np.concatenate([-f, beq])
    sol  = np.linalg.solve(KKT, rhs)
    return sol[:N], t_samples, y[:len(t_samples)], post_targets

def evaluate_pw(coeffs, segs, t_query):
    n = segs[0].degree; Nseg = len(segs); cpseg = n+1
    starts = [0.0, segs[0].T]
    y = np.zeros_like(t_query); yd = np.zeros_like(t_query)
    ydd = np.zeros_like(t_query); yddd = np.zeros_like(t_query)
    for i, t in enumerate(t_query):
        sidx = 0 if t <= starts[1] else 1
        tau  = t - starts[sidx]
        a    = coeffs[sidx*cpseg:(sidx+1)*cpseg]
        y[i]    = basis_row(tau, n, 0).dot(a)
        yd[i]   = basis_row(tau, n, 1).dot(a)
        ydd[i]  = basis_row(tau, n, 2).dot(a)
        yddd[i] = basis_row(tau, n, 3).dot(a)
    return y, yd, ydd, yddd

# --------------- Parameters ---------------

t_pick = 2.0
tF     = 3.5
segs   = [Segment(T=t_pick, degree=7), Segment(T=tF - t_pick, degree=7)]

# Quadrotor waypoints (example)
x0, z0        = -2.0,  2.0        # start ~ (-2, 2)
x_pick, z_pick =  0.50, 1.35      # position at pickup (min z ≈ 1.35 m)
xF, zF        =  1.55, 1.55       # finish ~ (1.5, 1.5)

# Target location (assume fixed at pickup point)
xT, zT = x_pick, z_pick

# --------------- Solve x and z first ---------------

x_spec = TrajectorySpec1D(segs, x0, xF, {1:0.0,2:0.0,3:0.0}, {1:0.0,2:0.0,3:0.0}, [x_pick])
z_spec = TrajectorySpec1D(segs, z0, zF, {1:0.0,2:0.0,3:0.0}, {1:0.0,2:0.0,3:0.0}, [z_pick])
cx, Hx = solve_min_snap_eq_only(x_spec)
cz, Hz = solve_min_snap_eq_only(z_spec)

# --------------- Solve β with soft constraints ---------------

# Start with the same H structure as x/z (snap on each segment)
H_beta = np.zeros_like(Hx); H_beta[:] = 0.0
H_beta[0:8, 0:8] = segment_cost_Q(7, segs[0].T)
H_beta[8:,  8:]  = segment_cost_Q(7, segs[1].T)

cb, t_los, beta_los, post_targets = solve_beta_with_soft_constraints(
    segs, H_beta, None, None, cx, cz, xT, zT, t_pick,
    rho_pre=2e6,  # strong "point-at-target" before pickup
    post_targets=[(t_pick+0.10, np.deg2rad(110.0)),
                  (t_pick+0.30, np.deg2rad(125.0)),
                  (t_pick+0.50, np.deg2rad(120.0)),
                  (t_pick+0.80, np.deg2rad(105.0)),
                  (t_pick+1.20, np.deg2rad(92.0)),
                  (t_pick+1.50, np.deg2rad(90.0))],
    rho_post=1e6
)

# --------------- Evaluate and plot ---------------

t = np.linspace(0.0, tF, 800)
x, xd, xdd, xddd      = evaluate_pw(cx, segs, t)
z, zd, zdd, zddd      = evaluate_pw(cz, segs, t)
beta, bd, bdd, bddd   = evaluate_pw(cb, segs, t)

# Recover theta (Appendix Eq. (18))
g  = 9.81; mq = 0.500; mg = 0.158; ms = mq + mg; Lg = 0.105
xdd_g = xdd - Lg*(bdd*np.sin(beta) + (bd**2)*np.cos(beta))
zdd_g = zdd - Lg*(bdd*np.cos(beta) - (bd**2)*np.sin(beta))
xdd_s = (mq*xdd + mg*xdd_g)/ms
zdd_s = (mq*zdd + mg*zdd_g)/ms
b3x, b3z = xdd_s, zdd_s + g
norm = np.sqrt(b3x**2 + b3z**2); b3x /= norm; b3z /= norm
theta = np.arctan2(b3x, b3z)

# Plots
plt.figure()
plt.plot(t, np.rad2deg(beta), label=r'$\beta^d$')
plt.plot(t, np.rad2deg(theta), label=r'$\theta^d$')
plt.scatter(t_los, np.rad2deg(beta_los), s=8, label='LOS samples (pre-pick)')
tp = np.array([pt[0] for pt in post_targets]); bv = np.array([pt[1] for pt in post_targets])
plt.scatter(tp, np.rad2deg(bv), s=20, label='Post-pick targets')
plt.axvline(x=t_pick, linestyle='--')
plt.xlabel('t (s)'); plt.ylabel('angle (deg)')
plt.title('β and θ with “point-at-target” before pickup (steep rise at t_pick)')
plt.legend(); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(t, x, label=r'$x_q^d$')
plt.plot(t, z, label=r'$z_q^d$')
plt.axvline(x=t_pick, linestyle='--')
plt.xlabel('t (s)'); plt.ylabel('position (m)')
plt.title('Desired quadrotor position (for reference)')
plt.legend(); plt.tight_layout(); plt.show()