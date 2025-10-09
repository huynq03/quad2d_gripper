
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
beta0_deg      = 25.0           # beta at t=0
beta_pick_deg  = 90.0           # vertical at pickup
betaF_deg      = 90.0           # settle near 90 at the end
beta_min_deg   = 5.0            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  non-negativity lower bound in degrees

# "Point-at-target" (pre-pick) and post-pick shaping weights
rho_pre   = 2e6  # strength for LOS before pickup
rho_post  = 1e7   # strength for shaping after pickup
rho_hinge = 1e8   # strength for hinge (β >= beta_min)

# Physical constants for theta mapping
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
    for i in range(r): c *= (k-i) # k(k-1)(k-2)...(k-r+1) = k!/(k-r)!
    return c

def basis_row(t, degree, r):
    row = np.zeros(degree+1)
    for k in range(degree+1):
        coeff = poly_deriv_coeff(k, r)
        row[k] = 0.0 if coeff == 0.0 else coeff * (t**(k-r)) # k!/(k-r)! * t^(k-r)
    return row

def segment_cost_Q(degree, T, snap_order=4):
    n=degree; Q=np.zeros((n+1,n+1)) # n = 7, snap_order=4, T la thoi gian cua doan
    for i in range(snap_order, n+1):
        for j in range(snap_order, n+1):
            ci=cj=1.0
            for a in range(snap_order): ci *= (i-a)
            for b in range(snap_order): cj *= (j-b)
            power = i + j - 2*snap_order + 1
            Q[i, j] = ci * cj * (T**power) / power
    return Q # tra ve ma tran chi phi Q dang cj*cj * int_0^T (d^snap_order p(t)/dt^snap_order)^2 dt

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
    """
    Return (H, f, Aeq, beq) for equality-only min-snap QP for a 2-segment, degree-7 piecewise polynomial.
    Cost: 0.5 c^T H c  (f=0 for pure min-snap), s.t. Aeq c = beq
    """
    segs  = spec.segments
    n     = segs[0].degree
    assert len(segs) == 2, "This helper expects exactly 2 segments" # mo ta 2 doan
    N     = (n+1)*2
    # Hessian for snap
    H = np.zeros((N, N))
    H[0:n+1, 0:n+1] = segment_cost_Q(n, segs[0].T, snap_order=snap_order)
    H[n+1:, n+1:]   = segment_cost_Q(n, segs[1].T, snap_order=snap_order)
    f = np.zeros(N)

    # Equality constraints
    Aeq_rows = []; beq = []
    def add(si, tau, r, val):
        row = np.zeros(N); i0 = si*(n+1)
        row[i0:i0+n+1] = basis_row(tau, n, r)
        Aeq_rows.append(row); beq.append(val)

    # start
    add(0, 0.0, 0, spec.start_pos)
    for r, v in spec.start_derivs.items(): add(0, 0.0, r, v)
    # interior boundary: pin position at end of seg 0 and enforce C^3 continuity
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
    """
    === SOLVE QP ===
    Minimize 0.5 c^T H c + f^T c   subject to  Aeq c = beq
    Solve KKT system:
      [H  Aeq^T] [c     ] = [ -f ]
      [Aeq  0  ] [lambda]   [ beq]
    """
    N = H.shape[0]
    Hreg = H + reg*np.eye(N)
    KKT  = np.block([[Hreg, Aeq.T], [Aeq, np.zeros((Aeq.shape[0], Aeq.shape[0]))]])
    rhs  = np.concatenate([-f, beq])
    sol  = np.linalg.solve(KKT, rhs)
    c    = sol[:N]
    return c

def add_soft_samples_to_objective(H, f, rows, values, weights):
    """
    Fold a weighted least-squares term  sum_i w_i (row_i c - value_i)^2  into (H,f).
    H <- H + 2 * R^T W R,   f <- f - 2 * R^T W y
    """
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

# Solve QP (equality) for x and z
cx = solve_qp_equality(Hx, fx, Ax, bx)
cz = solve_qp_equality(Hz, fz, Az, bz)

# ==============================
# Build QP for beta
# ==============================

# Equality structure (start/pickup/end + C^3 continuity)
beta0 = np.deg2rad(beta0_deg); beta_pick = np.deg2rad(beta_pick_deg); betaF = np.deg2rad(betaF_deg)
beta_spec = TrajectorySpec1D(segs, start_pos=beta0, end_pos=betaF,
                             start_derivs={1:0.0,2:0.0,3:0.0},
                             end_derivs={1:0.0,2:0.0,3:0.0},
                             interior_positions=[beta_pick])

Hb, fb, Ab, bb = build_eq_qp_1d(beta_spec)
# Pre-pick LOS soft constraints (clipped at beta_min)
beta_min = np.deg2rad(beta_min_deg)
rows = []; vals = []; wts = []

n = 7; cpseg = n+1
t_pre = np.linspace(0.20, t_pick-0.004, 60)
for tau in t_pre:
    # x,z at tau from first segment coeffs
    x_tau = basis_row(tau, n, 0).dot(cx[0:cpseg])
    z_tau = basis_row(tau, n, 0).dot(cz[0:cpseg])
    los = np.arctan2(z_pick - z_tau, x_pick - x_tau)
    los = max(los, beta_min)  # clip to enforce β target non-negative
    row = np.zeros(2*(n+1)); row[0:cpseg] = basis_row(tau, n, 0)
    rows.append(row); vals.append(los); wts.append(rho_pre)

# Post-pick shaping soft constraints
post_abs = np.array([t_pick+0.02, t_pick+0.08, t_pick+0.14, t_pick+0.20, t_pick+0.26, t_pick+0.34, t_pick+0.50, t_pick+0.80, t_pick+1.20, t_pick+1.50])
post_deg = np.array([ 95, 105, 112, 120, 124, 125, 120, 105,  92,  90])
for t_abs, val_deg in zip(post_abs, post_deg):
    tau = t_abs - t_pick
    row = np.zeros(2*(n+1)); row[cpseg:2*cpseg] = basis_row(tau, n, 0)
    rows.append(row); vals.append(np.deg2rad(val_deg)); wts.append(rho_post)

# Fold soft terms into (Hb, fb)
Hb_soft, fb_soft = add_soft_samples_to_objective(Hb, fb, rows, vals, wts)

# Solve QP (equality) for beta (1st pass)
cb = solve_qp_equality(Hb_soft, fb_soft, Ab, bb)

# Active-set soft-hinge for β >= beta_min on a dense grid
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
    for idx in viol[::2]:   # subsample to keep system well-conditioned
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

# theta from flatness mapping (Appendix Eq. (18))
xdd_g = xdd - Lg*(betadd*np.sin(beta) + (betad**2)*np.cos(beta))
zdd_g = zdd - Lg*(betadd*np.cos(beta) - (betad**2)*np.sin(beta))
xdd_s = (mq*xdd + mg*xdd_g)/ms
zdd_s = (mq*zdd + mg*zdd_g)/ms
b3x, b3z = xdd_s, zdd_s + g
norm = np.sqrt(b3x**2 + b3z**2); b3x/=norm; b3z/=norm
theta = np.arctan2(b3x, b3z)

# Plot β and θ
plt.figure()
plt.plot(t, np.rad2deg(beta), label=r'$\beta^d$')
plt.plot(t, np.rad2deg(theta), label=r'$\theta^d$')
# plt.axhline(y=beta_min_deg, linestyle='--')
plt.axvline(x=t_pick, linestyle='--')
plt.xlabel('t (s)'); plt.ylabel('angle (deg)')
plt.title('β and θ with β(t) ≥ {:.1f}° and "point-at-target" before pickup'.format(beta_min_deg))
plt.legend(); plt.tight_layout(); plt.show()

# Plot x and z (for reference)
plt.figure()
plt.plot(t, x, label=r'$x_q^d$')
plt.plot(t, z, label=r'$z_q^d$')
plt.axvline(x=t_pick, linestyle='--')
plt.xlabel('t (s)'); plt.ylabel('position (m)')
plt.title('Desired quadrotor position trajectories')
plt.legend(); plt.tight_layout(); plt.show()