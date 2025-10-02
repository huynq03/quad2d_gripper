import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================
# Parameters
# =============================
dt   = 0.02
T    = 10.0
m_q  = 0.6
m_p  = 0.15
g    = 9.81
I_xx = 0.15
l_p  = 0.35
l_q  = 0.2   # for visualization

# =============================
# Dynamics (JAX)
# state = [y, y_dot, z, z_dot, phi, phi_dot, theta, theta_dot]
# control = [u1 (thrust), u2 (torque)]
# =============================
@jax.jit
def jax_dynamics(state, control, dt=dt):
    y, y_dot, z, z_dot, phi, phi_dot, theta, theta_dot = state
    u1, u2 = control

    M = m_q + m_p

    y_ddot = (m_q + m_p*jnp.cos(theta)**2) / (m_q*M) * u1 * jnp.sin(phi) \
             + (m_p*jnp.cos(theta)*jnp.sin(theta)) / (m_q*M) * u1 * jnp.cos(phi) \
             + (m_p*l_p*(theta_dot**2)*jnp.sin(theta)) / M

    z_ddot = -g + (m_q + m_p*jnp.sin(theta)**2) / (m_q*M) * u1 * jnp.cos(phi) \
             + (m_p*jnp.cos(theta)*jnp.sin(theta)) / (m_q*M) * u1 * jnp.sin(phi) \
             + (m_p*l_p*(theta_dot**2)*jnp.cos(theta)) / M

    phi_ddot   = u2 / I_xx
    theta_ddot = -jnp.cos(theta) / (m_q*l_p) * u1 * jnp.sin(phi) \
                 - jnp.sin(theta) / (m_q*l_p) * u1 * jnp.cos(phi)

    next_state = state + jnp.array([
        y_dot, y_ddot, z_dot, z_ddot, phi_dot, phi_ddot, theta_dot, theta_ddot
    ]) * dt
    return next_state

# =============================
# Simple PID controller (cascade)
# =============================
class PID:
    def __init__(self, Kp, Ki, Kd, out_min=None, out_max=None):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.i_term = 0.0
        self.prev_e = 0.0
        self.out_min, self.out_max = out_min, out_max

    def __call__(self, ref, meas, dt):
        e = ref - meas
        self.i_term += e * dt
        d = (e - self.prev_e) / dt
        self.prev_e = e
        u = self.Kp*e + self.Ki*self.i_term + self.Kd*d
        if self.out_min is not None and self.out_max is not None:
            u = np.clip(u, self.out_min, self.out_max)
        return float(u)

class CascadePID:
    def __init__(self, target=(5.0, 5.0)):
        self.y_ref, self.z_ref = target
        self.pid_z = PID(15.0, 2.0, 10.0, out_min=-5, out_max=5)
        self.pid_y = PID(2.0, 0.0, 1.0, out_min=-0.3, out_max=0.3)
        self.pid_phi = PID(50.0, 1.0, 10.0, out_min=-10, out_max=10)
        self.pid_theta = PID(0.5, 0.0, 0.2, out_min=-0.2, out_max=0.2)

    def step(self, state):
        y, ydot, z, zdot, phi, phidot, theta, thetad = state
        u1 = (m_q+m_p) * (g + self.pid_z(self.z_ref, z, dt))
        phi_ref = self.pid_y(self.y_ref, y, dt)
        phi_ref -= self.pid_theta(0.0, theta, dt)
        u2 = self.pid_phi(phi_ref, phi, dt)
        return np.clip(u1, 0.0, 30.0), np.clip(u2, -15.0, 15.0)

# =============================
# Simulation
# =============================
def simulate(controller, T=T, dt=dt):
    N = int(T/dt)
    state = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, np.deg2rad(30), 0.0])
    states = [state]
    controls = []

    for _ in range(N):
        u1, u2 = controller.step(state)
        state = np.array(jax_dynamics(state, jnp.array([u1, u2])))
        states.append(state)
        controls.append([u1, u2])

    return np.array(states), np.array(controls)

# =============================
# Visualization
# =============================
def animate(states, controls, target=(5.0,5.0), dt=dt):
    y = states[:,0]; z = states[:,2]; phi = states[:,4]; theta = states[:,6]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(min(y.min(), target[0])-1, max(y.max(), target[0])+1)
    ax.set_ylim(min(z.min(), target[1])-1, max(z.max(), target[1])+1)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.add_patch(plt.Circle(target, 0.1, color="g", fill=False))

    frame_line, = ax.plot([], [], "k", lw=5)
    tether_line, = ax.plot([], [], "gray", lw=2)
    trail, = ax.plot([], [], "b-", lw=1, alpha=0.6)

    def update(i):
        yc, zc, phic, thetac = y[i], z[i], phi[i], theta[i]
        R = np.array([[np.cos(phic), -np.sin(phic)],
                      [np.sin(phic),  np.cos(phic)]])
        main = np.array([[-l_q, l_q],[0.0,0.0]])
        body = R @ main + np.array([[yc,yc],[zc,zc]])
        frame_line.set_data(body[0], body[1])

        pend = np.array([[0, l_p*np.sin(thetac)],
                         [0,-l_p*np.cos(thetac)]])
        pend_w = R @ pend + np.array([[yc,yc],[zc,zc]])
        tether_line.set_data(pend_w[0], pend_w[1])

        trail.set_data(y[:i+1], z[:i+1])
        return frame_line, tether_line, trail

    ani = FuncAnimation(fig, update, frames=len(states), interval=dt*1000, blit=True)
    plt.show()

# =============================
# Main
# =============================
def main():
    controller = CascadePID(target=(5.0, 5.0))
    states, controls = simulate(controller)
    animate(states, controls, target=(5.0,5.0))

if __name__ == "__main__":
    main()
