import numpy as np
from numpy import sin, cos, sqrt
# import cvxpy as cp
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from time import time


class ARROW:
    """ An Aerial Robot On Wheel
    simplified model, [x,theta] is the state
    """

    def __init__(self,
                 iniConfiguration=[0.0, np.pi / 4.0],    # [m, m, rad]
                 iniVelocity=[0.0, 0.0],
                 iniControl=[0.0, 0.0],
                 controlLaw='baseline',
                 k_d=3,
                 k_p=5,
                 l=2.0,     # length of bar
                 m_w=1.0,   # [kg]  - wheel
                 m_q=0.3,   # [kg]  - quadrotor
                 g=9.8,      # [m/s^2]
                 alpha=0.1,     # slop
                 c_x=0.0,
                 c_y=-1,
                 constraint='horizontal'
                 ):
        self.t = 0.0  # elapsed time
        self.q = np.matrix(iniConfiguration)
        self.dq = np.matrix(iniVelocity)
        self.u = np.matrix(iniControl)
        self.controlLaw = controlLaw

        self.params = (l, m_w, m_q, g, alpha, c_x, c_y)

        self.k_d = k_d
        self.k_p = k_p

        self.constraint = constraint

        self.data = {'t': [], 'q': [], 'dq': [], 'u': [],
                     'A': [], 'b': [], 'x_star': []
                     }
        self.data['t'].append(self.t)
        self.data['q'].append(self.q)
        self.data['dq'].append(self.dq)
        self.data['u'].append(self.u)
        self.data['A'].append(np.matrix(iniControl))
        self.data['b'].append(0.0)
        self.data['x_star'].append(np.array([0.0, 0.0]))

        print("ARROW initiated with (l, m_w, m_q, g) = ", self.params)
        print(controlLaw + " controller with (k_d, k_p)  = ", (k_d, k_p))

    def getConstraint(self):
        (l, m_w, m_q, g, alpha, c_x, c_y) = self.params

        # horizontal
        N = np.matrix([[0, 1, 0]])
        Delta_u = np.matrix([[1, 0], [0, 0], [0, 1]])
        dDelta_u = np.matrix([[0, 0], [0, 0], [0, 0]])

        if self.constraint == 'slop':
            # slop
            N = np.matrix([[alpha, 1, 0]])
            Delta_u = np.matrix([[-1, 0], [alpha, 0], [0, 1]])
            dDelta_u = np.matrix([[0, 0], [0, 0], [0, 0]])

        # elif constraint == 'circle':
        #     N = np.matrix([[-2 * c_x + 2 * x, -2 * c_y + 2 * y, 0]])
        #     Delta_u = np.matrix([[c_y - y, 0], [-c_x + x, 0], [0, 1]])
        #     dDelta_u = np.matrix([[-dy, 0], [dx, 0], [0, 0]])

        return Delta_u, dDelta_u

    def getLagrangian(self, q=None, dq=None):

        (l, m_w, m_q, g, alpha, c_x, c_y) = self.params
        x, theta = self.q[0].item(), self.q[1].item()
        dx, dtheta = self.dq[0].item(), self.dq[1].item()

        # M ddq + C dq + G = B u
        M = np.matrix([[m_q + m_w, 0, l * m_q * cos(theta)], [0, m_q + m_w, -l * m_q * sin(theta)], [l * m_q * cos(theta), -l * m_q * sin(theta), l**2 * m_q]])
        C = np.matrix([[0, 0, -dtheta * l * m_q * sin(theta)], [0, 0, -dtheta * l * m_q * cos(theta)], [0, 0, 0]])
        G = np.matrix([[0], [g * (m_q + m_w)], [-g * l * m_q * sin(theta)]])
        B = np.matrix([[1, 0], [0, 1], [l * cos(theta), -l * sin(theta)]])

        # no gravity
        # G = np.matrix([[0], [0], [0]])

        # horizontal
        # N = np.matrix([[0, 1, 0]])
        # Delta_u = np.matrix([[1, 0, 0], [0, 0, 1]])
        Delta_u, dDelta_u = self.getConstraint()
        # print(Delta_u)
        M_bar = Delta_u.transpose() * M * Delta_u
        C_bar = Delta_u.transpose() * (M * dDelta_u + C * Delta_u)
        G_bar = Delta_u.transpose() * G
        B_bar = Delta_u.transpose() * B

        # return M, C, G, B

        return M_bar, C_bar, G_bar, B_bar

    def runSimulation(self, dt):
        # choose the control law
        M, C, G, B = self.getLagrangian()

        u = self.LyapunovControl(self.k_d, self.k_p)

        # state update
        self.dq = self.dq + np.linalg.inv(M) * (-C * self.dq - G + B * u) * dt
        self.q = self.q + self.dq * dt

        # store data
        self.t += dt
        self.data['t'].append(self.t)
        self.data['q'].append(self.q)
        self.data['dq'].append(self.dq)
        self.data['u'].append(self.u)

        return self.q, self.dq

    def updateReference(self, qd, dqd, ddqd):
        self.qd, self.dqd, self.ddqd = qd, dqd, ddqd

    def getData(self):
        return self.data

    def LyapunovControl(self, k_d=3, k_p=5):
        (l, m_w, m_q, g, alpha, c_x, c_y) = self.params
        qd, dqd, ddqd = self.qd, self.dqd, self.ddqd
        q, dq = self.q, self.dq
        M, C, G, B = self.getLagrangian()

        vd = -k_p * (q - qd) + dqd
        v = - k_d * B.transpose() * (dq - vd)

        R = np.matrix([0.0])
        self.u = v + np.linalg.inv(B) * G
        #+ np.matrix([0, g * (m_q)]).transpose()
        return self.u

    def config(self):
        xt = self.q[0].item()
        theta = self.q[1].item()

        (l, m_w, m_q, g, alpha, c_x, c_y) = self.params

        u = self.u
        # + np.matrix([0, g * m_q]).transpose()

        # draw circle
        radius = 0.2
        angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
        ox = [radius * math.cos(a) for a in angles]
        oy = [radius * math.sin(a) for a in angles]

        # bar
        bx = np.matrix([xt, xt + l * math.sin(theta)])
        by = np.matrix([0.0, l * math.cos(theta)])

        # quadrotor
        # control
        phi = math.atan2(u[1].item(), u[0].item()) - math.pi / 2.0
        ux = np.matrix([0.0, u[0].item()]) * 0.2
        uy = np.matrix([0.0, u[1].item()]) * 0.2
        ux += float(bx[0, -1])
        uy += float(by[0, -1])

        qbar = 0.5
        qx = np.matrix([-qbar * math.cos(phi), qbar * math.cos(phi)])
        qy = np.matrix([-qbar * math.sin(phi), qbar * math.sin(phi)])
        qx = qx + float(bx[0, -1])
        qy = qy + float(by[0, -1])

        # wheeled sensor
        wx = np.copy(ox)
        wy = np.copy(oy)
        wx = wx + xt

        return wx, wy, bx, by, qx, qy, u


class RobotAnimation(animation.FuncAnimation):
    """docstring for RobotAnimation"""

    def __init__(self, robot, fig=None, ax=None, T=10, dt=1.0 / 30.0):
        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        self.fig = fig
        self.ax = ax

        self.robot = robot

        self.bar, = ax.plot([], [], '-', lw=1)
        self.quadrotor, = ax.plot([], [], '-r', lw=2)
        self.wheel, = ax.plot([], [], '-r', lw=1)
        self.floor, = ax.plot([-40, 40], [-0.20, -0.20], '-', lw=0.75, color='0.75')
        self.coord, = ax.plot([0, 0], [-0.20, 0.00], '-', lw=0.75, color='0.75')
        self.thrust = ax.quiver(0, 0, 0, 0, units='xy', scale=6, color='blue')
        self.time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        ts = time()
        self.animate(0)
        te = time()
        interval = 1000 * dt - (te - ts)

        super(RobotAnimation, self).__init__(fig, self.animate, frames=int(T / dt), interval=interval, blit=True, init_func=self.init, repeat=True)

    def init(self):
        self.bar.set_data([], [])
        self.quadrotor.set_data([], [])
        self.wheel.set_data([], [])
        # thrust.ax.set_UVC([], [])
        self.time_text.set_text('')
        return self.bar, self.quadrotor, self.time_text, self.wheel, self.thrust

    def animate(self, i):

        self.robot.runSimulation(dt)

        wx, wy, bx, by, qx, qy, u = self.robot.config()

        timet = i * dt
        self.bar.set_data(bx, by)
        self.quadrotor.set_data(qx, qy)
        self.wheel.set_data(wx, wy)
        self.thrust.set_UVC(u[0].item(), u[1].item())
        self.thrust.set_offsets((bx[0, 1], by[0, 1]))
        self.time_text.set_text('time = %.1f' % timet)
        return self.bar, self.quadrotor, self.time_text, self.wheel, self.thrust


if __name__ == '__main__':

    # initial pose
    q0 = np.matrix([0.0, np.pi / 4]).transpose()
    dq0 = np.matrix([0.0, 0.0]).transpose()
    u0 = np.matrix([0.0, 0.0]).transpose()
    # desired pose
    qd = np.matrix([1.0, -math.pi / 3]).transpose()
    dqd = np.matrix([0.0, 0.0]).transpose()
    ddqd = np.matrix([0.0, 0.0]).transpose()

    k_d = 3
    # k_p = 1
    k_p = 1  # violate the constraints

    robot = ARROW(q0, dq0, u0, 'Lyapunov', k_d=k_d, k_p=k_p, alpha=1)
    robot.updateReference(qd, dqd, ddqd)
    T = 10
    dt = 0.01  # 30 fps

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.xticks([]), plt.yticks([])
    ax.set_xlim((-4, 4)), ax.set_ylim((-1, 4))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.grid(color='0.75', linestyle='-', linewidth=0.75)

    # ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    # ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    # ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    # ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
    # ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    # ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')

    anim = RobotAnimation(robot, fig=fig, ax=ax, T=T, dt=dt)

    # anim.save('ARROW.mp4', fps=60, writer='ffmpeg')
    plt.show()