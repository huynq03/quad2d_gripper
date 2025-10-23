# hann@ieee.org

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


class Robot:
    def __init__(self, m, g):
        # only initiate essential params and variable
        self.m, self.g = m, g
        
    def dynamics(self):
        # M ddq + C dq + G = B u
        M = np.array([[self.m, 0], [0, self.m]])
        C = np.array([[0, 0], [0, 0]])
        B = np.eye(2)
        G = np.array([[0], [-self.m*self.g]])
        return M, C, B, G
    def step(self, q, dq, u, dt):
        M, C, B, G = self.dynamics()
        
        q_ddot = np.linalg.inv(M)@(B@u - G - C@(dq))
        dq += q_ddot * dt
        q += dq * dt
        
        return q, dq
    

class Planner: 
    def __init__(self, x_d=3., z_d=0.5):
        self.x_d, self.z_d = x_d, z_d
        
    def step(self, q, dq, dt):
        q_d = np.array([[self.x_d],[self.z_d]])
        dq_d = np.array([[0.],[0.]])
        ddq_d = np.array([[0.],[0.]])
        return q_d, dq_d, ddq_d


class Controller:
    def __init__(self, m, g, gain_b, gain_k):
        self.m, self.g = m, g
        self.b, self.k = gain_b, gain_k

    def dynamics(self):
        # M ddq + C dq + G = B u
        M = np.array([[self.m, 0], [0, self.m]])
        C = np.array([[0, 0], [0, 0]])
        B = np.eye(2)
        G = np.array([[0], [-self.m*self.g]])
        return M, C, B, G

    def step(self, q, dq, q_d, dq_d, ddq_d, dt):
        # u = G + M ddq_d - b(dq - dq_d) - k (q - q_d)
        M, C, B, G = self.dynamics()
        u = G + M@ddq_d - (dq - dq_d)*self.b - (q-q_d)*self.k
        return u

class Simulator:
    def __init__(self, m=1., g=9.8, b=2., k=1.):

        # data storage
        self.ts, self.qs, self.dqs = [], [], []
        
        # setup robot
        self.planner = Planner(x_d = 0.7, z_d=0.5) 
        self.control = Controller(m, g, b, k)
        self.robot = Robot(m, g)
     
    def run(self, dt=0.01, sim_time=60.):
        ts = np.arange(0, sim_time, dt)
        q_r, q_r_dot = 0., 0.
        q, dq = np.array([[0.], [0.]]), np.array([[0.2], [0.2]])
        
        for t in ts:               
            q_d, dq_d, ddq_d = self.planner.step(q, dq, dt)
            u = self.control.step(q,dq,q_d,dq_d,ddq_d, dt)
            q, dq = self.robot.step(q, dq,u,dt)
            self.ts.append(t.copy()), self.qs.append(q.copy()), self.dqs.append(dq.copy())

    def plot(self, fig=None, ax=None, label=None):
        if fig is None:
            fig, ax = plt.subplots()
        qs = np.concatenate(self.qs, axis=1)
        ax.grid()
        ax.plot(qs[1,:], qs[0,:], label=label)

        
if __name__ == '__main__':
    fig, ax = plt.subplots()
    sim_1 = Simulator(m=1., g=9.8, b= 2., k=1.)
    # sim_2 = Simulator(m=1., g=9.8, b= 2., k=1.5)
    sim_1.run()
    # sim_2.run()
    sim_1.plot(fig, ax, label='sim_1-k=1')
    # , sim_2.plot(fig, ax, label='sim_2 - k=1.5')
    plt.legend()
    plt.show()