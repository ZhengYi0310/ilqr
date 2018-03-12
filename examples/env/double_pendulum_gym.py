import logging
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import six
import abc

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from os import path

logger = logging.getLogger(__name__)


class DoublePendulumEnv(gym.Env):
    """
        double inverted pendulum is a 2-link pendulum with only the first joint actuated
        Intitially, both links point downwards. The goal is to swing the
        to the vertical position and balance there.
        **STATE:**
        The state consists of the sin() and cos() of the two rotational joint
        angles and the joint angular velocities :
        [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
        For both links, and angle of 0 corresponds to both links pointing upwards, 
        and the angle of both links corresponds to the vertical axies, the position rotation is
        counter-clockwise
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, max_torque=10, max_speed1=4 * np.pi, max_speed2=4 * np.pi,
                 torque_noise=None):
        self.m1 = m1  # [kg]
        self.m2 = m2  # [kg]
        self.l1 = l1  # [m]
        self.l2 = l2  # [m]
        self.I1 = 1. / 12. * m1 * math.pow(l1, 2)
        self.I2 = 1. / 12. * m2 * math.pow(l2, 2)
        self.g = 9.8
        self.max_speed1 = max_speed1
        self.max_speed2 = max_speed2
        self.max_torque = max_torque
        self.torque_noise = torque_noise
        self.viewer = None
        self.dt = 0.02

        high = np.array([1.0, 1.0, 1.0, 1.0, self.max_speed1, self.max_speed2])
        low = -high
        # Torque needs to be bounded
        # self.AVAIL_TORQUE = [-1., 0., +1]
        # self.action_space = spaces.Discrete(3)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=low, high=high)
        self.state = None
        # self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = [np.pi, 0, np.pi, 0]  # both links pointing downwards
        # self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        return self._get_ob()

    def _get_ob(self):
        theta1, thetaDot1, theta2, thetaDot2 = self.state
        return np.array([np.cos(theta1), np.sin(theta1), np.cos(theta2), np.sin(theta2), thetaDot1, thetaDot2])

    def _cost(self, state):
        # TODO edit the cost here when needed
        return math.pow(state[0], 2) + math.pow(state[1], 2) \
               + math.pow(state[2], 2) + math.pow(state[3], 2)

    def _step(self, u):
        s = self.state

        # squash the control torque into the bound, based on Marc Peter Deisenroth's Phd thesis
        # torque = self.AVAIL_TORQUE [u]
        torque = self.max_torque * np.sin(u)
        print torque
        # Add noise to the force action
        if self.torque_noise > 0:
            torque += self.torque_noise * np.random.randn()
        # Now augment the state with the torque action so it can
        # be passed to _dsdt
        augmented_s = np.append(s, [torque])
        ns = self._rk4(self._dsdt, augmented_s, [0, self.dt])
        # only care about the final step of integration returned by the integrator
        ns = ns[-1]
        ns = ns[:4]  # omit applied torque
        ns[0] = self._wrap(ns[0], -np.pi, np.pi)
        # TODO how to deal dtheta1 ?
        ns[2] = self._wrap(ns[2], -np.pi, np.pi)
        # TODO how to deal dtheta2
        self.state = ns
        print ns.size
        reward = self._cost(self.state)
        return (self._get_ob(), reward, False, {})

    def _wrap(self, x, m, M):
        diff = M - m
        while x > M:
            x = x - diff
        while x < m:
            x = x + diff
        return x

    def _rk4(self, derivs, y0, interval):
        '''

        :param derivs: returns the derivative of the system and has the signature 
                       ``dy = derivs(yi, ti)``
        :param y0: initial state vector
        :param interval: sample time interval to perform the integration 
        '''
        try:
            Ny = len(y0)
        except TypeError:
            yout = np.zeros((len(interval),), np.float_)
        else:
            yout = np.zeros((len(interval), Ny), np.float_)

        yout[0] = y0
        i = 0

        for i in np.arange(len(interval) - 1):
            thist = interval[i]
            dt = interval[i + 1] - thist
            dt2 = dt / 2.0
            y0 = yout[i]

            k1 = np.asarray(derivs(y0, thist))
            k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2))
            k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2))
            k4 = np.asarray(derivs(y0 + dt * k3, thist + dt))
            yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        return yout

    def _dsdt(self, augmented_s, t):
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        I1 = self.I1
        I2 = self.I2
        g = self.g
        u = augmented_s[-1]
        s = augmented_s[:-1]
        theta1 = s[0]
        dtheta1 = s[1]
        theta2 = s[2]
        dtheta2 = s[3]

        lc1 = 0.5
        lc2 = 0.5

        ###############################
        C1 = u + g * l1 * (0.5 * m1 + m2) * np.sin(theta1) - 0.5 * m2 * l1 * l2 * math.pow(dtheta2, 2) * np.sin(
            theta1 - theta2)
        C2 = 0.5 * m2 * l2 * g * np.sin(theta2) + 0.5 * m2 * l1 * l2 * math.pow(dtheta1, 2) * np.sin(theta1 - theta2)

        A11 = m2 * math.pow(l1, 2) + 0.25 * m1 * math.pow(l1, 2) + I1
        A12 = 0.5 * m2 * l1 * l2 * np.cos(theta1 - theta2)
        A21 = A12
        A22 = 0.25 * m2 * math.pow(l2, 2) + I2
        DetA = A11 * A22 - A12 * A21

        ddtheta1 = (A22 * C1 - A12 * C2) / DetA
        ddtheta2 = (-A21 * C1 - A11 * C2) / DetA

        '''
        d1 = m1 * lc1 ** 2 + m2 * \
                             (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2) \
               + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2

        ddtheta2 = (u + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
                       / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        '''

        ###############################
        return (dtheta1, ddtheta1, dtheta2, ddtheta2, 0.)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        from gym.envs.classic_control import rendering
        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
        if s is None: return None

        # TODO check if the transfomation is correct
        p1 = [-self.l1 * np.sin(s[0]), self.l1 * np.cos(s[0])]

        p2 = [-self.l1 * np.sin(s[0]) - self.l2 * np.sin(s[2]),
              self.l1 * np.cos(s[0]) + self.l2 * np.cos(s[2])]

        xys = np.array([[0, 0], p1, p2])
        thetas = [s[0] + 0.5 * np.pi, s[2] + 0.5 * np.pi]
        for ((x, y), th) in zip(xys, thetas):
            l, r, t, b = 0, 1, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, .8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')




