import numpy as np
from ilqr_dynamics import AutogradDynamics

class CartPoleDynamics(AutogradDynamics):
    """cartpole autograd dynamics """
    def __init__(self, dt, max_control=20, mc=1.0, mp=0.1, l=0.5, g=9.8, **kwargs):
        """
        CartPole Dynamics
        :param dt: Time step [s].
        :param c_l: Minimum bounds for control action [N].
        :param c_u: Maximum bounds for control action [N].
        :param mc:  Cart mass [kg]
        :param mp: Pendulum mass [kg].
        :param l: Pendulum length [m].
        :param g: Gravity acceleration [m/s^2].
        :param kwargs: Additional key-word arguments to pass to the
                       AutogradDynamics constructor.
                       
        Note:
        state: [x, x', sin(theta), cos(theta), theta']
        action: [F]
        theta: 0 is pointing down and increasing counter-clockwise. 
        """

        self.gravity = g
        self.masscart = mc
        self.masspole = mp
        self.total_mass = (self.masspole + self.masscart)
        self.length = l # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.upper_bound = max_control
        self.lower_bound = max_control * -1
        self.tau = dt  # seconds between state updates
        self.state = None

        x_init = 0
        x_dot_init = 0
        sin_theta_init = 0
        cos_theta_init = 1
        theta_dot_init = 0
        x_inputs = np.array([x_init, x_dot_init, sin_theta_init, cos_theta_init, theta_dot_init])
        u_inputs = np.array([0])

        def _state_eq(st, u):
            x, x_dot, theta, theta_dot = st
            force = u[0]
            costheta = np.cos(theta)
            sintheta = np.sin(theta)
            temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            return np.array([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])

        super(CartPoleDynamics, self).__init__(lambda x, u: _state_eq(x, u), x_inputs, u_inputs, **kwargs)

    @classmethod
    def augment_state(cls, state):
        """Augments angular state into a non-angular state by replacing theta
        with sin(theta) and cos(theta).
        In this case, it converts:
            [x, x', theta, theta'] -> [x, x', sin(theta), cos(theta), theta']
        Args:
            state: State vector [reducted_state_size].
        Returns:
            Augmented state size [state_size].
        """
        if state.ndim == 1:
            x, x_dot, theta, theta_dot = state
        else:
            x = state[:, 0].reshape(-1, 1)
            x_dot = state[:, 1].reshape(-1, 1)
            theta = state[:, 2].reshape(-1, 1)
            theta_dot = state[:, 3].reshape(-1, 1)

        return np.hstack([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])

    @classmethod
    def reduce_state(cls, state):
        """Reduces a non-angular state into an angular state by replacing
        sin(theta) and cos(theta) with theta.
        In this case, it converts:
            [x, x', sin(theta), cos(theta), theta'] -> [x, x', theta, theta']
        Args:
            state: Augmented state vector [state_size].
        Returns:
            Reduced state size [reducted_state_size].
        """
        if state.ndim == 1:
            x, x_dot, sin_theta, cos_theta, theta_dot = state
        else:
            x = state[:, 0].reshape(-1, 1)
            x_dot = state[:, 1].reshape(-1, 1)
            sin_theta = state[:, 2].reshape(-1, 1)
            cos_theta = state[:, 3].reshape(-1, 1)
            theta_dot = state[:, 4].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([x, x_dot, theta, theta_dot])
