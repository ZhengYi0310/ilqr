import gym
import env
from ilqr import iLQR
import numpy as np
from ilqr.ilqr_cost import QRCost


class RecedingHorizonController(object):
    """
    Receding horizon controller for model predictive control
    """

    def __init__(self, x0, u_seq_init, controller, env):
        """

        :param x0:initial state [state_dim_] 
        :param controller: controller to be updated 
        """
        self.xs_ = x0
        self.us_ = u_seq_init
        self.controller_ = controller
        self.env = env

    def set_state(self, x):
        """
        Set the current state of the 
        :param x: current state [state_dim_]
        :return: 
        """
        self.xs_ = x

    def control(self, step_size=1,
                initial_n_iterations=200, subsequent_n_iterations=10,
                *args, **kwargs):
        """
        Compute the optimal control command for every step as a 
        receding horizon planning problem 

        Note: The first iteration will be slow, but the successive ones will be
        significantly faster. (warm start)

        Note: This will automatically move the current controller's state to
        what the dynamics model believes will be the next state after applying
        the entire control path computed. Should you want to correct this state
        between iterations, simply use the `set_state()` method.

        Note: If your cost or dynamics are time dependent, then you might need
        to shift their internal state accordingly.

        :param u_seq_init: initial control command sequence [N, control_dim_] 
        :param step_size: Number of steps between each controller fit. Default: 1.
                          i.e. re-fit at every time step. You might need to increase this
                          depending on how powerful your machine is in order to run this
                          in real-time.
        :param initial_n_iterations: Initial max number of iterations to fit.
                                     Default: 100.
        :param subsequent_n_iterations: Subsequent max number of iterations to
                                        fit. Default: 1.
        :param args: xs: optimal state path [step_size+1, state_dim_].
                         optimal control path [step_size, control_dim_]
        :param kwargs: 
        :return: 
        """
        control_dim_ = self.controller_.dynamics.state_dimension
        n_iterations = initial_n_iterations
        while True:
            xs, us = self.controller_.update_u(self.xs_, self.us_, n_iterations=n_iterations, *args, **kwargs)
            self.env.render()
            self.xs_ = xs[step_size]
            yield xs[:step_size + 1], us[:step_size]

            # Set up next action path seed by simply moving along the current
            # optimal path and appending random unoptimal values at the end.
            obs, _, _, _ = env.step(us[0])
            self.xs_ = env.dynamics.augment_state(obs)
            us_start = us[step_size:]
            us_end = us[-step_size:]
            self.us_ = np.vstack([us_start, us_end])
            n_iterations = subsequent_n_iterations

env = gym.make('CartPoleContinuous-TensorDynamics').env

# Note that the augmented state is not all 0.
x_goal = env.dynamics.augment_state(np.array([0.0, 0.0, 0.0, 0.0]))

# Instantenous state cost.
Q = 100 * np.eye(env.dynamics.state_dimension)
Q[0, 0] = Q[1, 1] = 10.0
Q[4, 4] = 80
# Terminal state cost.
Q_terminal = 100 * np.eye(env.dynamics.state_dimension)

# Instantaneous control cost.
R = np.array([[1.0]])
cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)
N = 15 # only plan 15 steps ahead

x0 = env.dynamics.augment_state(env.reset)
us_init = np.random.uniform(-1, 1, (N, env.dynamics.control_dimension))
ilqr = iLQR(env.dynamics, cost, N)
mpc_controller = RecedingHorizonController(x0, us_init, ilqr, env)
mpc_controller.control()





