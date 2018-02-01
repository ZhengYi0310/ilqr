import six
import abc
import numpy as np
import os
import math
import warnings

@six.add_metaclass(abc.ABCMeta)
class BaseTrajOptimizer():
    """
    Base trajectory optimizer 
    """
    @abc.abstractmethod
    def update_u(self, x0, u_seq_init, *args, **kwargs):
        """
        
        :param x_0: Initial State [state_dimension]
        :param u_seq_init: initial control sequence [N, control_dimension]
        :param args, kwargs: Additional positional and key-word arguments.
        :return: Tuple of
                    xs: optimal state path [N+1, state_dimension]
                    us: optimal control sequence [N, control_dimention]
        """
        raise NotImplementedError

def iLQR(BaseTrajOptimizer):
    """Finite Horizon Iterative Linear Quadratic Regulator"""
    def __init__(self, sys_dynamics, cost_func, time_steps, max_reg = 1e10, use_second_order=False):
        """
        Construct a iLQR solver 
        :param sys_dynamics: dynamics model of the system
        :param cost_func: cost/reward function 
        :param time_steps: horizon length for the MPC 
        :param second_order: use the second-order expansion of the system
                             dynamics, default it is set to False, therefore 
                             it's iLQR instead of DDP 
        :return: 
        """
        self.dynamics = sys_dynamics
        self.cost = cost_func
        self.N = time_steps

        self.use_Hessians_ = (use_second_order and self.dynamics.has_hessians)
        if use_second_order and not self.dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # regularization terms, see Y.Tassa, T.Erez, E.Todorov, "Synthesis and stablization of
        # complex behaviors through online trajectory optimization"
        self.mu_min_ = 1e-6
        self.mu_max_ = max_reg
        self.mu_ = 1.0
        self.delta_0_ = 2.0
        self.delta_ = self.delta_0_

        self.k_ = np.zeros((self.N, self.dynamics.control_dimension))
        self.K_ = np.zeros((self.N, self.dynamics.control_dimension, self.dynamics.state_dimension))

        #TODO initialize the initial normial trajecotry

        super(iLQR, self).__init__()

    def _forward_rollout(self, x0, us):
        '''
        "Compute a trajectory rollout based on the given dynamics model"
        :param x0: the initial state [state_dimension]
        :param us: the control sequence [N, control dimension]
        :return: the computed state sequence xs [N + 1, state dimension]
        '''
        xs = np.array([x0])
        assert np.shape(us)[0] == self.N, 'The length of control sequence should {}, yet is ' \
                                          '{} instead.'.format(self.N, np.shape(us)[0])
        for i in range(0, self.N):
            xs_next = self.dynamics.fx(xs[-1], us[i], i)
            xs = np.append(xs, xs_next, axis=0)
    
    def update_u(self, x0, u_seq_init, n_interations=200, tolerance=1e-6, on_iteration=None, line_search_steps=10):
        """
        Update the optimzal control sequence
        :param x0: Initial State [state_dimension]
        :param u_seq_init: initial control sequence [N, control_dimension]
        :param n_interations: maximum number of iterations
        :param tolerance: Tolerance. Default: 1e-6.
        :param on_iteration: Callback at the end of each iteration with the
                following signature:
                (iteration_count, xs, us, J_opt, accepted, converged) -> None
                where:
                    iteration_count: Current iteration count.
                    xs: Current state seq.
                    us: Current action seq.
                    J_opt: Optimal cost-to-go.
                    accepted: Whether this iteration yielded an accepted result.
                    converged: Whether this iteration converged successfully.
                Default: None.
        :param line_search_steps: number of line searches after the backwardpass
        :return: 
        """

        #reset the Tikhonov regularization term
        self.mu_ = 1.0
        # Backtracking line search candidates 0 < alpha <= 1. (naively set)
        alphas = 1.1 ** (-np.arange(line_search_steps) ** 2)

        us = u_seq_init.copy()
        xs = self._forward_rollout(x0, us)
        J_opt = self._trajectory_cost(us, xs)
        k = self.k_
        K = self.K_

        converged = False
        for iteration in range(0, n_interations):
            accepted = False
            try:
                k, K = self._backwardpass(us, xs)

                #Backtracking Line search
                for alpha in alphas:
                    xs_new, us_new = self._line_search(xs, us, k, K, alpha)
                    J_opt_new = self._trajectory_cost(xs_new, us_new)

                    # TODO Comparing the actual reduction and expected reduction ?
                    #expected_delta_J = alpha * np.dot(self.k_[:-1].Transpose(), self.Qu_)
                    if J_opt_new < J_opt:

                        if np.abs((J_opt_new - J_opt) / J_opt) < tolerance:
                            converged = True
                        J_opt = J_opt_new
                        xs = xs_new
                        us = us_new

                        # Decrease the regularization term
                        self.delta_ = min(1, self.delta_0_) / self.delta_
                        self.mu_ *= self.delta_
                        if (self.mu_) <= self.mu_min_:
                            self.mu_ = 0
                        accepted = True
                        break
            except np.linalg.LinAlgError as e:
                # Quu was not PD and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))

            if not accepted:
                self.delta_ = max(self.delta_, 1) * self.delta_0_
                self.mu_ = max(self.mu_min_, self.mu_ * self.delta_)
                if self._mu_max and self._mu >= self._mu_max:
                    warnings.warn("exceeded max regularization term")
                    break

            if on_iteration:
                on_iteration(iteration, xs, us, J_opt, accepted, converged)

            if converged:
                break
        self.k_ = k
        self.K_ = K
        self.nominal_xs_ = xs
        self.nominal_us_ = us





alphas = 1.1**(-np.arange(10)**2)
alphas = np.zeros((4,5))
print np.shape(alphas)
print np.shape(alphas[:-1])
a = np.array([0])
print a