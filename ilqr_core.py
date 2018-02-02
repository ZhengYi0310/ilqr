import six
import abc
import numpy as np
from scipy import linalg
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

class iLQR(BaseTrajOptimizer):
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

        self.use_second_order_ = (use_second_order and self.dynamics.has_hessians)
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

    def _dynamics_rollout(self, x0, us):
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

    def _forward_pass(self, xs, us, k, K, alpha=1):
        """
        apply the updated control signal and rollout a trajecotry
        :param xs: Norminal state seq [N+1, state_dimension] 
        :param us: Norminal control seq [N, control_dimension]
        :param k:  Feedforward gains [N, control_dimension]
        :param K:  Feedbackword gains [N, control_dimension, state_dimension]
        :param alpha: Line search coefficient 
        :return: 
        """
        assert np.shape(us)[0] == self.N, 'The length of control sequence should {}, yet is ' \
                                          '{} instead.'.format(self.N, np.shape(us)[0])
        assert np.shape(xs)[0] == self.N + 1, 'The length of state sequence should {}, yet is ' \
                                              '{} instead.'.format(self.N + 1, np.shape(xs)[0])
        xs_new = np.zeros_like(xs)
        us_new = np.zeros_like(us)
        xs_new[0] = xs[0].copy()

        for i in range(0, self.N ):
            us_new[i] = us[i] + alpha * k[i] + np.dot(K[i], xs_new[i] - xs[i])
            xs_new[i+1] = self.dynamics.fx(xs_new[i], us_new[i], i)

    def _backward_pass(self, us, xs, control_limit=None):
        """
        Backsweep to compute the feed-forward and feed-backward gain k and K
        :param us: control seq [N, control_dimension]
        :param xs: state seq [N+1, state_dimension] 
        :return: 
        """
        k = np.zeros_like(self.k_)
        K = np.zeros_like(self.K_)
        V_x = self.cost.l_x(xs[-1], None, self.N, terminal=True)
        V_xx = self.cost.l_xx(xs[-1], None, self.N, terminal=True)

        for i in range(self.N - 1, -1, -1):
            x = xs[i]
            u = us[i]

            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._compute_Q_terms(x, u, V_x, V_xx, i)
            Q_uu_inv = linalg(Q_uu) #TODO use svd to avoid singularity ?
            if control_limit is None:
                # the regularized version
                k[i] = np.dot(-Q_uu_inv, Q_u)
                K[i] = np.dot(-Q_uu_inv, Q_ux)

                V_x = Q_x + np.dot(np.dot(K[i].T, Q_uu), k[i]) + np.dot(K[i].T, Q_u) \
                      + np.dot(Q_ux.T, k[i])

                V_xx = Q_xx + np.dot(np.dot(K[i].T, Q_uu), K[i]) + np.dot(K[i].T, Q_ux) \
                       + np.dot(Q_ux.T, K[i])
                V_xx = 0.5 * (V_xx +  V_xx.T)
            else:
                #TODO use newton projection methods to solve the QP
                pass
        return k, K

    def _compute_Q_terms(self, x, u, V_x, V_xx, i):
        """
        Compute the quadratic approximation for Q 
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param V_x: d/dx of the value function of the next time stamp [state_dimension]
        :param V_xx: d^2/dx^2 of the value function of the next time stamp [state_dimension, state_dimension]
        :param i: current time stamp
        :return: 
        """
        f_x = self.dynamics.f_x(x, u, i)
        f_u = self.dynamics.f_u(x, u, i)

        l_x = self.cost.l_x(x, u, i)
        l_u = self.cost.l_u(x, u, i)
        l_xx = self.cost.l_xx(x, u, i)
        l_ux = self.cost.l_ux(x, u, i)
        l_uu = self.cost.l_uu(x, u, i)

        Q_x = l_x + np.dot(f_x.T, V_x)
        Q_u = l_u + np.dot(f_u.T, V_x)
        Q_xx = l_xx + np.dot(np.dot(f_x.T, V_xx), f_x)

        reg = V_xx + self.mu_ * np.eye(self.dynamics.state_dimension)
        Q_ux = l_ux + np.dot(np.dot(f_u.T, reg), f_x)
        Q_uu = l_uu + np.dot(np.dot(f_u.T, reg), f_u)

        if self.use_second_order_:
            f_xx = self.dynamics.f_xx(x, u, i)
            f_ux = self.dynamics.f_ux(x, u, i)
            f_uu = self.dynamics.f_uu(x, u, i)
            #TODO CAREFUL!! IT'S TENSOR PRODUCT HERE!!
            Q_xx += np.dot(V_x, f_xx)
            Q_ux += np.dot(V_x, f_ux)
            Q_uu += np.dot(V_x, f_uu)
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    def _trajectory_cost(self, xs, us):
        """
        Compute the cost along one specific trajectory roll-out
        :param us: control seq [N, control_dimension]
        :param xs: state seq [N+1, state_dimension] 
        :return: 
        """
        J = map(lambda args: self.cost.l(*args), zip(xs[-1], us, range(0, self.N)))
        J = sum(J) + self.cost.l(xs[-1], None, self.N, terminal=True)
        return J

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
        xs = self._dynamics_rollout(x0, us)
        J_opt = self._trajectory_cost(us, xs)
        #k = self.k_
        #K = self.K_

        converged = False
        for iteration in range(0, n_interations):
            accepted = False
            try:
                k, K = self._backward_pass(us, xs)

                #Backtracking Line search
                for alpha in alphas:
                    xs_new, us_new = self._forward_pass(xs, us, k, K, alpha)
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
                        self.delta_ = min(1., self.delta_0_) / self.delta_
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
                self.delta_ = max(self.delta_, 1.) * self.delta_0_
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
print np.shape(alphas)
print np.shape(alphas[:-1])
a = np.zeros_like(alphas)
a[0] = alphas[0]


b = range(-100, 0)
zipped = zip(a, b)
print type(a)