from __future__ import absolute_import
import six
import abc
import numpy as onp
import autograd.numpy as np
from scipy import optimize
from diff_utils import jacobian_scalar, hessian_scalar

@six.add_metaclass(abc.ABCMeta)
class Cost():
    """Instantaneous Cost Function
    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """
    @abc.abstractmethod
    def l(self, x, u, i, terminal=False):
        """
        Evaluation of the instantaneous cost function
        :param x: current state [state_dimension]
        :param u: current control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_x(self, x, u, i, terminal=False):
        """
        Partial derivative of cost function with respect to x.
        :param x: current state [state_dimension]
        :param u: current control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: dl/dx [state_dimension]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_u(self, x, u, i, terminal=False):
        """
        Partial derivative of cost function with respect to u.
        :param x: current state [state_dimension]
        :param u: control control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: dl/du [state_dimension]
        """

        raise NotImplementedError

    @abc.abstractmethod
    def l_xx(self, x, u, i, terminal=False):
        """
        Second-Order Partial derivative of cost function with respect to x.
        :param x: current state [state_dimension]
        :param u: control control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: d^2l/du^2 [state_dimension, state_dimension]
        """

        raise NotImplementedError

    @abc.abstractmethod
    def l_ux(self, x, u, i, terminal=False):
        """
        Second-Order Partial derivative of cost function with respect to u, x.
        :param x: current state [state_dimension]
        :param u: control control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: d^2l/dudx [control_dimension, state_dimension]
        """

        raise NotImplementedError

    @abc.abstractmethod
    def l_uu(self, x, u, i, terminal=False):
        """
        Second-Order Partial derivative of cost function with respect to u, x.
        :param x: current state [state_dimension]
        :param u: control control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: d^2l/dudu [control_dimension, control_dimension]
        """

        raise NotImplementedError

class FiniteDifferenceCost(Cost):
    """
    Finite difference approximated instantaneous cost
    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """
    def __init__(self, l, l_terminal, state_dimension, control_dimension, x_eps=None, u_eps=None):
        """
        Construct a finite difference cost object
        :param l: instantaneous cost function to be approximated (x, u, i) -> scalar
        :param l_terminal: terminal cost function to be approximated (x, i) -> scalar
        :param state_dimension: state dimension 
        :param control_dimension: control dimension
        :param x_eps: Increment to the state to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
        :param u_eps: Increment to the action to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
        Note:
            The square root of the provided epsilons are used when computing
            the Hessians instead.
        """
        self.l_ = l
        self.l_terminal_ = l_terminal
        self.state_dimension_ = state_dimension
        self.control_dimension_ = control_dimension

        self.x_eps_ = x_eps if x_eps else onp.sqrt(onp.finfo(float).eps)
        self.u_eps_ = u_eps if x_eps else onp.sqrt(onp.finfo(float).eps)

        self.x_eps_hess_ = onp.sqrt(self.x_eps_)
        self.u_eps_hess_ = onp.sqrt(self.u_eps_)

        super(FiniteDifferenceCost, self).__init__()

    def l(self, x, u, i, terminal=False):
        """
        Instantaneous cost function
        :param x: current state [state_dimension]
        :param u: current control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: 
        """
        if terminal:
            return self.l_terminal_(x, i)
        return self.l_(x, u, i)

    def l_x(self, x, u, i, terminal=False):
        """
        Partial derivative of cost function with respect to x.
        :param x: current state [state_dimension]
        :param u: current control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: dl/dx [state_dimension]
        """
        if terminal:
            return optimize.approx_fprime(x, lambda x : self.l_terminal_(x, u, i), self.x_eps_)
        return optimize.approx_fprime(x, lambda x : self.l_(x, u, i), self.x_eps_)

    def l_u(self, x, u, i, terminal=False):
        """
        Partial derivative of cost function with respect to u.
        :param x: current state [state_dimension]
        :param u: control control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: dl/du [control_dimension]
        """
        if terminal:
            return onp.zeros(self.control_dimension_)
        return optimize.approx_fprime(u, lambda u : self.l_(x, u, i), self.x_eps_)

    def l_xx(self, x, u, i, terminal=False):
        """
        Second-Order Partial derivative of cost function with respect to x.
        :param x: current state [state_dimension]
        :param u: control control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: d^2l/du^2 [state_dimension, state_dimension]
        """

        return np.vstack([optimize.approx_fprime(x, lambda x : self.l_x(x, u, i, terminal)[m], self.x_eps_hess_)
                   for m in range(0, self.state_dimension_)])

    def l_ux(self, x, u, i, terminal=False):
        """
        Second-Order Partial derivative of cost function with respect to u, x.
        :param x: current state [state_dimension]
        :param u: control control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: d^2l/dudx [control_dimension, state_dimension]
        """
        if terminal:
            return onp.zeros((self.control_dimension_, self.state_dimension_))
        return np.vstack([optimize.approx_fprime(x, lambda x : self.l_u(x, u, i, terminal)[m], self.x_eps_hess_)
                          for m in range(0, self.state_dimension_)])

    def l_uu(self, x, u, i, terminal=False):
        """
        Second-Order Partial derivative of cost function with respect to u, x.
        :param x: current state [state_dimension]
        :param u: control control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: d^2l/dudu [control_dimension, control_dimension]
        """
        if terminal:
            return onp.zeros((self.control_dimension_, self.state_dimension_))
        return np.vstack([optimize.approx_fprime(u, lambda u : self.l_u(x, u, i, terminal)[m], self.x_eps_hess_)
                          for m in range(0, self.control_dimension_)])

class AutogradCost(Cost):
    """
    Cost function with auto differentiation 
    """

    def __init__(self, l, l_terminal, x, u, i=None, **kwargs):
        """
        Construct an AutogradCost object 
        :param l: cost function to be auto differentiated 
        :param l_terminal: terminal cost function to be approximated 
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param dt: time step variable
        :param kwargs: Additional keyword-arguments to pass to autograd functions 
        """
        self.l_ = l
        self.l_terminal_ = l_terminal
        self.i_ = i

        t_inv_inputs = np.hstack([x, u])
        t_inputs = np.hstack([x, u, i])
        terminal_inputs = np.hstack([x, i])
        self.x_input_ = x
        self.u_input_ = u
        self.t_inputs_ = t_inputs
        self.t_inv_inputs_ = t_inv_inputs

        self.state_dim_ = len(x)
        self.control_dim_ = len(u)

        self._J = jacobian_scalar(l, t_inv_inputs, self.state_dim_, self.control_dim_) #[jacobian_x, jacobian_u]
        self._Q = hessian_scalar(l, t_inv_inputs, self.state_dim_, self.control_dim_) #[hessian_xx, hessian_ux, hessian_uu]

        self._l = l
        self._l_x = self._J[0]
        self._l_u = self._J[1]
        self._l_xx = self._Q[0]
        self._l_ux = self._Q[1]
        self._l_uu = self._Q[2]


        self._J_terminal = jacobian_scalar(l_terminal, t_inputs, self.state_dim_, self.control_dim_)
        self._Q_terminal = hessian_scalar(l_terminal, t_inputs, self.state_dim_, self.control_dim_)


        self._l_terminal = l_terminal
        self._l_x_terminal = self._J_terminal[0] #[jacobian_x]
        self._l_xx_terminal = self._Q_terminal[0] #[hessian_xx]



        super(AutogradCost, self).__init__()

    @property
    def x(self):
        """The state variables."""
        return self.x_input_

    @property
    def u(self):
        """The control variables."""
        return self._u_input_

    @property
    def i(self):
        """The time step variable."""
        return self.i_

    def l(self, x, u, i, terminal=False):
        """
        Evaluation of the instantaneous cost function
        :param x: current state [state_dimension]
        :param u: current control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: 
        """
        if terminal:
            return np.asscalar(self._l_terminal(x, u, i))
        return np.asscalar(self._l_terminal(x, u))


    def l_x(self, x, u, i, terminal=False):
        """
        Partial derivative of cost function with respect to x.
        :param x: current state [state_dimension]
        :param u: current control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: dl/dx [state_dimension]
        """
        if terminal:
            return np.array(self._l_x_terminal(x, u, i))
        return np.array(self._l_x(x, u))

    def l_u(self, x, u, i, terminal=False):
        """
        Partial derivative of cost function with respect to u.
        :param x: current state [state_dimension]
        :param u: control control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: dl/du [control_dimension]
        """
        if terminal:
            return np.zeros(self.control_dim_)
        return np.array(self._l_u(x, u))

    def l_xx(self, x, u, i, terminal=False):
        """
        Second-Order Partial derivative of cost function with respect to x.
        :param x: current state [state_dimension]
        :param u: control control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: d^2l/du^2 [state_dimension, state_dimension]
        """
        if terminal:
            return np.array(self._l_xx_terminal(x, u, i))
        return np.array(self._l_xx_(x, u))

    def l_ux(self, x, u, i, terminal=False):
        """
        Second-Order Partial derivative of cost function with respect to u, x.
        :param x: current state [state_dimension]
        :param u: control control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: d^2l/dudx [control_dimension, state_dimension]
        """
        if terminal:
            return np.zeros((self.control_dim_, self.state_dim_))
        return np.array(self._l_ux(x, u))

    @abc.abstractmethod
    def l_uu(self, x, u, i, terminal=False):
        """
        Second-Order Partial derivative of cost function with respect to u, x.
        :param x: current state [state_dimension]
        :param u: control control [control_dimension]
        :param i: current time stamp
        :param terminal: 
        :return: d^2l/dudu [control_dimension, control_dimension]
        """
        if terminal:
            return np.zeros((self.control_dim_, self.control_dim_))
        return np.array(self._l_uu(x, u))

class QRCost(Cost):
    """Quadratic Regulator Instantaneous Cost."""
    def __init__(self, Q, R, Q_terminal=None, x_goal=None, u_goal=None):
        """
        Construct a QR cost 
        :param Q: Quadratic state cost matrix [state_dimension, state_dimension].
        :param R: Quadratic control cost matrix [control_dimension, control_dimension].
        :param Q_terminal: Terminal quadratic state cost matrix [state_dimension, state_dimension].
        :param x_goal: goal state [state_dimension]
        :param u_goal: goal control [control_dimension]
        """
        self.Q_ = np.array(Q)
        self.R_ = np.array(R)

        if Q_terminal is None:
            self.Q_terminal_ = Q_terminal
        else:
            self.Q_terminal_ = Q_terminal

        if x_goal is None:
            self.x_goal_ = np.zeros(Q.shape[0])
        else:
            self.x_goal_ = np.array(x_goal)

        if u_goal is None:
            self.u_goal_ = np.zeros(R.shape[0])
        else:
            self.u_goal_ = np.array(u_goal)

        assert self.Q_.shape == self.Q_terminal_.shape, "Q & Q_terminal mismatch"
        assert self.Q_.shape[0] == self.Q_.shape[1], "Q must be square"
        assert self.R_.shape[0] == self.R_.shape[1], "R must be square"
        assert self.Q_.shape[0] == self.x_goal_.shape[0], "Q ({} x {}) & x_goal ({}, ) mismatch".\
                                                        format(self.Q.shape[0], self.Q.shape[0], self.x_goal.shape[0])
        assert self.R_.shape[0] == self.u_goal_.shape[0], "R ({} x {}) & u_goal ({}, ) mismatch".\
                                                        format(self.R.shape[0], self.R.shape[0], self.u_goal.shape[0])

        # Precompute some common constants.
        self.Q_plus_Q_T_ = self.Q_ + self.Q_.T
        self.R_plus_R_T_ = self.R_ + self.R_.T
        self.Q_plus_Q_T_terminal_ = self.Q_terminal_ + self.Q_terminal_.T
        super(QRCost, self).__init__()

    def l(self, x, u, i, terminal=False):
        """
        Instantaneous cost function.
        :param x: current state [state_dim].
        :param u: current control [control_dim]
        :param i: current time step
        :param terminal: 
        :return: 
        """
        Q = self.Q_terminal_ if terminal else self.Q
        R = self.R_
        x_diff = x - self.x_goal
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff) * 0.5

        if terminal:
            return squared_x_cost

        u_diff = u - self.u_goal
        return squared_x_cost + u_diff.T.dot(R).dot(u_diff) * 0.5

    def l_x(self, x, u, i, terminal=False):
        """
        Partial derivative of cost function with respect to u.
        :param x: current state [state_dim].
        :param u: current control [control_dim]
        :param i: current time step
        :param terminal: 
        :return: 
        """
        Q_plus_Q_T = self.Q_plus_Q_T_terminal_ if terminal else self.Q_plus_Q_T_
        x_diff = x - self.x_goal
        return x_diff.T.dot(Q_plus_Q_T)

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            dl/du [action_size].
        """
        if terminal:
            return np.zeros_like(self.u_goal_)

        u_diff = u - self.u_goal_
        return u_diff.T.dot(self.R_plus_R_T_)

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        return self.Q_plus_Q_T_terminal_ if terminal else self.Q_plus_Q_T_

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            d^2l/dudx [action_size, state_size].
        """
        return np.zeros((self.R.shape[0], self.Q.shape[0]))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            return np.zeros_like(self.R)

        return self.R_plus_R_T_

class PathQRCost(Cost):

    """Quadratic Regulator Instantaneous Cost for trajectory following."""

    def __init__(self, Q, R, x_path, u_path=None, Q_terminal=None):
        """Constructs a QRCost.
        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            R: Quadratic control cost matrix [action_size, action_size].
            x_path: Goal state path [N+1, state_size].
            u_path: Goal control path [N, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
        """
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.x_path = np.array(x_path)

        state_size = self.Q.shape[0]
        action_size = self.R.shape[0]
        path_length = self.x_path.shape[0]

        if Q_terminal is None:
            self.Q_terminal = self.Q
        else:
            self.Q_terminal = np.array(Q_terminal)

        if u_path is None:
            self.u_path = np.zeros(path_length - 1, action_size)
        else:
            self.u_path = np.array(u_path)

        assert self.Q.shape == self.Q_terminal.shape, "Q & Q_terminal mismatch"
        assert self.Q.shape[0] == self.Q.shape[1], "Q must be square"
        assert self.R.shape[0] == self.R.shape[1], "R must be square"
        assert state_size == self.x_path.shape[1], "Q & x_path mismatch"
        assert action_size == self.u_path.shape[1], "R & u_path mismatch"
        assert path_length == self.u_path.shape[0] + 1, \
            "x_path must be 1 longer than u_path"

        # Precompute some common constants.
        self._Q_plus_Q_T = self.Q + self.Q.T
        self._R_plus_R_T = self.R + self.R.T
        self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T

        super(PathQRCost, self).__init__()

    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        x_diff = x - self.x_path[i]
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

        if terminal:
            return squared_x_cost

        u_diff = u - self.u_path[i]
        return squared_x_cost + u_diff.T.dot(R).dot(u_diff)

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            dl/dx [state_size].
        """
        Q_plus_Q_T = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
        x_diff = x - self.x_path[i]
        return x_diff.T.dot(Q_plus_Q_T)

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            dl/du [action_size].
        """
        if terminal:
            return np.zeros_like(self.u_path)

        u_diff = u - self.u_path[i]
        return u_diff.T.dot(self._R_plus_R_T)

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        return self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            d^2l/dudx [action_size, state_size].
        """
        return np.zeros((self.R.shape[0], self.Q.shape[0]))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            return np.zeros_like(self.R)

        return self._R_plus_R_T
