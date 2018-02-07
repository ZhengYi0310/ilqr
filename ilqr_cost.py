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
            return onp.asscalar(self._l_terminal(x, u, i))
        return onp.asscalar(self._l_terminal(x, u))


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
            return self._l_x_terminal(x, u, i)
        return self._l_x(x, u)

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
            return onp.zeros(self.control_dim_)
        return self._l_u(x, u)

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
            return self._l_xx_terminal(x, u, i)
        return self._l_xx_(x, u)

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
            return onp.zeros((self.control_dim_, self.state_dim_))
        return self._l_ux(x, u)

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
            return onp.zeros((self.control_dim_, self.control_dim_))
        return self._l_uu(x, u)