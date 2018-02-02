import six
import abc
import numpy as np
from scipy import optimize



@six.add_metaclass(abc.ABCMeta)
class Dynamics():
    """Dynamics Model"""
    @property
    @abc.abstractmethod
    def state_dimension(self):
        """State Dimension"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def control_dimension(self):
        """Control Dimension"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def has_hessians(self):
        """Whether the second order expansions are available"""
        raise NotImplementedError

    @abc.abstractmethod
    def f(self, x, u, i):
        """
        x_dot = f(x, u, i)
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: state_next [state_dimension]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_x(self, x, u, i):
        """
        partial derivative of the dynamics w.r.t the state
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: df/dx [state_dimension, state_dimension]
        """
    @abc.abstractmethod
    def f_u(self, x, u, i):
        """
        partial derivative of the dynamics w.r.t the control
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: df/du [state_dimension, control_dimension]
        """

    @abc.abstractmethod
    def f_xx(self, x, u, i):
        """
        second partial derivative of the dynamics w.r.t the state
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: d^2f/dx^2 [state_dimension, state_dimension, state_dimension]
        """

    @abc.abstractmethod
    def f_ux(self, x, u, i):
        """
        second partial derivative of the dynamics w.r.t the state and the control
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: d^2f/dx^2 [state_dimension, control_dimension, state_dimension]
        """
    @abc.abstractmethod
    def f_uu(self, x, u, i):
        """
        second partial derivative of the dynamics w.r.t t the control
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: d^2f/dx^2 [control_dimension, control_dimension, control_dimension]
        """

class FiniteDifferenceDynamics(Dynamics):
    """
    Finite difference approximated dynamics model
    """
    def __init__(self, f, state_dimension, control_dimension, x_eps=None, u_eps=None):
        """
        Construct the finite difference dynamics model.
        :param f: function to be approximated by finite difference
        :param state_dimension: 
        :param control_dimension: 
        :param x_eps: Increment to the state to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
        :param u_eps: Increment to the state to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
        """
        self.f_ = f
        self.state_dimension_ = state_dimension
        self.control_dimension_ = control_dimension

        self.x_eps_ = x_eps if x_eps else np.sqrt(np.finfo(float).eps)
        self.u_eps_ = u_eps if x_eps else np.sqrt(np.finfo(float).eps)

        self.x_eps_hess_ = np.sqrt(self.x_eps_)
        self.u_eps_hess_ = np.sqrt(self.u_eps_)

        super(FiniteDifferenceDynamics, self).__init__()

    @property
    def state_dimension(self):
        """State Dimension"""
        return self.state_dimension_

    @property
    def control_dimension(self):
        """Control Dimension"""
        return self.control_dimension_

    @property
    def has_hessians(self):
        """Whether the second order expansions are available"""
        return True

    def f(self, x, u, i):
        """
        x_dot = f(x, u, i)
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: state_next [state_dimension]
        """
        return self.f_(x, u, i)

    def f_x(self, x, u, i):
        """
        partial derivative of the dynamics w.r.t the state
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: df/dx [state_dimension, state_dimension]
        """
        J = np.vstack([optimize.approx_fprime(x, lambda x : self.f_(x, u, i)[m],
                                              self.x_eps_) for m in range(0, self.state_dimension_)])
        return J

    def f_u(self, x, u, i):
        """
        partial derivative of the dynamics w.r.t the control
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: df/du [state_dimension, control_dimension]
        """
        J = np.vstack([optimize.approx_fprime(u, lambda u: self.f_(x, u, i)[m],
                                              self.u_eps_) for m in range(0, self.state_dimension_)])
        return J

    def f_xx(self, x, u, i):
        """
        second partial derivative of the dynamics w.r.t the state
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: d^2f/dx^2 [state_dimension, state_dimension, state_dimension]
        """
        Q = np.array([
                        [optimize.approx_fprime(x, lambda x: self.f_x(x, u, i)[m, n],
                                              self.u_eps_hess_) for n in range(0, self.state_dimension_)]
                    ] for m in range (0, self.state_dimension_))
        return Q

    def f_ux(self, x, u, i):
        """
        second partial derivative of the dynamics w.r.t the state and the control
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: d^2f/dx^2 [state_dimension, control_dimension, state_dimension]
        """
        Q = np.array([
                         [optimize.approx_fprime(x, lambda x: self.f_u(x, u, i)[m, n],
                                                 self.u_eps_hess_) for n in range(0, self.control_dimension_)]
                     ] for m in range(0, self.state_dimension_))
        return Q

    def f_uu(self, x, u, i):
        """
        second partial derivative of the dynamics w.r.t t the control
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: d^2f/dx^2 [control_dimension, control_dimension, control_dimension]
        """
        Q = np.array([
                         [optimize.approx_fprime(u, lambda u: self.f_u(x, u, i)[m, n],
                                                 self.u_eps_hess_) for n in range(0, self.control_dimension_)]
                     ] for m in range(0, self.control_dimension_))
        return Q



