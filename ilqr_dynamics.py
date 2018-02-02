import six
import abc
import numpy as np



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