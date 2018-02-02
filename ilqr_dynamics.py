import six
import abc
from autograd import grad, jacobian
import numpy as onp
import autograd.numpy as np
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

        self.x_eps_ = x_eps if x_eps else onp.sqrt(onp.finfo(float).eps)
        self.u_eps_ = u_eps if x_eps else onp.sqrt(onp.finfo(float).eps)

        self.x_eps_hess_ = onp.sqrt(self.x_eps_)
        self.u_eps_hess_ = onp.sqrt(self.u_eps_)

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
        :return: d^2f/dx^2 [state_dimension, control_dimension, control_dimension]
        """
        Q = np.array([
                         [optimize.approx_fprime(u, lambda u: self.f_u(x, u, i)[m, n],
                                                 self.u_eps_hess_) for n in range(0, self.control_dimension_)]
                     ] for m in range(0, self.control_dimension_))
        return Q

class AutogradDynamics(Dynamics):
    """
    dynamic model to with auto differentiation 
    """

    def __init__(self, f, x, u, dt=None, use_second_order=False, **kwargs):
        """
        Contruct a dynamics model with auto differentiation 
        :param f: function to be auto differentiated 
        :param x: state [state_dimension]
        :param u: contrl [control_dimension]
        :param dt: time step variable 
        :param use_second_order: Evaluate the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
        :param kwargs: Additional keyword-arguments to pass to autograd functions 
        """
        self.f_ = f
        self.dt_ = dt

        t_inv_inputs = np.hstack([x, u]).tolist()
        t_inputs = np.hstack([x, u]).tolist()
        self.x_input_ = x
        self.u_input_ = u
        self.inputs_ = t_inputs
        self.t_inv_inputs_ = t_inv_inputs

        self.state_dimension_ = len(x)
        self.control_dimension_ = len(u)

        self._J = jacobian_vec(f, t_inv_inputs, self.state_dim_)
        self._f = as_function(f, t_inputs, name='f', **kwargs)
        self._f_x = as_function(self._J[:, :self.state_dim_], t_inputs, name='f_x', **kwargs)
        self._f_u = as_function(self._J[:, self.state_dim_:], t_inputs, name='f_u', **kwargs)

        self.use_second_order_ = use_second_order
        if use_second_order:
            self._Q = hessian_vec(f, t_inv_inputs, self.state_dim_)
            self._f_xx = as_function(self._Q[:, :self.state_dim_, :self.state_dim_], t_inputs, name='f_xx', **kwargs)
            self._f_ux = as_function(self._Q[:, self.state_dim_:, :self.state_dim_], t_inputs, name='f_ux', **kwargs)
            self._f_uu = as_function(self._Q[:, self.state_dim_:, self.state_dim_:], t_inputs, name='f_uu', **kwargs)

        super(AutogradDynamics, self).__init__()

    @property
    def state_dimension(self):
        """State Dimension"""
        return self.state_dim_

    @property
    def control_dimension(self):
        """Control Dimension"""
        return self.control_dimension_

    @property
    def has_hessians(self):
        """Whether the second order expansions are available"""
        return self.use_second_order_

    def f(self, x, u, i):
        """
        x_dot = f(x, u, i)
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: state_next [state_dimension]
        """
        return self._f(*np.hstack([x, u, i]))

    def f_x(self, x, u, i):
        """
        partial derivative of the dynamics w.r.t the state
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: df/dx [state_dimension, state_dimension]
        """
        return self._f_x(*np.hstack([x, u, i]))

    def f_u(self, x, u, i):
        """
        partial derivative of the dynamics w.r.t the control
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: df/du [state_dimension, control_dimension]
        """
        return self._f_u(*np.hstack([x, u, i]))

    def f_xx(self, x, u, i):
        """
        second partial derivative of the dynamics w.r.t the state
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: d^2f/dx^2 [state_dimension, state_dimension, state_dimension]
        """
        if not self.use_second_order_:
            raise NotImplementedError
        else:
            return self._f_xx(*np.hstack([x, u, i]))


    def f_ux(self, x, u, i):
        """
        second partial derivative of the dynamics w.r.t the state and the control
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: d^2f/dx^2 [state_dimension, control_dimension, state_dimension]
        """
        if not self.use_second_order_:
            raise NotImplementedError
        else:
            return self._f_ux(*np.hstack([x, u, i]))

    def f_uu(self, x, u, i):
        """
        second partial derivative of the dynamics w.r.t t the control
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: d^2f/dx^2 [state_dimension, control_dimension, control_dimension]
        """
        if not self.use_second_order_:
            raise NotImplementedError
        else:
            return self._f_uu(*np.hstack([x, u, i]))




def multiply(*args):
    for num in args:
       print(num)
a = np.array((1,2,3)).reshape(3, 1)
b = np.array((2,3,4)).reshape(3, 1)
print(np.hstack([a , b]).tolist())
print(multiply(*(np.hstack([a , b]))))
