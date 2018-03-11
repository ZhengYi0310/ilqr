import six
import abc
from scipy import optimize
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
    dynamic model with auto differentiation 
    """

    def __init__(self, f, x, u, i=None, use_second_order=False, **kwargs):
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
        self.i_ = i

        t_inv_inputs = np.hstack([x, u])
        t_inputs = np.hstack([x, u, i])
        self.x_input_ = x
        self.u_input_ = u
        self.t_inputs_ = t_inputs
        self.t_inv_inputs_ = t_inv_inputs

        self.state_dimension_ = len(x)
        self.control_dimension_ = len(u)

        #self._J = jacobian_vector(f, t_inv_inputs, self.state_dimension_, self.control_dimension_)
        self._f = f
        self._f_x = jacobian(self._f, 0)
        self._f_u = jacobian(self._f, 1)
        self._i = i

        self.use_second_order_ = use_second_order
        if use_second_order:
            #self._Q = hessian_vector(f, t_inv_inputs, self.state_dimension_, self.control_dimension_)
            self._f_xx = jacobian(self._f_x, 0)
            self._f_ux = jacobian(self._f_u, 0)
            self._f_uu = jacobian(self._f_u, 1)

        super(AutogradDynamics, self).__init__()

    @property
    def state_dimension(self):
        """State Dimension"""
        return self.state_dimension_

    @property
    def control_dimension(self):
        """Control Dimension"""
        return self.control_dimension_

    @property
    def x(self):
        """The state variables."""
        return self._x_input_

    @property
    def u(self):
        """The control variables."""
        return self._u_input_

    @property
    def i(self):
        """The time step variable."""
        return self.i_

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
        return np.array(self._f(x, u)).reshape((self.state_dimension_, 1))

    def f_x(self, x, u, i):
        """
        partial derivative of the dynamics w.r.t the state
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: df/dx [state_dimension, state_dimension]
        """
        #array_jacobian_state =np.array(self._f_x[0](x, u))
        #assert (self.state_dimension_ == len(self._f_x)), "the state vector has length {}, yet the " \
        #                                                  "returned jacobian has length {}".format(self.state_dimension_, len(self._f_x))
        #for k in range(1, self.state_dimension_):
        #    array_jacobian_state = np.vstack((array_jacobian_state, np.array(self._f_x[k](x, u))))

        return self._f_x(x, u)#np.array(array_jacobian_state)

    def f_u(self, x, u, i):
        """
        partial derivative of the dynamics w.r.t the control
        :param x: state [state_dimension]
        :param u: control [control_dimension]
        :param i: current time stamp
        :return: df/du [state_dimension, control_dimension]
        """
        #array_jacobian_control = np.array(self._f_x[0](x, u))
        #assert (self.state_dimension_ == len(self._f_u)), "the state vector has length {}, yet the " \
        #                                                  "returned jacobian has length {}".format(self.state_dimension_, len(self._f_x))
        #for k in range(1, self.state_dimension_):
        #    array_jacobian_control.vstack((array_jacobian_control, np.array(self._f_x[k](x, u))))
        return self._f_u(x, u)#np.array(array_jacobian_control)

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
            #list_hessian_xx = []
            #assert (self.state_dimension_ == len(self._f_xx)), "the state vector has length {}, yet the " \
            #                                                  "returned jacobian has length {}".format(
            #    self.state_dimension_, len(self._f_xx))
            #for k in range(0, self.state_dimension_):
            #    list_hessian_xx.append(np.array(self._f_xx[k](x, u)))
            return self._f_xx(x, u)#np.array(list_hessian_xx)


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
            #list_hessian_ux = []
            #assert (self.state_dimension_ == len(self._f_ux)), "the state vector has length {}, yet the " \
            #                                                   "returned jacobian has length {}".format(
            #    self.state_dimension_, len(self._f_ux))
            #for k in range(0, self.state_dimension_):
            #    list_hessian_ux.append(np.array(self._f_ux[k](x, u)))
            return self._f_ux(x, u)#np.array(list_hessian_ux)

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
            #list_hessian_uu = []
            #assert (self.state_dimension_ == len(self._f_uu)), "the state vector has length {}, yet the " \
            #                                                   "returned jacobian has length {}".format(
            #    self.state_dimension_, len(self._f_ux))
            #for k in range(0, self.state_dimension_):
            #    list_hessian_uu.append(np.array(self._f_uu[k](x, u)))
            return self._f_uu(x, u)#np.array(list_hessian_uu)




