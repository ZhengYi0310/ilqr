
import abc
import sys
import autograd.numpy as np
from autograd import grad, jacobian
import math

def jacobian_scalar(expr, vars, state_dim, control_dim):
    """
    Computes the jacobian of a scalar expression w.r.t to (vector) variables
    :param expr: the function expression l(x,u) or l(x,u,i)
    :param vars: a h-stacked version of variables , [x_vec, u_vec, i] or just [x_vec, u_vec] 
    :return: a list of function to compute corresponding derivatives
    """
    num_vars = vars.size
    dim_sum = state_dim + control_dim
    assert (num_vars == dim_sum + 1 or num_vars == dim_sum), "the number of variables can only be {} + {} or {} + {} + 1, " \
                                                             "yet the {} is passed".format(state_dim, control_dim, state_dim, control_dim, num_vars)
    x = vars[:state_dim]

    if (num_vars == dim_sum): # non-terminal cost
        u = vars[state_dim:]
        jacobian_x = jacobian(expr, 0)
        jacobian_u = jacobian(expr, 1)
        return [jacobian_x, jacobian_u]
    else:                     # terminal cost
        u = vars[state_dim:-1]
        jacobian_x = jacobian(expr, 0)
        return [jacobian_x]

def hessian_scalar(expr, vars, state_dim, control_dim):
    """
    Computes the jacobian of a scalar expression w.r.t to (vector) variables
    :param expr: the function expression l(x,u) or l(x,u,i)
    :param wrt: a h-stacked version of variables , [x_vec, u_vec, i] or just [x_vec, u_vec] 
    :return: a list of function to compute corresponding derivatives
    """
    num_vars = vars.size
    dim_sum = state_dim + control_dim
    assert (
    num_vars == dim_sum + 1 or num_vars == dim_sum), "the number of variables can only be {} + {} or {} + {} + 1, " \
                                                     "yet the {} is passed".format(state_dim, control_dim, state_dim,
                                                                                   control_dim, num_vars)

    x = vars[:state_dim]
    if (num_vars == dim_sum): # non-terminal cost 
        u = vars[state_dim:]
        hessian_xx = jacobian(jacobian(expr, 0), 0)
        hessian_ux = jacobian(jacobian(expr, 1), 0)
        hessian_uu = jacobian(jacobian(expr, 1), 1)
        return [hessian_xx, hessian_ux, hessian_uu]
    else:                     # terminal cost
        u = vars[state_dim:-1]
        hessian_xx = jacobian(jacobian(expr, 0), 0)
        return [hessian_xx]

def jacobian_vector(expr, vars, state_dim, control_dim):
    """
    Compute the jacobian of a vector w.r.t (vector) variables
    :param expr: a (vector) list of functions to be differentiated 
    :param vars: a h-stacked version of variables , [x_vec, u_vec] 
    :param state_dim: 
    :param control_dim: 
    :return: 
    """
    num_vars = vars.size
    dim_sum = state_dim + control_dim
    assert (num_vars == num_vars == dim_sum), "the number of variables can only be {} + {}, " \
                                                     "yet the {} is passed".format(state_dim, control_dim, num_vars)
    list_jacobian_state = []
    list_jacobian_control =[]
    for i in range(0, len(expr)):
        list_jacobian_state.append(jacobian_scalar(expr, vars, state_dim, control_dim)[0])
        list_jacobian_control.append(jacobian_scalar(expr, vars, state_dim, control_dim)[0])
    return [list_jacobian_state, list_jacobian_control]

def hessian_vector(expr, vars, state_dim, control_dim):
    """
    Compute the hessian of a vector w.r.t (vector) variables
    :param expr: a (vector) list of functions to be differentiated 
    :param vars: a h-stacked version of variables , [x_vec, u_vec] 
    :param state_dim: 
    :param control_dim: 
    :return: 
    """
    num_vars = vars.size
    dim_sum = state_dim + control_dim
    assert (num_vars == num_vars == dim_sum), "the number of variables can only be {} + {}, " \
                                              "yet the {} is passed".format(state_dim, control_dim, num_vars)
    list_hessian_xx = []
    list_hessian_ux = []
    list_hessian_uu = []
    for i in range(0, len(expr)):
        list_hessian_xx.append(hessian_scalar(expr, vars, state_dim, control_dim)[0])
        list_hessian_ux.append(hessian_scalar(expr, vars, state_dim, control_dim)[0])
        list_hessian_uu.append(hessian_vector(expr, vars, state_dim, control_dim)[0])
    return [list_hessian_xx, list_hessian_ux, list_hessian_uu]









fun = lambda x, u : np.dot(x, x.T)  + np.dot(u, u.T) * 2 + x[4] * x[2] + x[3] * x[3] - u[1] * u[1] + u[0] * u[2] + 1. / x[1] + np.sin(u[1] * x[0])

x = np.array([1., 2., 3., 4., 5.])
u = np.array([1., 0.5, -3.])
i = 5.
#print x.reshape((5, 1))



stacked_input = np.hstack([x, u])


hessian_result = hessian_scalar(fun, stacked_input, 5, 3)
print len(hessian_result)
print np.array(hessian_result[2](x, u))
result = jacobian_scalar(fun, stacked_input, 5 ,3)
#print np.array(result)
#print np.asscalar(np.array([5.]))

'''
jacobian_list = []
for i in range(2):
    jacobian_list[i] = jacobian(fun, i)

print stacked_input.shape[0]
print stacked_input[0]
result = jacobian_fun(stacked_input[0], stacked_input[1])
print result

from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs
import matplotlib.pyplot as plt
def tanh(x):                 # Define a function
    y = np.exp(-x)
    return (1.0 - y) / (1.0 + y)
x = np.linspace(-7, 7, 200)
plt.plot(x, tanh(x),
         x, egrad(tanh)(x),                                     # first  derivative
         x, egrad(egrad(tanh))(x),                              # second derivative
         x, egrad(egrad(egrad(tanh)))(x),                       # third  derivative
         x, egrad(egrad(egrad(egrad(tanh))))(x),                # fourth derivative
         x, egrad(egrad(egrad(egrad(egrad(tanh)))))(x),         # fifth  derivative
         x, egrad(egrad(egrad(egrad(egrad(egrad(tanh))))))(x))  # sixth  derivative
plt.show()
'''
a = np.array([[1, 1, 1], [-1, -1, -1]])
b = np.array([[0, 0, 0], [2, 2, 2]])
c = ([[-1, -1, -1], [0.5, 0.5, 0.5]])
d = []
d.append(np.array(a))
d.append(np.array(b))
d.append(np.array(c))
print np.array(d).shape