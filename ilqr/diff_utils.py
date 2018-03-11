
import abc
import sys
import autograd.numpy as np
from autograd import grad, jacobian
import scipy.linalg as sla
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
        list_jacobian_state.append(jacobian_scalar(expr[i], vars, state_dim, control_dim)[0])
        list_jacobian_control.append(jacobian_scalar(expr[i], vars, state_dim, control_dim)[1])
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
        list_hessian_xx.append(hessian_scalar(expr[i], vars, state_dim, control_dim)[0])
        list_hessian_ux.append(hessian_scalar(expr[i], vars, state_dim, control_dim)[1])
        list_hessian_uu.append(hessian_scalar(expr[i], vars, state_dim, control_dim)[2])
    return [list_hessian_xx, list_hessian_ux, list_hessian_uu]

def _state_eq(st, u):
        x, x_dot, theta, theta_dot = st
        force = u[0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        x = u[0] + x_dot
        x_dot = x
        return np.array([x, x_dot, theta, theta_dot])


x = np.array([1., 2., 3., 4.])
print len(x)
x_reshape = np.reshape(x, (-1, 1))
print x_reshape.shape
print x.shape
print (x + x_reshape).shape
u = np.array([2., 0.5, -3])
#print np.vstack((x, u))
fun_vec = lambda x, u : [np.dot(x, x.T) + u[1],  np.dot(u, u.T) * 2 + x[4] * x[2] + x[3] * x[3] - u[1] * u[1] + u[0] * u[2] + u[0] * x[1]]
#for f in fun_vec:
#    print f(x, u)
#print jacobian(fun_vec, 0)
fun = lambda x, u : np.dot(x, x.T)  + np.dot(u, u.T) * 2 + x[4] * x[2] + x[3] * x[3] - u[1] * u[1] + u[0] * u[2] + u[0] * x[1]
#print jacobian(fun, 0)(x, u)
print jacobian(_state_eq, 0)(x, u)
# Optdict = {'maxIter': 100, 'minGrad': 1e-8, 'minRelImprove':1e-8, 'stepDec':0.6,  'minStep': 1e-22, 'Armijo': 0.1, 'print': 0}
#
# lower = np.array([-2., 0.5, -3.])
# temp1 = np.array([1, 0, 1, 0, 1., 2, 1, -3])
# temp = np.hstack([x, u])
#
#
# c = (temp == temp1) & (temp1 >0 )
#print u * lower
#print sla.norm(temp1[c])
# temp1[c] = 10
# print temp1 * c
# print c
# A = np.random.randint(5 ,size=(8,8))
# print A
# c = np.array(c, np.bool_)
# print c
# print np.count_nonzero(c)
# d = np.outer(c, c)
# print d
# print A[d]
# e = np.ones((5,), np.bool_)
# print e


#temp1[(temp == temp1) & (temp1 >0 )] = 3
#temp1[np.where(temp1 == 0)[0]] = 3
#print temp1
# print np.isfinite(temp).all()
# print temp
#temp = np.array([0, 1, 1])
#upper = np.array([-1., 1.5, 0.])
#clamp = lambda x : np.maximum(lower, np.minimum(upper, x))
#print clamp(temp)
i = 5.
#print x.reshape((5, 1))




stacked_input = np.hstack([x, u])


# hessian_result = hessian_scalar(fun, stacked_input, 5, 3)
# hessian_vec_result = hessian_vector(fun_vec, stacked_input, 5, 3)
# jacobian_vec_result = jacobian_vector(fun_vec, stacked_input, 5, 3)
# print jacobian_vec_result[0][0](x, u)
# print jacobian_vec_result[0][1](x, u)
# print jacobian_vec_result[1][0](x, u)
# print jacobian_vec_result[1][1](x, u)

# print np.array(hessian_result[1](x, u))
# result = jacobian_scalar(fun, stacked_input, 5 ,3)
# print np.array(result[0](x, u))
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
# a = np.array([[1, 1, 1], [-1, -1, -1]])
# b = np.array([[0, 0, 0], [2, 2, 2]])
# import scipy.spatial.distance as ssd
# #a[:, 1 ,None]
# #print ssd.cdist(a[:, 1 ,None], b[:, 1, None], 'sqeuclidean')
# c = ([[-1, -1, -1], [0.5, 0.5, 0.5]])
# d = []
# a = np.array([[1, 1, 1], [-2, -1, -1]])
c = np.array([[0.1, 0.2, 0.1, 0], [0.1, 0.2, 0.1, -1]])
d = np.array([[2], [1], [0], [0]])
print np.dot(c, d).shape
# print np.hstack([a, c])
#
# from sklearn.cluster import KMeans
# import time
# start_time = time.time()
# X = np.array([[-0.4, 0.9], [0.01, 4], [12, 9],[4, 2], [4, 4], [4, 0]])
# print X[0, :].shape
# e = np.ones((6,), np.bool_)
# e[2] = False
# e[3] = False
# print X[e, :]
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# print("Updating DTC GP dynamics model...., takes about {:.5}s".format(str(time.time() - start_time)))
# print kmeans.cluster_centers_
# import logging
# logging.debug('Hi from myfunc')
#
# d = np.ones((5,1))
# print d.shape
#
# u = np.array([1., 0.5, -3.])
# active = np.ones(3, dtype=bool)
# active[0] = 0
# print(u[active])
#
# #print(c/u)
#
#
#
# #print a[:, None,:].shape
# #print b[None, :, :].shape
# e =  (a[:, None, :] - b[None, :, :])
# #print e
# #print e / u
#
# #print 1 - np.sum(e**2, axis=-1)
# '''
# print np.array(d).shape
# print slice(0, 5, 1)
# '''
# import scipy.spatial.distance as ssd
# from scipy import array, linalg, dot
# a = array([[1,2],[2,5]])
# #print np.inner(a, a)
#
# L_u = linalg.cholesky(a)
# L_l = linalg.cholesky(a, lower=True)
# print L_u
# print L_l
# print(dot(L_u.T, L_u))
# print(dot(L_l, L_l.T))
#
# c = np.array([[0.1, 0.2, 0.1, 8], [-0.1, 0.2, -0.1, 3], [0, 0.03, -0.05, 2], [0, 0, 0 ,1]])
# print np.diag(np.diag(c))

