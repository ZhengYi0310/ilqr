import six
import abc
import numpy as np
import scipy.linalg as sla
import logging
import os
import math

def boxQP(H, g, lower, upper, x0=None, *args, **kwargs):
    """
    Minimize 0.5*x'*H*x + x'*g  s.t. lower<=x<=upper
    using projected Newton methods 
    
    :param H: Positive definite matrix (n x n)
    :param g: bias vector (n,)
    :param lower: lower bounds (n,)
    :param upper: upper bounds (n,)
    :param x0: initial guess
    :param args: see below
    :param kwargs: see below
    :return:  x       solution   (n)
              result  result type (roughly, higher is better, see below)
              Hfree   subspace cholesky factor   (n_free * n_free)
              free            set of free dimensions     (n)
    """

    n = H.shape[0]
    clamped = np.zeros((n, ),dtype=np.bool_)
    free = np.zeros((n, ), dtype=np.bool_)
    oldvalue = 0
    result = -1
    gnorm = 0
    nfactor = 0
    total_iter = 0
    trace = []
    H_ff = np.ones((n, n),dtype=np.float64)
    clamp = lambda x: np.maximum(lower, np.minimum(upper, x))


    if x0 is not None:
        assert(x0.shape[0] == n), "boxQP: dimension mismatch between H{} and x{}!".format(str(n)+"x"+str(n), str(n))
        x = clamp(x0)
    else:
        LU = np.hstack([lower, upper])
        assert (np.isfinite(LU).all() == True), "boxQP: invalid inputs, inputs need to be finite!"
        x = (lower + upper) / 2

    # 'maxIter' maximum number of iterations
    # 'minGrad' minimm norm of non-fixed gradient
    # 'minRelImprove' minimum relative improvement
    # 'stepDec' factor for decreasing step size
    # 'minStep' minimum stepsize for line search
    # 'Armijo' Armijo parameter (fraction of linear improvement required)
    # 'print' verbosity
    Optdict = {'maxIter': 100, 'minGrad': 1e-8, 'minRelImprove':1e-8, 'stepDec':0.6,  'minStep': 1e-22, 'Armijo': 0.1, 'verbose': 0}
    if kwargs is not None:
        for key, value in kwargs.iteritems():
            print("reset  {} == {}".format(key, value))
            Optdict[str(key)] = value

    currentvalue = np.dot(x.T, g) + 0.5 * np.dot(np.dot(x.T, H), x)

    if Optdict['verbose'] > 0:
        print('==========\nStarting box-QP, dimension {}, initial value: {}\n'.format(n, currentvalue))

    # main loop

    for iter in range(0, Optdict['maxIter']):


        if result != -1:
            break
        # check relative improvement

        if ((iter > 1) and (oldvalue - currentvalue) < Optdict['minRelImprove'] * abs(oldvalue) and (oldvalue - currentvalue) > 0):
            result = 4
            break
        oldvalue = currentvalue

        # get gradient
        grad = g + np.dot(H, x)

        # find clamped dimensions
        oldclamped = clamped
        clamped = np.zeros((n,), dtype=np.bool_)
        clamped[(x==lower) & (grad > 0)] = True
        clamped[(x==upper) & (grad < 0)] = True
        free = ~clamped

        # check for all clamped
        if (clamped.all()):
            result = 6
            break

        # factorize if clamped has changed
        #if iter == 1:
        #    factorize = True
        #else:
        #    factorize = (oldclamped != clamped).any()

        #if factorize:
        if (not np.all(np.linalg.eigvals(H) > 0)):
            result = -1
            break


        if iter == 1:
            factorize = True
        else:
            factorize = (oldclamped != clamped).any()
        if factorize:
            free_dim_num = np.count_nonzero(free)
            free_outer = np.outer(free, free)
            H_ff = H[free_outer].reshape((free_dim_num, free_dim_num))
            H_ff = sla.cholesky(H_ff, lower=True)
            nfactor += 1

        # check gradient norm
        gnorm = sla.norm(grad[free])
        if gnorm < Optdict['minGrad']:
            result = 5
            break

        # get the search direction
        free_dim_num = np.count_nonzero(free)
        clamped_dim_num = np.count_nonzero(clamped)
        free_clamped_outer = np.outer(free, clamped)
        H_fc = H[free_clamped_outer].reshape((free_dim_num, clamped_dim_num))
        grad_clamped = g[free] + np.dot(H_fc, x[clamped])
        search = np.zeros((n,), np.float64)
        search[free] = sla.cho_solve((H_ff, True), grad_clamped) * -1 - x[free]

        # check for descent direction
        #sdotg = np.sum(search * grad)



        # armijo linesearch
        step = 1
        nstep = 0
        xc = clamp(x + step * search)
        sdotg = np.sum((xc - x) * grad)
        # if sdotg >= 0:
        #     result = 0
        #     print ("free dim: {}".format(free))
        #     print ("can't find a gradient descent direction/n")
        #     print ("x: {}".format(x))
        #     break
        vc = np.dot(xc.T, g) + 0.5 * np.dot(np.dot(xc.T, H), xc)
        while (vc - oldvalue)/(sdotg + 1e-10) < Optdict['Armijo']:
            step = step * Optdict['stepDec']
            nstep = nstep + 1
            xc = clamp(x + step * search)
            vc = np.dot(xc.T, g) + 0.5 * np.dot(np.dot(xc.T, H), xc)
            sdotg = np.sum((xc - x) * grad)
            if step < Optdict['minStep']:
                result = 2
                break

        if Optdict['verbose'] > 0:
            print('iter {}  value {} |g| {}  reduction {}  linesearch {}^{}  n_clamped {}\n'.format(iter, vc, gnorm, oldvalue-vc, Optdict['stepDec'], nstep, sum(clamped)))

        # accept candidate
        x = xc
        #print x
        currentvalue = vc

        if iter >= Optdict['maxIter']:
            result = 1


    results = ['Hessian is not positive definite', #...  result = -1
               'No descent direction found', # result = 0  SHOULD NOT OCCUR
               'Maximum main iterations exceeded', # result = 1
               'Maximum line-search iterations exceeded', # result = 2
               'No bounds, returning Newton point', # result = 3
               'Improvement smaller than tolerance', # result = 4
               'Gradient norm smaller than tolerance', # result = 5
               'All dimensions are clamped'] # result = 6

    if Optdict['verbose'] > 0:
        print('RESULT: {}.\niterations {}  gradient {} final value {}  factorizations {}\n'.format(
        results[result+1], total_iter, gnorm, currentvalue, nfactor))

    return x, result, H_ff, free



kwargs = {'maxIter': 100, 'minGrad': 1e-8,  'minRelImprove':1e-8,  'stepDec':0.9,  'minStep': 1e-22,  'Armijo': 0.1,  'verbose': 1}
n 		= 4
g 		= np.random.randn(n, )
H 		= np.random.randn(n, n)
H 		= np.dot(H.T, H)
lower 	= -np.ones((n, ), dtype=np.float64) * 10
upper 	=  np.ones((n, ), dtype=np.float64) * 10
import time
start_time = time.time()
x, result, H_ff, free = boxQP(H, g, lower, upper, np.random.randn(n, ), **kwargs)
print free
print("solving QP...., takes about {:.5}s".format(str(time.time() - start_time)))
print result