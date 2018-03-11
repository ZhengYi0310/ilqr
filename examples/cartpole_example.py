import numpy as np
import matplotlib.pyplot as plt

from ilqr import iLQR
from ilqr.ilqr_cost import QRCost
#from ilqr.dynamics import constrain
from ilqr.cartpole import CartPoleDynamics



dt = 0.05
dynamics = CartPoleDynamics(dt)

# Note that the augmented state is not all 0.
x_goal = dynamics.augment_state(np.array([0.0, 0.0, 0.0, 0.0]))

# Instantenous state cost.
Q = 100 * np.eye(dynamics.state_dimension)
Q[1, 1] = Q[4, 4] = 10.0

# Terminal state cost.
Q_terminal = 100 * np.eye(dynamics.state_size)

# Instantaneous control cost.
R = np.array([[1.0]])

cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)

N = 500
x0 = dynamics.augment_state(np.array([0.0, 0.0, 0.0, 0.0]))
us_init = np.random.uniform(-1, 1, (N, dynamics.action_size))
ilqr = iLQR(dynamics, cost, N)

J_hist = []

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = dynamics.reduce_state(xs[-1])
    print("iteration", iteration_count, info, J_opt, final_state)

    
xs, us = ilqr.fit(x0, us_init, n_iterations=500, on_iteration=on_iteration)

# Reduce the state to something more reasonable.
xs = dynamics.reduce_state(xs)

