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
Q_terminal = 100 * np.eye(dynamics.state_dimension)

# Instantaneous control cost.
R = np.array([[5.0]])
print R
cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)

N = 100
x0 = dynamics.augment_state(np.array([0.0, 0.0, np.pi, 0.0]))
us_init = np.random.uniform(-1, 1, (N, dynamics.control_dimension))
ilqr = iLQR(dynamics, cost, N)

J_hist = []

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = dynamics.reduce_state(xs[-1])
    print("iteration", iteration_count, info, J_opt, final_state)


xs, us = ilqr.update_u(x0, us_init, on_iteration=on_iteration)

# Reduce the state to something more reasonable.
xs = dynamics.reduce_state(xs)

def wrapTo2Pi(x):

    xwrap = np.remainder(x, 2 * np.pi)
    mask = np.abs(xwrap) > (np.pi * 2)
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
    return xwrap

#for i in range(0, xs.shape[0]):
#xs[:, 2] = wrapTo2Pi(xs[:, 2])

# print xs[480:, 2:]
# print xs.shape


t = np.arange(N + 1) * dt
x = xs[:, 0]
x_dot = xs[:, 1]
theta = np.unwrap(xs[:, 2])  # Makes for smoother plots.
# print xs[480:, 2:]

theta_dot = xs[:, 3]

plt.figure(0)
plt.plot(theta, theta_dot)
plt.plot(theta[0], theta_dot[0], '-o', color='red', markersize=10)
plt.plot(theta[-1], theta_dot[-1], '-o', color='yellow', markersize=10)
plt.xlabel("theta (rad)")
plt.ylabel("theta_dot (rad/s)")
plt.title("Orientation Phase Plot")
plt.show()

plt.figure(1)
plt.plot(t, theta)
plt.xlabel("time (s)")
plt.ylabel("Orientation (rad)")
plt.title("Orientation path")

plt.figure(2)
plt.plot(t[:-1], us)
plt.xlabel("time (s)")
plt.ylabel("Force (N)")
plt.title("Action path")
plt.show()

plt.figure(3)
plt.plot(t, x)
plt.xlabel("time (s)")
plt.ylabel("Position (m)")
plt.title("Position path")
plt.show()

plt.figure(4)
plt.plot(t, x_dot)
plt.xlabel("time (s)")
plt.ylabel("Velocity (m)")
plt.title("Velocity path")
plt.show()

plt.figure(5)
plt.plot(t, theta_dot)
plt.xlabel("time (s)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Angular velocity path")
plt.show()

plt.figure(6)
plt.plot(J_hist)
plt.xlabel("Iteration")
plt.ylabel("Total cost")
plt.title("Total cost-to-go")
plt.show()