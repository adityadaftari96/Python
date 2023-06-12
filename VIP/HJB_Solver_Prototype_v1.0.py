"""This is v1.0 version. It ignores the boundary value at p=0 and p=1 in the solution.
It only considers the solution as p -> 0 and p -> 1.

Finite Difference for optimal trend following trading rules
It is of the form A * ui_h_n = Bi_h_n,
where 'i' corresponds to the initial state - 0 or 1,
h corresponds to the space variable value, h = i*dp
n corresponds to the time variable value, n = j*dt.
u0 and u1 is equivalent to V0 and V1 in the paper. u(t,p) = V(T-t,p)
It is the expected value of the trend following trading strategy with initial state 0 and 1 respectively,
corresponding to flat and long initial positions.
B0 and B1 correspond to the right hand side vector in the finite difference scheme for state i = 0 and 1 respectively."""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TDMA_Solver import *

## Market Parameters used in paper, DO NOT CHANGE.
# lambda1 = 0.36
# lambda2 = 2.53
# mu1 = 0.18
# mu2 = -0.77
# sigma = 0.184
# Kb = 0.001
# Ks = 0.001
# r = 0.0679
# T = 1

## Market Parameters for adhoc testing
# params = {'lambda1': 0.36, 'lambda2': 2.53, 'mu1': 0.18,
#           'mu2': -0.77, 'sigma': 0.184, 'K': 0.001, 'rho': 0.072}
# lambda1 = params['lambda1']
# lambda2 = params['lambda2']
# mu1 = params['mu1']
# mu2 = params['mu2']
# sigma = params['sigma']
# Kb = params['K']
# Ks = params['K']
# r = params['rho']
# T = 1

## Market Parameters for Case a in paper
lambda1 = 0.2
lambda2 = 30
mu1 = 0.15
mu2 = 0.10
sigma = 0.20
Kb = 0.0006
Ks = 0.0006
r = 0.085
T = 1

## Market Parameters for Case b in paper
# lambda1 = 20
# lambda2 = 1
# mu1 = 0.20
# mu2 = 0.00
# sigma = 0.45
# Kb = 0.05
# Ks = 0.05
# r = 0.08
# T = 1

## Grid Parameters
R = 1  # space variable (probability in this case) upper limit, p belongs to (0,1)
N = 4000  # number of divisions of the time period T
M = 800  # number of divisions of the space R
forward_diff = False
backward_diff = False

ttm_arr = np.full((N + 1, 1), np.NaN)  # time to maturity array
pb_boundary_arr = np.full((N + 1, 1), np.NaN)  # buy threshold array
ps_boundary_arr = np.full((N + 1, 1), np.NaN)  # sell threshold array

ttm_arr[0] = 0
pb_boundary_arr[0] = 1
ps_boundary_arr[0] = 1
# ps_boundary_arr[0] = (r - mu2 + 0.5 * (sigma ** 2)) / (mu1 - mu2)

dt = T / N  # N*dt = T
dp = R / M  # M*dp = R
i_rng = np.arange(1, M)

p = i_rng * dp
f_p = (mu1 - mu2) * p + mu2 - 0.5 * sigma ** 2  # f(p) as shown in the paper

factor1 = dt * ((mu1 - mu2) * p * (1 - p) / (sigma * dp)) ** 2
factor2 = (dt / dp) * ((lambda1 + lambda2) * p - lambda2)

# three Diagonal vectors/arrays for forming the tri-diagonal matrix in the finite difference scheme
if forward_diff:
    # based on forward difference in space variable
    a = -0.5 * factor1
    b = 1 + factor1 - factor2
    c = -0.5 * factor1 + factor2
elif backward_diff:
    # based on backward difference in space variable
    a = (-0.5 * factor1) - factor2
    b = 1 + factor1 + factor2
    c = -0.5 * factor1
else:
    # based on mixed forward and backward difference in space variable
    # for p <= lambda2/(lambda1 + lambda2), use forward difference.
    # for p > lambda2/(lambda1 + lambda2), use backward difference.
    a = np.where(p <= lambda2 / (lambda1 + lambda2), -0.5 * factor1, (-0.5 * factor1) - factor2)
    b = np.where(p <= lambda2 / (lambda1 + lambda2), 1 + factor1 - factor2, 1 + factor1 + factor2)
    c = np.where(p <= lambda2 / (lambda1 + lambda2), -0.5 * factor1 + factor2, -0.5 * factor1)

# u0 and u1 at state i and n=0. (or Time = T - n*dt = T)
# They are arrays of size M-1, even though the space is divided by M creating M+1 divisions.
# The 0th and last value are not taken into the solution as they correspond to p=0 and p=1 which are not included as in the paper.
u0_i_n = np.zeros(M - 1)
u1_i_n = np.ones(M - 1) * np.log(1 - Ks)

## Loop for solving the finite difference scheme iteratively starting from the maturity
for n in range(1, N + 1):
    print("time step={0}".format(n))
    u0_0 = n * dt * (r + lambda2)  # boundary value for u0 near p=0
    u1_0 = n * dt * (f_p[0] + lambda2) + np.log(1 - Ks)  # boundary value for u1 near p=0

    u0_end = n * dt * (r - lambda1)  # boundary value for u0 near p=1
    u1_end = n * dt * (f_p[-1] - lambda1) + np.log(1 - Ks)  # boundary value for u1 near p=1

    B0_h_n = u0_i_n[1:-1] + r * dt
    B1_h_n = u1_i_n[1:-1] + f_p[1:-1] * dt

    B0_h_n[0] = B0_h_n[0] - a[0] * u0_0  # adjustment of missing 0th position element in the scheme for u0
    B0_h_n[-1] = B0_h_n[-1] - c[-1] * u0_end  # adjustment of missing last position element in the scheme for u0

    B1_h_n[0] = B1_h_n[0] - a[0] * u1_0  # adjustment of missing 0th position element in the scheme for u1
    B1_h_n[-1] = B1_h_n[-1] - c[-1] * u1_end  # adjustment of missing last position element in the scheme for u1

    # finding the value of u0 and u1 at time n+1 using TDMA solver.
    u0_i_n_ = TDMA_solver(a, b, c, B0_h_n)
    u1_i_n_ = TDMA_solver(a, b, c, B1_h_n)

    d0 = (u0_i_n_ - u1_i_n_ + np.log(1 + Kb)) < 0  # free boundary condition for u0 and time n+1

    u0_1 = (~ d0) * u0_i_n_  # u0 at time n+1 will be assigned u0_i_n_ when d0 is False
    u0_2 = d0 * (u1_i_n_ - np.log(1 + Kb))  # u0 at time n+1 will be assigned (u1_i_n_ - np.log(1 + Kb)) when d0 is True
    u0_i_n_temp = u0_1 + u0_2
    u0_i_n = np.append(np.append(u0_0, u0_i_n_temp), u0_end)

    d1 = (u1_i_n_ - u0_i_n_ - np.log(1 - Ks)) >= 0  # free boundary condition for u1 at time n+1

    u1_1 = d1 * u1_i_n_  # u1 at time n+1 will be assigned u1_i_n_ when d1 is True
    u1_2 = (~ d1) * (u0_i_n_ + np.log(1 - Ks))  # u1 at time n+1 will be assigned (u0_i_n_ + np.log(1 - Ks)) when d1 is False
    u1_i_n_temp = u1_1 + u1_2
    u1_i_n = np.append(np.append(u1_0, u1_i_n_temp), u1_end)

    ttm_arr[n] = n * dt
    # Filling free boundary values where the value of d0 and d1 switch from 0 to 1 or from 1 to 0 for the first time
    if sum(d0) == 0:
        pb_boundary_arr[n] = 1
    else:
        pb_boundary_arr[n] = p[np.where(d0 == 1)[0][0]]

    if sum(d1) == 0:
        ps_boundary_arr[n] = 0
    else:
        ps_boundary_arr[n] = p[np.where(d1 == 1)[0][0]]

print("Ps={:.10f}, Pb={:.10f}".format(ps_boundary_arr[-1][0], pb_boundary_arr[-1][0]))

time_arr = np.flip(ttm_arr)

plt.figure()
plt.plot(time_arr, pb_boundary_arr, time_arr, ps_boundary_arr)
plt.legend(['Buy threshold', 'Sell threshold'])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
# tick_values = np.arange(0.85, 1.05, 0.05)
# plt.yticks(tick_values)
plt.title('Optimal Buy and Sell Boundaries')
plt.ylabel('p')
plt.xlabel('t')
plt.show()

