# Finite Difference for optimal trend following trading rules
# It is of the form A * ui_h_n = Bi_h_n,
# where 'i' corresponds to the initial state - 0 or 1,
# h corresponds to the space variable value, h = i*dp
# n corresponds to the time variable value, n = j*dt.
# z = V1 - V0; u(t,p) = Z(T-t,p)
# It is the expected value of the difference of trend following trading strategy with initial state 1 and 0,
# corresponding to long and flat initial positions.
# B is the right hand side vector in the finite difference scheme.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TDMA_Solver import TDMA_solver

## Market Parameters used in paper, DO NOT CHANGE.
lambda1 = 0.36
lambda2 = 2.53
mu1 = 0.18
mu2 = -0.77
sigma = 0.184
Kb = 0.001
Ks = 0.001
r = 0.0679
T = 1

## Market Parameters for adhoc testing
# lambda1 = 1.38
# lambda2 = 1.99
# mu1 = 0.39
# mu2 = -0.43
# sigma = 0.006
# Kb = 0.001
# Ks = 0.001
# r = 4.51
# T = 1

## Market Parameters for Case a in paper
# lambda1 = 0.2
# lambda2 = 30
# mu1 = 0.15
# mu2 = 0.10
# sigma = 0.20
# Kb = 0.0006
# Ks = 0.0006
# r = 0.085
# T = 1

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
N = 600  # number of divisions of the time period T
M = N  # number of divisions of the space R

ttm_arr = np.full((N + 1, 1), np.NaN)  # time to maturity array
pb_boundary_arr = np.full((N + 1, 1), np.NaN)  # buy threshold array
ps_boundary_arr = np.full((N + 1, 1), np.NaN)  # sell threshold array

ttm_arr[0] = 0
pb_boundary_arr[0] = 1
# ps_boundary_arr[0] = 1
ps_boundary_arr[0] = (r - mu2 + 0.5 * (sigma ** 2)) / (mu1 - mu2)

dt = T / N  # N*dt = T
dp = R / M  # M*dp = R
i_rng = np.arange(1, M)

factor1 = dt * ((mu1 - mu2) * i_rng * (1 - i_rng * dp) / sigma) ** 2
factor2 = (dt / dp) * ((lambda1 + lambda2) * i_rng * dp - lambda2)

# three Diagonal vectors/arrays for forming the tri-diagonal matrix in the finite difference scheme
a = -0.5 * factor1
b = 1 + factor1 - factor2
c = -0.5 * factor1 + factor2

p = i_rng * dp
f_p = (mu1 - mu2) * p + mu2 - 0.5 * sigma ** 2  # f(p) as shown in the paper

# u0 and u1 at state i and n=0. (or Time = T - n*dt = T)
# They are arrays of size M-1, even though the space is divided by M creating M+1 divisions.
# The 0th and last value are not taken into the solution as they correspond to p=0 and p=1 which are not included as in the paper.
z_i_n = np.ones(M - 1) * np.log(1 - Ks)

## Loop for solving the finite difference scheme iteratively starting from the maturity
for n in range(1, N + 1):
    z_0 = n * dt * (f_p[0] - r) + np.log(1 - Ks)  # boundary value for z near p=0
    z_end = n * dt * (f_p[-1] - r) + np.log(1 - Ks)  # boundary value for z near p=1

    B_h_n = z_i_n[1:-1] + (f_p[1:-1] - r) * dt

    B_h_n[0] = B_h_n[0] - a[0] * z_0  # adjustment of missing 0th position element in the scheme for z
    B_h_n[-1] = B_h_n[-1] - c[-1] * z_end  # adjustment of missing last position element in the scheme for z

    # finding the value of z at time n+1 using TDMA solver.
    z_i_n_ = TDMA_solver(a, b, c, B_h_n)

    # free boundary condition for z and time n+1
    pb = z_i_n_ - np.log(1 + Kb) < 0  # when pb is true, it is optimal to continue, otherwise buy (switch from V0 to V1)
    ps = z_i_n_ - np.log(1 - Ks) > 0  # when ps is true, it is optimal to continue, otherwise square off (switch from V1 to V0)
    db = pb & ps  # double boundary

    z_i_n_temp = np.maximum(np.minimum(z_i_n_, np.log(1 + Kb)), np.log(1 - Ks))
    z_i_n = np.append(np.append(z_0, z_i_n_temp), z_end)

    ttm_arr[n] = n * dt
    # Filling free boundary values where the value of d0 and d1 switch from 0 to 1 or from 1 to 0 for the first time
    if True not in db:
        ps_boundary_arr[n] = 1
        pb_boundary_arr[n] = 1
    else:
        ps_boundary_arr[n] = p[np.where(db)[0][0]]
        pb_boundary_arr[n] = p[np.where(db)[0][-1]]

time_arr = np.flip(ttm_arr)

plt.figure()
plt.plot(time_arr, pb_boundary_arr, time_arr, ps_boundary_arr)
plt.legend(['Buy threshold', 'Sell threshold'])
plt.xlim([0.0, 1.0])
plt.ylim([0.7, 1.0])
plt.title('Optimal Buy and Sell Boundaries')
plt.ylabel('p')
plt.xlabel('t')
plt.show()
