# Finite Difference for optimal trend following trading rules
# It is of the form A * ui_h_n = Bi_h_n,
# where 'i' corresponds to the initial state - 0 or 1,
# h corresponds to the space variable value, h = i*dp
# n corresponds to the time variable value, n = j*dt.
# u0 and u1 is equivalent to V0 and V1 in the paper.
# It is the expected value of the trend following trading strategy with initial state 0 and 1 respectively,
# corresponding to flat and long initial positions.
# B0 and B1 correspond to the right hand side vector in the finite difference scheme for state i = 0 and 1 respectively.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from copy import deepcopy


def TDMA_solver(a, b, c, d):

    """
    Tri-diagonal matrix solver. It is used to solve a system of linear equations of the form Qx = d,
    where x is the solution, d is the constant coefficient on the RHS,
    Q is a tri-diagonal matrix having a,b,c as the diagonal vectors/arrays.
    :return: solution array x.
    """
    # a, b, c are the column vectors for the compressed tri-diagonal matrix, d is the right vector
    a_ = deepcopy(a)
    b_ = deepcopy(b)
    c_ = deepcopy(c)
    d_ = deepcopy(d)
    n = len(d_)

    c_[0] = c_[0] / b_[0]
    d_[0] = d_[0] / b_[0]

    for i in range(1, n - 1):
        temp_var = b_[i] - (a_[i] * c_[i - 1])
        c_[i] = c_[i] / temp_var
        d_[i] = (d_[i] - (a_[i] * d_[i - 1])) / temp_var
    d_[n - 1] = (d_[n - 1] - (a_[n - 1] * d_[n - 2])) / (b_[n - 1] - (a_[n - 1] * c_[n - 2]))

    ans = np.zeros(n)  # solution of the simultaneous equations
    ans[n - 1] = d_[n - 1]
    for i in range(n - 2, -1, -1):
        ans[i] = d_[i] - (c_[i] * ans[i + 1])

    return ans

def get_thresholds(params):


    # # Market Parameters
    lambda1 = params['lambda1']
    lambda2 = params['lambda2']
    mu1 = params['mu1']
    mu2 = params['mu2']
    sigma = params['sigma']
    Kb = params['K']
    Ks = params['K']
    r = params['rho']
    T = 1

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
    ps_boundary_arr[0] = (r - mu2 + 0.5*(sigma**2))/(mu1-mu2)


    dt = T/N
    dp = R/M
    # N*dt = T M*dp = R
    i_rng = np.arange(1,M)

    factor1 = dt*((mu1-mu2)*i_rng*(1-i_rng*dp)/sigma)**2
    factor2 = (dt/dp)*((lambda1+lambda2)*i_rng*dp - lambda2)
    
    # three Diagonal vectors/arrays for forming the tri-diagonal matrix in the finite difference scheme
    a = -0.5*factor1
    b = 1 + factor1 - factor2
    c = -0.5*factor1 + factor2

    p = i_rng * dp
    f_p = (mu1-mu2)*p + mu2 - 0.5*sigma**2

    # u at state i and n=0. (or Time = T - n*dt = T)
    # They are arrays of size M-1, even though the space is divided by M creating M+1 divisions.
    # The 0th and last value are not taken into the solution as they correspond to p=0 and p=1 which are not included as in the paper.
    u0_i_n = np.zeros(M-1)
    u1_i_n = np.ones(M-1)*np.log(1-Ks)
    
    ## Loop for solving the finite difference scheme iteratively starting from the maturity
    for n in range(1, N+1):
        u0_0 = n * dt * (r + lambda2)  # boundary value for u0 near p=0
        u1_0 = n * dt * (f_p[0] + lambda2) + np.log(1 - Ks)  # boundary value for u1 near p=0

        u0_end = n*dt*(r - lambda1)  # boundary value for u0 near p=1
        u1_end = n*dt*(f_p[-1] - lambda1) + np.log(1-Ks)   # boundary value for u1 near p=1

        B0_h_n = u0_i_n[1:-1] + r*dt
        B1_h_n = u1_i_n[1:-1] + f_p[1:-1]*dt

        B0_h_n[0] = B0_h_n[0] - a[0] * u0_0  # adjustment of missing 0th position element in the scheme for u0
        B0_h_n[-1] = B0_h_n[-1] - c[-1] * u0_end  # adjustment of missing last position element in the scheme for u0

        B1_h_n[0] = B1_h_n[0] - a[0] * u1_0  # adjustment of missing 0th position element in the scheme for u1
        B1_h_n[-1] = B1_h_n[-1] - c[-1] * u1_end  # adjustment of missing last position element in the scheme for u1
    
        # finding the value of u0 and u1 at time n+1 using TDMA solver.
        u0_i_n_ = TDMA_solver(a, b, c, B0_h_n)
        u1_i_n_ = TDMA_solver(a, b, c, B1_h_n)

        d0 = (u0_i_n_ - u1_i_n_ + np.log(1+Kb)) < 0    # free boundary condition for u0 and time n+1

        u0_1 = (~ d0) * u0_i_n_  # u0 at time n+1 will be assigned u0_i_n_ when d0 is False
        u0_2 = d0 * (u1_i_n_ - np.log(1 + Kb))  # u0 at time n+1 will be assigned (u1_i_n_ - np.log(1 + Kb)) when d0 is True
        u0_i_n_temp = u0_1 + u0_2
        u0_i_n = np.append(np.append(u0_0, u0_i_n_temp), u0_end)

        d1 = (u1_i_n_ - u0_i_n_ - np.log(1-Ks)) >= 0   # free boundary condition for u1 at time n+1
        
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

    time_arr = np.flip(ttm_arr)

    

    plt.figure()
    plt.plot(time_arr, pb_boundary_arr, time_arr, ps_boundary_arr)
    plt.legend(['Buy threshold', 'Sell threshold'])
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.7, 1.0])
    plt.title('Optimal Buy and Sell Boundaries')
    plt.ylabel('p')
    plt.xlabel('t')
    plt.show()

    # print(len(time_arr))

    # return mode(pb_boundary_arr).mode[0][0], mode(ps_boundary_arr).mode[0][0]
    # return np.median(pb_boundary_arr),np.median(ps_boundary_arr)
    return pb_boundary_arr[120][0], ps_boundary_arr[120][0]


if __name__ == "__main__":
        
    params = {'lambda1': 0.4269883778758232,
     'lambda2': 0.9557474365893148,
     'mu1': 0.1712263615896295,
     'mu2': -0.3639727373977152,
     'sigma': 0.20062754510614522,
     'K': 0.001,
     'rho': 0.03374543103159356}

    print(get_thresholds(params))
