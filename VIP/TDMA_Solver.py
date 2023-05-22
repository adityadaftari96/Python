from copy import deepcopy

import numpy as np


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
