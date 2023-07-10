from copy import deepcopy
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import sys

warnings.filterwarnings("error")


def TDMA_solver(a, b, c, d):
    """
    Tri-diagonal matrix solver. It is used to solve a system of linear equations of the form Qx = d,
    where x is the solution, d is the constant coefficient on the RHS,
    Q is a tri-diagonal matrix having a,b,c as the diagonal vectors/arrays.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    :return: solution array x.
    """
    # a, b, c are the column vectors for the compressed tri-diagonal matrix, d is the right vector
    b_ = deepcopy(b)
    c_ = deepcopy(c)
    d_ = deepcopy(d)
    n = len(d_)

    c_[0] = c_[0] / b_[0]
    d_[0] = d_[0] / b_[0]

    for i in range(1, n):
        denominator = b[i] - (a[i] * c_[i - 1])
        c_[i] = c[i] / denominator
        d_[i] = (d[i] - (a[i] * d_[i - 1])) / denominator

    x = np.zeros(n)  # solution of the simultaneous equations
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - (c_[i] * x[i + 1])

    return x


def TDMA_solver_scipy(a, b, c, d):

    a = np.append(a[1:], [0])
    c = np.append([0], c[:-1])
    n = len(d)

    data = [a, b, c]  # list of all the data
    diags = [-1, 0, 1]  # which diagonal each vector goes into
    A = sparse.spdiags(data, diags, n, n, format='csc')  # create the matrix

    x = spsolve(A, d)  # solve for x, should be all 1's
    return x


class GEPP():
    """
    Gaussian elimination with partial pivoting.
    input: A is an n x n numpy matrix
           b is an n x 1 numpy array
    output: x is the solution of Ax=b
            with the entries permuted in
            accordance with the pivoting
            done by the algorithm
    post-condition: A and b have been modified.
    :return
    """

    def __init__(self, A, b, doPricing=True):
        #super(GEPP, self).__init__()

        self.A = A                      # input: A is an n x n numpy matrix
        self.b = b                      # b is an n x 1 numpy array
        self.doPricing = doPricing

        self.n = None                   # n is the length of A
        self.x = None                   # x is the solution of Ax=b

        self._validate_input()          # method that validates input
        self._elimination()             # method that conducts elimination
        self._backsub()                 # method that conducts back-substitution

    def _validate_input(self):
        self.n = len(self.A)
        if self.b.size != self.n:
            raise ValueError("Invalid argument: incompatible sizes between" +
                             "A & b.", self.b.size, self.n)

    def _elimination(self):
        """
        k represents the current pivot row. Since GE traverses the matrix in the
        upper right triangle, we also use k for indicating the k-th diagonal
        column index.
        :return
        """

        # Elimination
        for k in range(self.n - 1):
            if self.doPricing:
                # Pivot
                maxindex = abs(self.A[k:, k]).argmax() + k
                if self.A[maxindex, k] == 0:
                    raise ValueError("Matrix is singular.")
                # Swap
                if maxindex != k:
                    self.A[[k, maxindex]] = self.A[[maxindex, k]]
                    self.b[[k, maxindex]] = self.b[[maxindex, k]]
            else:
                if abs(self.A[k, k]) <= sys.float_info.epsilon:
                    raise ValueError("Pivot element is zero. Try setting doPricing to True.")
            # Eliminate
            for row in range(k + 1, self.n):
                multiplier = self.A[row, k] / self.A[k, k]
                self.A[row, k:] = self.A[row, k:] - multiplier * self.A[k, k:]
                self.b[row] = self.b[row] - multiplier * self.b[k]

    def _backsub(self):
        # Back Substitution

        self.x = np.zeros(self.n)
        for k in range(self.n - 1, -1, -1):
            self.x[k] = np.float64(self.b[k] - np.dot(self.A[k, k + 1:], self.x[k + 1:])) / self.A[k, k]


def TDMA_solver_GEPP(a, b, c, d):
    a = np.append(a[1:], [0])
    c = np.append([0], c[:-1])
    n = len(d)

    data = [a, b, c]  # list of all the data
    diags = [-1, 0, 1]  # which diagonal each vector goes into
    A = sparse.spdiags(data, diags, n, n, format='csc')  # create the matrix
    A = A.toarray()

    GaussElimPiv = GEPP(np.copy(A), np.copy(d), doPricing=False)
    return GaussElimPiv.x