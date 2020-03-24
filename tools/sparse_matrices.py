import cvxopt
import numpy as np

m = 8
n = 44
action_space = 4
A_r = np.zeros((m, m, n))
A_t = np.zeros((n, m, n))

for i in range(m):
    for j in range(n):
        A_r[i, i, j] = 1

for i in range(n):
    for j in range(m):
        A_t[i, j, i] = 1

size = m*n*4*4
A = np.concatenate((A_r.reshape((m, m*n)), A_t.reshape((n, m*n))), axis=0)
A_full = cvxopt.sparse([A])