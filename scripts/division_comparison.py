import numpy as np
import numba
import time

from coupled_cluster.cc_helper import (
    amplitude_scaling_one_body,
    amplitude_scaling_two_body,
    amplitude_scaling_one_body_lambda,
    amplitude_scaling_two_body_lambda,
)


n = 20
l = 120
m = l - n
o = slice(0, n)
v = slice(n, l)

f = np.random.random((l, l)).astype(np.complex128)
t = np.random.random((m, m, n, n)).astype(np.complex128)
l = np.random.random((n, n, m, m)).astype(np.complex128)


@numba.njit(cache=True)
def antisymmetrize_t(t, m, n):
    for a in range(m):
        for b in range(a, m):
            for i in range(n):
                for j in range(i, n):
                    t[a, b, j, i] = -t[a, b, i, j]
                    t[b, a, i, j] = -t[a, b, i, j]
                    t[b, a, j, i] = t[a, b, i, j]

    return t


t = antisymmetrize_t(t, m, n)
l = antisymmetrize_t(l, n, m)
t_c = t.copy()
l_c = l.copy()
t_n = np.zeros_like(t_c)
l_n = np.zeros_like(l_c)


d_2 = (
    np.diag(f)[o].reshape(-1, 1)
    + np.diag(f)[o]
    - np.diag(f)[v].reshape(-1, 1, 1, 1)
    - np.diag(f)[v].reshape(-1, 1, 1)
)

d_2_for = np.zeros_like(d_2)


@numba.njit(cache=True)
def build_d_2(d_2_for, f, m, n):
    for a in range(m):
        for b in range(a, m):
            for i in range(n):
                for j in range(i, n):
                    d_2_for[a, b, i, j] = (
                        f[i, i] + f[j, j] - f[a + n, a + n] - f[b + n, b + n]
                    )
                    d_2_for[a, b, j, i] = d_2_for[a, b, i, j]
                    d_2_for[b, a, i, j] = d_2_for[a, b, i, j]
                    d_2_for[b, a, j, i] = d_2_for[a, b, i, j]

    return d_2_for


d_2_for = build_d_2(d_2_for, f, m, n)
np.testing.assert_allclose(d_2, d_2_for, atol=1e-10)

num_iter = 10

for i in range(num_iter):
    t0 = time.time()
    np.divide(t, d_2, out=t_n)
    t1 = time.time()

    print("Numpy (t): {0} sec".format(t1 - t0))

d_2_l = d_2.transpose(2, 3, 0, 1).copy()
for i in range(num_iter):
    t0 = time.time()
    np.divide(l, d_2_l, out=l_n)
    t1 = time.time()

    print("Numpy (l): {0} sec".format(t1 - t0))


for i in range(num_iter):
    t_c = t.copy()
    t0 = time.time()
    amplitude_scaling_two_body(t_c, f, m, n)
    t1 = time.time()

    print("Cython (t): {0} sec".format(t1 - t0))

for i in range(num_iter):
    l_c = l.copy()
    t0 = time.time()
    amplitude_scaling_two_body_lambda(l_c, f, m, n)
    t1 = time.time()

    print("Cython (l): {0} sec".format(t1 - t0))


np.testing.assert_allclose(t / d_2_for, t_n, atol=1e-8)
np.testing.assert_allclose(t / d_2_for, t_c, atol=1e-8)
np.testing.assert_allclose(t_n, t_c, atol=1e-8)

# np.testing.assert_allclose(t / d_2_for, t_n, atol=1e-8)
# np.testing.assert_allclose(t / d_2_for, t_c, atol=1e-8)
np.testing.assert_allclose(l_n, l_c, atol=1e-8)
