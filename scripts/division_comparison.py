import numpy as np
import time

from coupled_cluster.cc_helper import (
    amplitude_scaling_one_body,
    amplitude_scaling_two_body,
)


n = 20
l = 120
m = l - n
o = slice(0, n)
v = slice(n, l)

f = np.random.random((l, l))
t = np.random.random((m, m, n, n))

d_2 = np.diag(f)[o].reshape(-1, 1, 1, 1)


t0 = time.time()
amplitude_scaling_two_body(t, f, m, n)
t1 = time.time()

print("Cython: {0} sec".format(t1 - t0))

t0 = time.time()
