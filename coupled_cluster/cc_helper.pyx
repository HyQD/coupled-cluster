import cython
from cython.parallel import prange

import numpy as np
cimport numpy as np


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def amplitude_scaling_one_body_lambda(
        np.ndarray[np.complex128_t, ndim=2] l,
        np.ndarray[np.complex128_t, ndim=2] f,
        int m, int n, double tol=1e-10
):
    cdef int a, i
    cdef np.complex128_t divisor, val

    for a in prange(m, nogil=True):
        for i in range(n):
            divisor = f[i, i] - f[a + n, a + n]

            if abs(divisor) < tol:
                continue

            val = l[i, a] / divisor
            l[i, a] = val

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def amplitude_scaling_two_body_lambda(
        np.ndarray[np.complex128_t, ndim=4] l,
        np.ndarray[np.complex128_t, ndim=2] f, int m, int n, double tol=1e-10):
    cdef int a, b, i, j
    cdef np.complex128_t divisor, val

    for a in prange(m, nogil=True):
        for b in range(m):#a, m):
            for i in range(n):
                for j in range(n):#i, n):
                    divisor = f[i, i] + f[j, j] \
                            - f[a + n, a + n] - f[b + n, b + n]

                    if abs(divisor) < tol:
                        continue

                    l[i, j, a, b] = l[i, j, a, b] / divisor


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def amplitude_scaling_one_body(
        np.ndarray[np.complex128_t, ndim=2] t,
        np.ndarray[np.complex128_t, ndim=2] f,
        int m, int n, double tol=1e-10
):
    cdef int a, i
    cdef np.complex128_t divisor, val

    for a in prange(m, nogil=True):
        for i in range(n):
            divisor = f[i, i] - f[a + n, a + n]

            if abs(divisor) < tol:
                continue

            val = t[a, i] / divisor
            t[a, i] = val

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def amplitude_scaling_two_body(
        np.ndarray[np.complex128_t, ndim=4] t,
        np.ndarray[np.complex128_t, ndim=2] h,
        int m, int n, double tol=1e-10
):
    cdef int a, b, i, j
    cdef np.complex128_t divisor

    for a in prange(m, nogil=True):
        for b in range(m):
            for i in range(n):
                for j in range(n):
                    divisor = h[i, i] + h[j, j] \
                            - h[a + n, a + n] - h[b + n, b + n]

                    if abs(divisor) < tol:
                        continue

                    t[a, b, i, j] = t[a, b, i, j] / divisor
