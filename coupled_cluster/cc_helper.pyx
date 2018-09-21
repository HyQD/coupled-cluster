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
        for b in range(a, m):
            for i in range(n):
                for j in range(i, n):
                    divisor = f[i, i] + f[j, j] \
                            - f[a + n, a + n] - f[b + n, b + n]

                    if abs(divisor) < tol:
                        continue

                    val = l[i, j, a, b] / divisor

                    l[i, j, a, b] = val
                    l[i, j, b, a] = -val
                    l[j, i, a, b] = -val
                    l[j, i, b, a] = val


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
    cdef np.complex128_t divisor, val

    for a in prange(m, nogil=True):
        for b in range(a, m):
            for i in range(n):
                for j in range(i, n):
                    divisor = h[i, i] + h[j, j] \
                            - h[a + n, a + n] - h[b + n, b + n]

                    if abs(divisor) < tol:
                        continue

                    val = t[a, b, i, j] / divisor

                    t[a, b, i, j] = val
                    t[a, b, j, i] = -val
                    t[b, a, i, j] = -val
                    t[b, a, j, i] = val
