import cython
from cython.parallel import prange

import numpy as np
cimport numpy as np

from libc.math cimport fabs

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def amplitude_scaling_one_body_mat(
        np.ndarray[double, ndim=2] t,
        np.ndarray[double, ndim=2] f_hh,
        np.ndarray[double, ndim=2] f_pp,
        int m, int n, double tol=1e-10
):
    cdef int a, i
    cdef double divisor, val

    for a in prange(m, nogil=True):
        for i in range(n):
            divisor = f_hh[i, i] - f_pp[a, a]

            if fabs(divisor) < tol:
                continue

            val = t[a, i] / divisor
            t[a, i] = val


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def amplitude_scaling_two_body_mat(
        np.ndarray[double, ndim=2] t,
        np.ndarray[double, ndim=2] f_hh,
        np.ndarray[double, ndim=2] f_pp,
        int m, int n, double tol=1e-10
):
    cdef int a, b, i, j, A, I
    cdef double divisor, val

    for a in prange(m, nogil=True):
        for b in range(m):
            for i in range(n):
                for j in range(n):
                    divisor = f_hh[i, i] + f_hh[j, j] - f_pp[a, a] - f_pp[b, b]

                    if fabs(divisor) < tol:
                        continue

                    A = a * m + b
                    I = i * n + j

                    val = t[A, I] / divisor
                    t[A, I] = val

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def amplitude_scaling_one_body_lambda(
        np.ndarray[double, ndim=2] l,
        np.ndarray[double, ndim=2] f,
        int m, int n, double tol=1e-10
):
    cdef int a, i
    cdef double divisor, val

    for a in prange(m, nogil=True):
        for i in range(n):
            divisor = f[i, i] - f[a + n, a + n]

            if fabs(divisor) < tol:
                continue

            val = l[i, a] / divisor
            l[i, a] = val

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def amplitude_scaling_two_body_lambda(
        np.ndarray[double, ndim=4] l,
        np.ndarray[double, ndim=2] f, int m, int n, double tol=1e-10):
    cdef int a, b, i, j
    cdef double divisor, val

    for a in prange(m, nogil=True):
        for b in range(a, m):
            for i in range(n):
                for j in range(i, n):
                    divisor = f[i, i] + f[j, j] \
                            - f[a + n, a + n] - f[b + n, b + n]

                    if fabs(divisor) < tol:
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
        np.ndarray[double, ndim=2] t,
        np.ndarray[double, ndim=2] f,
        int m, int n, double tol=1e-10
):
    cdef int a, i
    cdef double divisor, val

    for a in prange(m, nogil=True):
        for i in range(n):
            divisor = f[i, i] - f[a + n, a + n]

            if fabs(divisor) < tol:
                continue

            val = t[a, i] / divisor
            t[a, i] = val

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def amplitude_scaling_two_body(
        np.ndarray[double, ndim=4] t,
        np.ndarray[double, ndim=2] h, int m, int n, double tol=1e-10):
    cdef int a, b, i, j
    cdef double divisor, val

    for a in prange(m, nogil=True):
        for b in range(a, m):
            for i in range(n):
                for j in range(i, n):
                    divisor = h[i, i] + h[j, j] \
                            - h[a + n, a + n] - h[b + n, b + n]

                    if fabs(divisor) < tol:
                        continue

                    val = t[a, b, i, j] / divisor

                    t[a, b, i, j] = val
                    t[a, b, j, i] = -val
                    t[b, a, i, j] = -val
                    t[b, a, j, i] = val


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def amplitude_scaling_two_body_sparse(
        np.ndarray[unsigned char, ndim=2] indices,
        np.ndarray[double, ndim=1] data,
        np.ndarray[double, ndim=2] h, int n, double tol=1e-10
):
    cdef int a, b, i, j, index, length
    cdef double divisor
    cdef np.ndarray[unsigned char, ndim=1] a_arr, b_arr, i_arr, j_arr

    a_arr = indices[0]
    b_arr = indices[1]
    i_arr = indices[2]
    j_arr = indices[3]

    length = len(data)

    for index in prange(length, nogil=True):
        a = a_arr[index]
        b = b_arr[index]
        i = i_arr[index]
        j = j_arr[index]

        divisor = h[i, i] + h[j, j] - h[n + a, n + a] - h[n + b, n + b]

        if fabs(divisor) < tol:
            continue

        data[index] = data[index] / divisor
