import pytest
import numpy as np
from coupled_cluster.cc_helper import (
    compute_reference_energy,
    construct_d_t_1_matrix,
    construct_d_t_2_matrix,
)


@pytest.fixture
def random_f_matrix():
    n = 10
    l = 30
    o = slice(0, n)
    v = slice(n, l)

    f = np.random.random((l, l)) + 1j * np.random.random((l, l))

    return f, o, v


def test_d_t_1_matrix(random_f_matrix):
    f, o, v = random_f_matrix

    n = o.stop
    m = v.stop - o.stop

    d_t_1 = construct_d_t_1_matrix(f, o, v, np)
    d_t_1_ver = np.zeros((m, n), dtype=f.dtype)

    for a in range(m):
        for i in range(n):
            d_t_1_ver[a, i] = f[i, i] - f[a + n, a + n]

    np.testing.assert_allclose(d_t_1, d_t_1_ver)


def test_d_t_2_matrix(random_f_matrix):
    f, o, v = random_f_matrix

    n = o.stop
    m = v.stop - o.stop

    d_t_2 = construct_d_t_2_matrix(f, o, v, np)
    d_t_2_ver = np.zeros((m, m, n, n), dtype=f.dtype)

    for a in range(m):
        for b in range(m):
            for i in range(n):
                for j in range(n):
                    d_t_2_ver[a, b, i, j] = (
                        f[i, i] + f[j, j] - f[a + n, a + n] - f[b + n, b + n]
                    )

    np.testing.assert_allclose(d_t_2, d_t_2_ver)


def test_reference_energy():
    n = 10
    l = 20
    o = slice(0, n)
    v = slice(n, l)

    h = np.random.random((l, l)) + 1j * np.random.random((l, l))
    u = np.random.random((l, l, l, l)) + 1j * np.random.random((l, l, l, l))
    u = u + u.transpose(1, 0, 3, 2)
    u = u - u.transpose(0, 1, 3, 2)

    f = h + np.trace(u[:, o, :, o], axis1=1, axis2=3)

    e_ref = compute_reference_energy(f, u, o, v, np=np)

    e_test = np.einsum("ii ->", h[o, o]) + 0.5 * np.einsum(
        "ijij ->", u[o, o, o, o]
    )
    e_test_f = np.einsum("ii ->", f[o, o]) - 0.5 * np.einsum(
        "ijij ->", u[o, o, o, o]
    )

    assert abs(e_ref - e_test) < 1e-10
    assert abs(e_ref - e_test_f) < 1e-10
