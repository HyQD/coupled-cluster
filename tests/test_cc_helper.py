import numpy as np
from coupled_cluster.cc_helper import compute_reference_energy


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
