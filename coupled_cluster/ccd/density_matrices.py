def compute_one_body_density_matrix(t, l, o, v, rho=None, np=None):
    if rho is None:
        rho = np.zeros((v.stop, v.stop), dtype=t.dtype)

    if np is None:
        import numpy as np

    rho.fill(0)
    rho[o, o] += np.eye(o.stop)
    rho[o, o] -= 0.5 * np.tensordot(l, t, axes=((0, 2, 3), (2, 0, 1)))

    rho[v, v] += 0.5 * np.tensordot(t, l, axes=((1, 2, 3), (3, 0, 1)))

    return rho
