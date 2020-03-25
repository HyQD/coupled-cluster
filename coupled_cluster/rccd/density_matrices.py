def compute_one_body_density_matrix(t2, l2, o, v, np, out=None):

    nocc = o.stop
    nvirt = v.stop - nocc

    rho = np.zeros((nocc + nvirt, nocc + nvirt), dtype=t2.dtype)

    rho[o, o] += 2 * np.eye(nocc)
    rho[o, o] -= np.einsum("kjab,baik->ij", l2, t2)

    rho[v, v] += np.einsum("ijac,bcij->ab", l2, t2)

    return rho


def compute_two_body_density_matrix(t, l, o, v, np, out=None):
    """
    The final two body density matrix should satisfy

        np.einsum('pqpq->', rho_qspr) = N(N-1)

    where N is the number of electrons.
    """
    return np.zeros((v.stop, v.stop, v.stop, v.stop), dtype=t.dtype)
