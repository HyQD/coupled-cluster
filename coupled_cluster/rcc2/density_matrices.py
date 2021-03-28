def compute_one_body_density_matrix(t1, t2, l1, l2, o, v, np, out=None):

    nocc = o.stop
    nvirt = v.stop - nocc

    rho = np.zeros((nocc + nvirt, nocc + nvirt), dtype=t1.dtype)

    rho[: o.stop, : o.stop] += 2 * np.eye(o.stop)
    rho[o, o] -= np.einsum("kjab,baik->ij", l2, t2)
    rho[o, o] -= np.einsum("ja,ai->ij", l1, t1)

    rho[v, o] = 2 * t1
    rho[v, o] -= np.einsum("ak,jkcb,bcij->ai", t1, l2, t2, optimize=True)
    rho[v, o] -= np.einsum("ci,jkcb,abjk->ai", t1, l2, t2, optimize=True)
    rho[v, o] -= np.einsum("jb,abji->ai", l1, t2)
    rho[v, o] += 2 * np.einsum("jb,abij->ai", l1, t2)
    rho[v, o] -= np.einsum("jb,aj,bi->ai", l1, t1, t1, optimize=True)

    rho[o, v] = l1

    rho[v, v] = np.einsum("ia,bi->ab", l1, t1)
    rho[v, v] += np.einsum("ijac,bcij->ab", l2, t2)

    return rho
