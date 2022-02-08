from opt_einsum import contract


def compute_one_body_density_matrix(t1, t2, l1, l2, o, v, np, out=None):

    nocc = o.stop
    nvirt = v.stop - nocc

    rho = np.zeros((nocc + nvirt, nocc + nvirt), dtype=t1.dtype)

    rho[: o.stop, : o.stop] += 2 * np.eye(o.stop)
    rho[o, o] -= contract("kjab,baik->ji", l2, t2)
    rho[o, o] -= contract("ja,ai->ji", l1, t1)

    rho[v, o] = 2 * t1
    rho[v, o] -= contract("ak,jkcb,bcij->ai", t1, l2, t2, optimize=True)
    rho[v, o] -= contract("ci,jkcb,abjk->ai", t1, l2, t2, optimize=True)
    rho[v, o] -= contract("jb,abji->ai", l1, t2)
    rho[v, o] += 2 * contract("jb,abij->ai", l1, t2)
    rho[v, o] -= contract("jb,aj,bi->ai", l1, t1, t1, optimize=True)

    rho[o, v] = l1

    rho[v, v] = contract("ia,bi->ba", l1, t1)
    rho[v, v] += contract("ijac,bcij->ba", l2, t2)

    return rho
