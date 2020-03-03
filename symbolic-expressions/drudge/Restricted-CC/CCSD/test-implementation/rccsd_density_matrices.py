import numpy as np


def one_body_density_matrix(l1, l2, t1, t2):
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    o, v = slice(0, nocc), slice(nocc, nvirt + nocc)

    rho = np.zeros((nocc + nvirt, nocc + nvirt))

    rho[o, o] = 2 * np.eye(nocc)
    rho[o, o] -= 2 * np.einsum("kjab,baik->ij", l2, t2)
    rho[o, o] -= np.einsum("ja,ai->ij", l1, t1)

    rho[v, o] = 2 * t1
    rho[v, o] -= 2 * np.einsum("ak,jkcb,bcij->ai", t1, l2, t2, optimize=True)
    rho[v, o] -= 2 * np.einsum("ci,jkcb,abjk->ai", t1, l2, t2, optimize=True)
    rho[v, o] -= np.einsum("jb,abji->ai", l1, t2)
    rho[v, o] += 2 * np.einsum("jb,abij->ai", l1, t2)
    rho[v, o] -= np.einsum("jb,aj,bi->ai", l1, t1, t1, optimize=True)

    rho[o, v] = l1

    rho[v, v] = np.einsum("ia,bi->ab", l1, t1)
    rho[v, v] += 2 * np.einsum("ijac,bcij->ab", l2, t2)

    return rho
