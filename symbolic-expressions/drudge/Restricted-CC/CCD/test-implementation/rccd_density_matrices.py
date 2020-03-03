import numpy as np


def one_body_density_matrix(l2, t2):
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    o, v = slice(0, nocc), slice(nocc, nvirt + nocc)
    rho = np.zeros((nocc + nvirt, nocc + nvirt))

    rho_ij = 2 * np.eye(nocc)
    rho_ij -= 2 * np.einsum("kjab,baik->ij", l2, t2)

    rho_ab = 2 * np.einsum("ijac,bcij->ab", l2, t2)

    rho[o, o] = rho_ij
    rho[v, v] = rho_ab

    return rho
