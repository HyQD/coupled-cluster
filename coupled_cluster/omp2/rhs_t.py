from opt_einsum import contract


def compute_t_2_amplitudes(f, u, t, o, v, np, out=None):

    nocc = t.shape[2]
    nvirt = t.shape[0]

    r_T2 = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t.dtype)
    r_T2 += u[v, v, o, o]

    Pij = contract("abik,kj->abij", t, f[o, o])
    r_T2 -= Pij
    r_T2 += Pij.swapaxes(2, 3)

    Pab = contract("ac,cbij->abij", f[v, v], t)
    r_T2 += Pab
    r_T2 -= Pab.swapaxes(0, 1)

    return r_T2


def compute_l_2_amplitudes(f, u, t, l, o, v, np, out=None):
    return t.T.conj()
