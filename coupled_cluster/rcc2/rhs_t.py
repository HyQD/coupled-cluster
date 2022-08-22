# Labelling of the different terms comes from the book "Many-Body Methods in
# Chemistry and Physics" by I. Shavitt and R. J. Bartlett.


# Diagrams for CC2 amplitude equations

from opt_einsum import contract


def compute_t_1_amplitudes(
    F, F_t, W_t, t1, t2, o, v, np, intermediates=None, out=None
):

    """
    if out is None:
        out = np.zeros_like(t_1)
    """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]

    r_T1 = np.zeros((nvirt, nocc), dtype=t1.dtype)
    r_T1 += F_t[v, o]

    r_T1 += 2 * contract("bcij, ajbc->ai", t2, W_t[v, o, v, v])
    r_T1 -= contract("bcij, ajcb->ai", t2, W_t[v, o, v, v])

    r_T1 -= 2 * contract("abjk, jkib->ai", t2, W_t[o, o, o, v])
    r_T1 += contract("abjk, kjib->ai", t2, W_t[o, o, o, v])

    tt = 2 * t2 - t2.swapaxes(2, 3)
    r_T1 += contract("jb, abij->ai", F_t[o, v], tt)

    return r_T1


def compute_t_2_amplitudes(
    F, F_t, W_t, t1, t2, o, v, np, intermediates=None, out=None
):
    nocc = t1.shape[1]
    nvirt = t1.shape[0]

    r_T2 = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t2.dtype)

    r_T2 += W_t[v, v, o, o]

    r_T2 += contract("bc, acij->abij", F[v, v], t2)
    r_T2 += contract("ac, bcji->abij", F[v, v], t2)

    r_T2 -= contract("kj, abik->abij", F[o, o], t2)
    r_T2 -= contract("ki, abkj->abij", F[o, o], t2)

    return r_T2
