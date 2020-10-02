def compute_one_body_density_matrix(t, l, o, v, np, out=None):

    nocc = t.shape[2]
    nvirt = t.shape[0]
    nso = v.stop

    opdm = np.zeros((nso, nso), dtype=t.dtype)

    opdm[o, o] += np.identity(o.stop)

    # Build one particle density matrix
    opdm[v, v] += (1 / 2) * np.einsum(
        "ijac,bcij -> ba", t.T.conj(), t, optimize=True
    )
    opdm[o, o] += -(1 / 2) * np.einsum(
        "jkab,abik -> ji", t.T.conj(), t, optimize=True
    )

    return opdm


def compute_two_body_density_matrix(t, l, o, v, np, out=None):

    nocc = t.shape[2]
    nvirt = t.shape[0]
    nso = v.stop

    tpdm = np.zeros((nso, nso, nso, nso), dtype=t.dtype)

    ################################################################
    delta = np.eye(o.stop)

    term = np.einsum("ki, lj -> klij", delta, delta)
    term -= term.swapaxes(2, 3)
    tpdm[o, o, o, o] += term

    term_lj = -0.5 * np.tensordot(t.T.conj(), t, axes=((1, 2, 3), (3, 0, 1)))
    term = np.einsum("ki, lj -> klij", delta, term_lj)
    term -= term.swapaxes(0, 1)
    term -= term.swapaxes(2, 3)
    tpdm[o, o, o, o] += term

    # Complexity: O(n^2 m^3)
    W_ba = 0.5 * np.tensordot(t, t.T.conj(), axes=((1, 2, 3), (3, 0, 1)))
    term_jbia = np.einsum("ji, ba -> jbia", delta, W_ba)

    # rho^{jb}_{ia}
    tpdm[o, v, o, v] += term_jbia
    # rho^{bj}_{ia}
    tpdm[v, o, o, v] -= term_jbia.transpose(1, 0, 2, 3)
    # rho^{jb}_{ai}
    tpdm[o, v, v, o] -= term_jbia.transpose(0, 1, 3, 2)
    # rho^{bj}_{ai}
    tpdm[v, o, v, o] += term_jbia.transpose(1, 0, 3, 2)
    ################################################################

    tpdm[v, v, o, o] += t
    tpdm[o, o, v, v] += t.T.conj()

    return tpdm


def compute_two_body_density_matrix_old(t, l, o, v, np, out=None):
    nocc = t.shape[2]
    nvirt = t.shape[0]
    nso = v.stop

    tpdm_corr = np.zeros((nso, nso, nso, nso), dtype=t.dtype)

    # Build two particle density matrix
    tpdm_corr[v, v, o, o] = t.copy()
    tpdm_corr[o, o, v, v] = t.T.conj()

    opdm_ref = np.zeros((nso, nso), dtype=t.dtype)
    opdm_corr = np.zeros((nso, nso), dtype=t.dtype)

    opdm_ref[o, o] += np.identity(o.stop)

    # Build one particle density matrix
    opdm_corr[v, v] = (1 / 2) * np.einsum(
        "ijac,bcij -> ba", t.T.conj(), t, optimize=True
    )
    opdm_corr[o, o] = -(1 / 2) * np.einsum(
        "jkab,abik -> ji", t.T.conj(), t, optimize=True
    )

    tpdm2 = np.einsum("rp,sq -> rspq", opdm_corr, opdm_ref, optimize=True)
    tpdm3 = np.einsum("rp,sq->rspq", opdm_ref, opdm_ref, optimize=True)

    tpdm = (
        tpdm_corr
        + tpdm2
        - tpdm2.transpose((1, 0, 2, 3))
        - tpdm2.transpose((0, 1, 3, 2))
        + tpdm2.transpose((1, 0, 3, 2))
        + tpdm3
        - tpdm3.transpose((1, 0, 2, 3))
    )

    return tpdm
