from opt_einsum import contract


def compute_eta(h, u, rho_qp, rho_qspr, o, v, np):
    eta = np.zeros(h.shape, dtype=np.complex128)
    A_ibaj = compute_A_ibaj(rho_qp, o, v, np=np)
    R_ia = compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np=np)

    A_iajb = A_ibaj.transpose(0, 2, 3, 1)
    eta_jb = -1j * np.linalg.tensorsolve(A_iajb, R_ia)

    # To make the indices match in the tensorsolve, we treat R_tilde_ai as
    # R_tilde_bj. Thus we get the tensor equation
    #
    #   -i A^{ja}_{bi} eta^{b}_{j} = R_tilde^{a}_{i}
    #
    #    => -i A^{ib}_{aj} eta^{a}_{i} = R_tilde^{b}_{j}.
    #
    # Rephrased as a matrix equation with compound indices in paranthesis we get
    #
    #    => -i A_tilde_{(bj), (ai)} eta_{(ai)} = R_tilde_{(bj)}.
    #
    # We solve this equation for eta_{(ai)} <==> eta^{b}_{j}.
    R_tilde_bj = compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np=np)
    A_bjai = A_ibaj.transpose(1, 3, 2, 0)
    eta_ai = 1j * np.linalg.tensorsolve(A_bjai, R_tilde_bj)

    eta[o, v] += eta_jb
    eta[v, o] += eta_ai

    return eta


def compute_A_ibaj(rho_qp, o, v, np):
    delta_ij = np.eye(o.stop)
    delta_ba = np.eye(v.stop - o.stop)

    A_ibaj = contract("ba, ij -> ibaj", delta_ba, rho_qp[o, o])
    A_ibaj -= contract("ij, ba -> ibaj", delta_ij, rho_qp[v, v])

    return A_ibaj


def compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np):

    R_ia = np.dot(rho_qp[o, o], h[o, v])
    R_ia -= np.dot(h[o, v], rho_qp[v, v])

    R_ia += contract("ijkl, klaj->ia", rho_qspr[o, o, o, o], u[o, o, v, o])
    R_ia += contract("ijbc, bcaj->ia", rho_qspr[o, o, v, v], u[v, v, v, o])
    R_ia += contract("ibjc, jcab->ia", rho_qspr[o, v, o, v], u[o, v, v, v])
    R_ia += contract("ibcj, cjab->ia", rho_qspr[o, v, v, o], u[v, o, v, v])

    R_ia -= contract("ibjk, jkab->ia", u[o, v, o, o], rho_qspr[o, o, v, v])
    R_ia -= contract("ibcd, cdab->ia", u[o, v, v, v], rho_qspr[v, v, v, v])
    R_ia -= contract("ijbk, bkaj->ia", u[o, o, v, o], rho_qspr[v, o, v, o])
    R_ia -= contract("ijkb, kbaj->ia", u[o, o, o, v], rho_qspr[o, v, v, o])

    return R_ia


def compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np):

    R_tilde_ai = np.dot(rho_qp[v, v], h[v, o])
    R_tilde_ai -= np.dot(h[v, o], rho_qp[o, o])

    R_tilde_ai += contract(
        "jkib, abjk->ai", u[o, o, o, v], rho_qspr[v, v, o, o]
    )
    R_tilde_ai += contract(
        "cdib, abcd->ai", u[v, v, o, v], rho_qspr[v, v, v, v]
    )
    R_tilde_ai += contract(
        "bkij, ajbk->ai", u[v, o, o, o], rho_qspr[v, o, v, o]
    )
    R_tilde_ai += contract(
        "kbij, ajkb->ai", u[o, v, o, o], rho_qspr[v, o, o, v]
    )

    R_tilde_ai -= contract(
        "klij, ajkl->ai", rho_qspr[o, o, o, o], u[v, o, o, o]
    )
    R_tilde_ai -= contract(
        "bcij, ajbc->ai", rho_qspr[v, v, o, o], u[v, o, v, v]
    )
    R_tilde_ai -= contract(
        "jcib, abjc->ai", rho_qspr[o, v, o, v], u[v, v, o, v]
    )
    R_tilde_ai -= contract(
        "cjib, abcj->ai", rho_qspr[v, o, o, v], u[v, v, v, o]
    )

    return R_tilde_ai


def compute_R_ia_compact(h, u, rho_qp, rho_qspr, o, v, np):

    """
    The use of ":"-slices leads to unpredictable performance.
    However, the function is kept for testing purposes.
    """

    R_ia = np.dot(rho_qp[o, o], h[o, v])
    R_ia -= np.dot(h[o, v], rho_qp[v, v])

    R_ia += contract("ispr, pras->ia", rho_qspr[o, :, :, :], u[:, :, v, :])
    R_ia -= contract("irqs, qsar->ia", u[o, :, :, :], rho_qspr[:, :, v, :])

    return R_ia


def compute_R_tilde_ai_compact(h, u, rho_qp, rho_qspr, o, v, np):

    """
    The use of ":"-slices leads to unpredictable performance.
    However, the function is kept for testing purposes.
    """

    R_tilde_ai = np.dot(rho_qp[v, v], h[v, o])
    R_tilde_ai -= np.dot(h[v, o], rho_qp[o, o])
    R_tilde_ai += contract(
        "pqir, arpq->ai", u[:, :, o, :], rho_qspr[v, :, :, :]
    )
    R_tilde_ai -= contract(
        "aqrs, rsiq->ai", u[v, :, :, :], rho_qspr[:, :, o, :]
    )

    return R_tilde_ai
