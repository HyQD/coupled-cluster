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

    A_ibaj = np.einsum("ba, ij -> ibaj", delta_ba, rho_qp[o, o])
    A_ibaj -= np.einsum("ij, ba -> ibaj", delta_ij, rho_qp[v, v])

    return A_ibaj


def compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np):

    R_ia = np.einsum("pa,ip->ia", h[:, v], rho_qp[o, :]) - np.einsum(
        "iq,qa->ia", h[o, :], rho_qp[:, v]
    )
    R_ia -= 0.5 * np.einsum(
        "iqrs,rsaq->ia", u[o, :, :, :], rho_qspr[:, :, v, :]
    )
    R_ia -= 0.5 * np.einsum(
        "pirs,rspa->ia", u[:, o, :, :], rho_qspr[:, :, :, v]
    )
    R_ia += 0.5 * np.einsum(
        "pqra,ripq->ia", u[:, :, :, v], rho_qspr[:, o, :, :]
    )
    R_ia += 0.5 * np.einsum(
        "pqas,ispq->ia", u[:, :, v, :], rho_qspr[o, :, :, :]
    )

    return R_ia


def compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np):
    R_tilde_ai = np.einsum("pi,ap->ai", h[:, o], rho_qp[v, :]) - np.einsum(
        "aq,qi->ai", h[v, :], rho_qp[:, o]
    )
    R_tilde_ai -= 0.5 * np.einsum(
        "aqrs,rsiq->ai", u[v, :, :, :], rho_qspr[:, :, o, :]
    )
    R_tilde_ai -= 0.5 * np.einsum(
        "pars,rspi->ai", u[:, v, :, :], rho_qspr[:, :, :, o]
    )
    R_tilde_ai += 0.5 * np.einsum(
        "pqri,rapq->ai", u[:, :, :, o], rho_qspr[:, v, :, :]
    )
    R_tilde_ai += 0.5 * np.einsum(
        "pqis,aspq->ai", u[:, :, o, :], rho_qspr[v, :, :, :]
    )

    return R_tilde_ai
