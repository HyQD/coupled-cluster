def compute_eta(h, u, rho_qp, rho_qspr, o, v, np):

    eta = np.zeros(h.shape, dtype=np.complex128)

    rho_oo = rho_qp[o, o]
    rho_vv = rho_qp[v, v]

    n_o, U = np.linalg.eig(rho_oo)
    n_v, V = np.linalg.eig(rho_vv)

    Uinv = np.linalg.inv(U)
    Vinv = np.linalg.inv(V)

    N_vo = n_v.reshape(-1, 1) - n_o
    N_ov = n_o.reshape(-1, 1) - n_v

    R_jb = compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np=np)
    R_bj = compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np=np)

    G_tilde_ov = np.dot(Uinv, np.dot(R_jb, V)) / N_ov
    G_tilde_vo = np.dot(Vinv, np.dot(R_bj, U)) / N_vo

    eta[o, v] = -1j * np.dot(U, np.dot(G_tilde_ov, Vinv))
    eta[v, o] = -1j * np.dot(V, np.dot(G_tilde_vo, Uinv))

    return eta


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
