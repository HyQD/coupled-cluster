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
    R_ia = np.dot(rho_qp[o, o], h[o, v])
    R_ia -= np.dot(h[o, v], rho_qp[v, v])
    R_ia += 0.5 * np.tensordot(
        # rho^{is}_{pr}
        rho_qspr[o, :, :, :],
        # u^{pr}_{as}
        u[:, :, v, :],
        # axes=((s, p, r), (s, p, r))
        axes=((1, 2, 3), (3, 0, 1)),
    )
    R_ia -= 0.5 * np.tensordot(
        # u^{ir}_{qs}
        u[o, :, :, :],
        # rho^{qs}_{ar}
        rho_qspr[:, :, v, :],
        # axes=((r, q, s), (r, q, s))
        axes=((1, 2, 3), (3, 0, 1)),
    )

    return R_ia


def compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np):
    R_tilde_ai = np.dot(rho_qp[v, v], h[v, o])
    R_tilde_ai -= np.dot(h[v, o], rho_qp[o, o])
    R_tilde_ai += 0.5 * np.tensordot(
        # rho^{as}_{pr}
        rho_qspr[v, :, :, :],
        # u^{pr}_{is}
        u[:, :, o, :],
        # axes=((s, p, r), (s, p, r))
        axes=((1, 2, 3), (3, 0, 1)),
    )
    R_tilde_ai -= 0.5 * np.tensordot(
        # u^{ar}_{qs}
        u[v, :, :, :],
        # rho^{qs}_{ir}
        rho_qspr[:, :, o, :],
        # axes=((r, q, s), (r, q, s))
        axes=((1, 2, 3), (3, 0, 1)),
    )

    return R_tilde_ai
