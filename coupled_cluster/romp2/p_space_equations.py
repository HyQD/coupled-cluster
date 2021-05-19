from opt_einsum import contract


def compute_eta(h, u, rho_qp, rho_qspr, o, v, np):
    eta = np.zeros(h.shape, dtype=np.complex128)
    A_ibaj = compute_A_ibaj(rho_qp, o, v, np=np)
    R_ia = compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np=np)

    A_iajb = A_ibaj.transpose(0, 2, 3, 1)
    eta_jb = -1j * np.linalg.tensorsolve(A_iajb, R_ia)

    eta[o, v] += eta_jb
    eta[v, o] -= eta_jb.conj().T

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
    R_ia += np.tensordot(
        # rho^{is}_{pr}
        rho_qspr[o, :, :, :],
        # u^{pr}_{as}
        u[:, :, v, :],
        # axes=((s, p, r), (s, p, r))
        axes=((1, 2, 3), (3, 0, 1)),
    )
    R_ia -= np.tensordot(
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
    R_tilde_ai += np.tensordot(
        # rho^{as}_{pr}
        rho_qspr[v, :, :, :],
        # u^{pr}_{is}
        u[:, :, o, :],
        # axes=((s, p, r), (s, p, r))
        axes=((1, 2, 3), (3, 0, 1)),
    )
    R_tilde_ai -= np.tensordot(
        # u^{ar}_{qs}
        u[v, :, :, :],
        # rho^{qs}_{ir}
        rho_qspr[:, :, o, :],
        # axes=((r, q, s), (r, q, s))
        axes=((1, 2, 3), (3, 0, 1)),
    )

    return R_tilde_ai
