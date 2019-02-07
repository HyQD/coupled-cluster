def compute_eta(h, u, rho_qp, rho_qspr, o, v, np):
    eta = np.zeros(h.shape, dtype=np.complex128)
    A_ibaj = compute_A_ibaj(rho_qp, o, v, np=np)
    R_ia = compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np=np)
    R_tilde_ai = compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np=np)

    """
    It seems like tensorsolve is sensitive to where we place the 
    1j's !!!!!!!!!!!
    """

    A_iajb = A_ibaj.transpose(0, 2, 3, 1)
    A_iajb = A_iajb.reshape(
        o.stop * (v.stop - o.stop), o.stop * (v.stop - o.stop)
    )

    # print("det(A_iajb: %g" % np.linalg.det(A_iajb))
    eta_jb = np.linalg.solve(A_iajb, R_ia.reshape(o.stop * (v.stop - o.stop)))
    eta_jb = -1j * eta_jb.reshape(o.stop, (v.stop - o.stop))
    # eta_jb = np.linalg.tensorsolve(-1j*A_iajb, R_ia)
    # eta_jb = -1j*np.linalg.tensorsolve(A_iajb, R_ia)

    """
    Using A^{ja}_{bi} = -A^{aj}_{ib}
    Then Eq. [34 d] becomes, 
    i A^{aj}_{ib} \eta^b_j = R_tilde^a_i
    -> A_{ai,bj} \eta_{bj} = R_tilde_{ai} 
    """

    A_ajib = compute_A_bija(rho_qp, o, v, np=np)
    A_ajib = A_ajib.transpose(0, 2, 3, 1)
    A_ajib = A_ajib.reshape(
        o.stop * (v.stop - o.stop), o.stop * (v.stop - o.stop)
    )
    # print("det(A_aibj: %g" % np.linalg.det(A_aibj))
    # eta_bj = np.linalg.tensorsolve(1j*A_aibj, R_tilde_ai)
    eta_bj = np.linalg.solve(
        A_ajib, R_tilde_ai.reshape(o.stop * (v.stop - o.stop))
    )
    eta_bj = -1j * eta_bj.reshape(v.stop - o.stop, o.stop)

    eta[o, v] += eta_jb
    eta[v, o] += eta_bj

    return eta


def compute_A_ibaj(rho_qp, o, v, np):
    delta_ij = np.eye(o.stop)
    delta_ba = np.eye(v.stop - o.stop)

    A_ibaj = np.einsum("ba, ij -> ibaj", delta_ba, rho_qp[o, o])
    A_ibaj -= np.einsum("ij, ba -> ibaj", delta_ij, rho_qp[v, v])

    return A_ibaj


def compute_A_bija(rho_qp, o, v, np):
    delta_ij = np.eye(o.stop)
    delta_ba = np.eye(v.stop - o.stop)

    A_ibaj = -np.einsum("ba, ij -> bija", delta_ba, rho_qp[o, o])
    A_ibaj += np.einsum("ij, ba -> bija", delta_ij, rho_qp[v, v])

    return A_ibaj


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
