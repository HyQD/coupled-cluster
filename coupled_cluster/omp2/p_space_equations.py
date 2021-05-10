def compute_eta(h, u, rho_qp, rho_qspr, o, v, np):

    # Eq. (23) in: https://aip.scitation.org/doi/10.1063/1.5020633
    R_ai = compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np)

    # Solve P-space equations for X^b_j
    delta_ij = np.eye(o.stop)
    delta_ba = np.eye(v.stop - o.stop)

    A_aibj = np.einsum("ab, ji -> aibj", delta_ba, rho_qp[o, o])
    A_aibj -= np.einsum("ji, ab -> aibj", delta_ij, rho_qp[v, v])

    X_bj = -1j * np.linalg.tensorsolve(A_aibj, R_ai)
    X = np.zeros(h.shape, dtype=np.complex128)

    X[v, o] = X_bj
    X[o, v] = -X_bj.T.conj()

    return X


def compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np):

    R_ai = np.einsum("aj,ji->ai", h[v, o], rho_qp[o, o])
    R_ai += 0.5 * np.einsum(
        "arqs,qsir->ai", u[v, :, :, :], rho_qspr[:, :, o, :], optimize=True
    )

    R_ai -= np.einsum("ab, bi->ai", rho_qp[v, v], h[v, o])
    R_ai -= 0.5 * np.einsum(
        "qsir,arqs->ai", u[:, :, o, :], rho_qspr[v, :, :, :], optimize=True
    )

    return R_ai
