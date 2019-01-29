def compute_eta(h, u, rho_qp, rho_qspr, o, v, np):
    eta = np.zeros_like(h)

    A_ibaj = compute_A_ibaj(rho_qp, o, v, np=np)
    R_ia = compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np=np)
    R_tilde_ai = compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np=np)

    A_iajb = A_ibaj.transpose(0, 2, 3, 1)
    eta_jb = np.linalg.tensorsolve(-1j * A_iajb, R_ia)

    A_aibj = A_ibaj.transpose(2, 0, 1, 3)
    eta_bj = np.linalg.tensorsolve(1j * A_aibj, R_tilde_ai)

    eta[o, v] += eta_jb
    eta[v, o] += eta_jb

    return eta


def compute_A_ibaj(rho_qp, o, v, np=np):
    delta_ij = np.eye(o.stop)
    delta_ba = np.eye(v.stop - o.stop)

    A_ibaj = np.einsum("ba, ij -> ibaj", delta_ba, rho_qp[o, o])
    A_ibaj -= np.einsum("ij, ba -> ibaj", delta_ij, rho_qp[v, v])

    return A_ibaj


def compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np):
    R_ia = np.dot(rho_qp[o, o], h[o, v])
    R_ia -= np.dot(h[o, v], rho_qp[v, v])
    R_ia += 0.5 * np.tensordot(
        rho_qspr[o, :, :, :], u[:, :, v, :], axes=((1, 2, 3), (3, 0, 1))
    )
    R_ia -= 0.5 * np.tensordot(
        u[o, :, :, :], rho_qspr[:, :, v, :], axes=((1, 2, 3), (3, 0, 1))
    )

    return R_ia


def compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np):
    R_tilde_ai = np.dot(rho_qp[v, v], h[v, o])
    R_tilde_ai -= np.dot(h[v, o], rho_qp[o, o])
    R_tilde_ai += 0.5 * np.tensordot(
        rho_qspr[v, :, :, :], u[:, :, o, :], axes=((1, 2, 3), (3, 0, 1))
    )
    R_tilde_ai -= 0.5 * np.tensordot(
        u[v, :, :, :], rho_qspr[:, :, o, :], axes=((1, 2, 3), (3, 0, 1))
    )

    return R_tilde_ai


# """
#
# The indexing is extremely confusing. Following the definitions in Kvaal (2012), the
# rationale is that:
#
# rho_p^q = <\tilde{\Psi}|c_p^\dagger c_q|\Psi> -> rho_pq
# h_q^p = <\tilde{\phi}_p|\hat{h}|\phi_q> -> h_pq
# rho_{pq}^{qs} = <\tilde{\Psi}|c_p^\dagger c_r^\dagger c_s c_q|\Psi> -> rho_prqs
# u_{qs}^{pr} = <\tilde{\phi}_p \tilde{\phi}_r |\hat{u}| \phi_q \phi_s> -> u_prqs
# \eta_q^p = <\tilde{\phi}_p|\dot{\phi}_q> -> \eta_pq
#
# """
#
#
# def compute_eta(occ, virt, h, u, Dpq, Dpqrs):
#
#    eta = np.zeros(h.shape, dtype=np.complex128)
#
#    A_ajib = compute_A_ajib(Dpq[occ, occ], Dpq[virt, virt])
#    R_ai = compute_R_ai(occ, virt, h, u, Dpq, Dpqrs)
#    R_ia = compute_R_ia(occ, virt, h, u, Dpq, Dpqrs)
#    eta_jb = np.linalg.solve(-1j * A_ajib, R_ai)
#
#    """
#    Compute eta_bj
#    """
#
#    eta[occ, virt] = eta_jb.reshape(
#        Dpq[occ, occ].shape[0], Dpq[virt, virt].shape[0]
#    )
#
#    return eta
#
#
# def compute_A_ajib(rho_ij, rho_ab):
#    """
#    Equation 31 in Kvaal(2012)
#    """
#    delta_ij = np.eye(rho_ij.shape[0])
#    delta_ab = np.eye(rho_ab.shape[0])
#    A_ajib = np.einsum("ab,ji->aijb", delta_ab, rho_ij)
#    A_ajib -= np.einsum("ji,ab->aijb", delta_ij, rho_ab)
#
#    # A_ajib is reshaped since we want to solve a linear equation for eta
#    A_ajib = A_ajib.reshape(
#        rho_ab.shape[0] * rho_ij.shape[0], rho_ab.shape[0] * rho_ij.shape[0]
#    )
#    return A_ajib
#
#
# def compute_R_ai(occ, virt, h, u, Dpq, Dpqrs):
#    """
#    Right hand side of equation 34c in Kvaal(2012)
#    """
#    R_ai = np.einsum("ji,ja->ai", Dpq[occ, occ], h[occ, virt], optimize=True)
#    R_ai -= np.einsum("ab,ib->ai", Dpq[virt, virt], h[occ, virt], optimize=True)
#    R_ai += 0.5 * np.einsum(
#        "pris,pras->ai", Dpqrs[:, :, occ, :], u[:, :, virt, :], optimize=True
#    )
#    R_ai -= 0.5 * np.einsum(
#        "arqs,irqs->ai", Dpqrs[virt, :, :, :], u[occ, :, :, :], optimize=True
#    )
#    R_ai = R_ai.reshape(Dpq[occ, occ].shape[0] * Dpq[virt, virt].shape[0])
#    return R_ai
#
#
# def compute_R_ia(occ, virt, h, u, Dpq, Dpqrs):
#    """
#    Right hand side of equation 34d in Kvaal(2012)
#    """
#    R_ia = np.einsum("ba,bi->ia", Dpq[virt, virt], h[virt, occ], optimize=True)
#    R_ia -= np.einsum("ij,aj->ia", Dpq[occ, occ], h[virt, occ], optimize=True)
#    R_ia += 0.5 * np.einsum(
#        "pras,pris->ia", Dpqrs[:, :, virt, :], u[:, :, occ, :], optimize=True
#    )
#    R_ia -= 0.5 * np.einsum(
#        "irqs,arqs->ia", Dpqrs[occ, :, :, :], u[virt, :, :, :], optimize=True
#    )
#    R_ia = R_ia.reshape(Dpq[occ, occ].shape[0] * Dpq[virt, virt].shape[0])
#    return R_ia