import numpy as np

"""

The indexing is extremely confusing. Following the definitions in Kvaal (2012), the 
rationale is that: 

rho_p^q = <\tilde{\Psi}|c_p^\dagger c_q|\Psi> -> rho_pq
h_q^p = <\tilde{\phi}_p|\hat{h}|\phi_q> -> h_pq
rho_{pq}^{qs} = <\tilde{\Psi}|c_p^\dagger c_r^\dagger c_s c_q|\Psi> -> rho_prqs
u_{qs}^{pr} = <\tilde{\phi}_p \tilde{\phi}_r |\hat{u}| \phi_q \phi_s> -> u_prqs
\eta_q^p = <\tilde{\phi}_p|\dot{\phi}_q> -> \eta_pq

"""


def compute_eta(occ, virt, h, u, Dpq, Dpqrs):

    eta = np.zeros(h.shape, dtype=np.complex128)

    A_ajib = compute_A_ajib(Dpq[occ, occ], Dpq[virt, virt])
    R_ai = compute_R_ai(occ, virt, h, u, Dpq, Dpqrs)
    R_ia = compute_R_ia(occ, virt, h, u, Dpq, Dpqrs)
    eta_jb = np.linalg.solve(-1j * A_ajib, R_ai)

    """
    Compute eta_bj
    """

    eta[occ, virt] = eta_jb.reshape(
        Dpq[occ, occ].shape[0], Dpq[virt, virt].shape[0]
    )

    return eta


def compute_A_ajib(rho_ij, rho_ab):
    """
    Equation 31 in Kvaal(2012)
    """
    delta_ij = np.eye(rho_ij.shape[0])
    delta_ab = np.eye(rho_ab.shape[0])
    A_ajib = np.einsum("ab,ji->aijb", delta_ab, rho_ij)
    A_ajib -= np.einsum("ji,ab->aijb", delta_ij, rho_ab)

    # A_ajib is reshaped since we want to solve a linear equation for eta
    A_ajib = A_ajib.reshape(
        rho_ab.shape[0] * rho_ij.shape[0], rho_ab.shape[0] * rho_ij.shape[0]
    )
    return A_ajib


def compute_R_ai(occ, virt, h, u, Dpq, Dpqrs):
    """
    Right hand side of equation 34c in Kvaal(2012)
    """
    R_ai = np.einsum("ji,ja->ai", Dpq[occ, occ], h[occ, virt], optimize=True)
    R_ai -= np.einsum("ab,ib->ai", Dpq[virt, virt], h[occ, virt], optimize=True)
    R_ai += 0.5 * np.einsum(
        "pris,pras->ai", Dpqrs[:, :, occ, :], u[:, :, virt, :], optimize=True
    )
    R_ai -= 0.5 * np.einsum(
        "arqs,irqs->ai", Dpqrs[virt, :, :, :], u[occ, :, :, :], optimize=True
    )
    R_ai = R_ai.reshape(Dpq[occ, occ].shape[0] * Dpq[virt, virt].shape[0])
    return R_ai


def compute_R_ia(occ, virt, h, u, Dpq, Dpqrs):
    """
    Right hand side of equation 34d in Kvaal(2012)
    """
    R_ia = np.einsum("ba,bi->ia", Dpq[virt, virt], h[virt, occ], optimize=True)
    R_ia -= np.einsum("ij,aj->ia", Dpq[occ, occ], h[virt, occ], optimize=True)
    R_ia += 0.5 * np.einsum(
        "pras,pris->ia", Dpqrs[:, :, virt, :], u[:, :, occ, :], optimize=True
    )
    R_ia -= 0.5 * np.einsum(
        "irqs,arqs->ia", Dpqrs[occ, :, :, :], u[virt, :, :, :], optimize=True
    )
    R_ia = R_ia.reshape(Dpq[occ, occ].shape[0] * Dpq[virt, virt].shape[0])
    return R_ia
