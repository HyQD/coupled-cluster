from coupled_cluster.cc_helper import compute_reference_energy
from coupled_cluster.ccd.rhs_t import compute_t_2_amplitudes


def compute_ccd_ground_state_energy(f, u, t, o, v, np):
    energy = compute_reference_energy(f, u, o, v, np=np)
    energy += compute_ground_state_energy_correction(u, t, o, v, np=np)

    return energy


def compute_ground_state_energy_correction(u, t, o, v, np):
    return 0.25 * np.tensordot(
        u[v, v, o, o], t, axes=((0, 1, 2, 3), (0, 1, 2, 3))
    )


def compute_time_dependent_energy(f, u, t, l, o, v, np):
    energy = compute_ground_state_energy_correction(u, t, o, v, np)
    rhs_t = compute_t_2_amplitudes(f, u, t, o, v, np=np)
    energy += 0.25 * np.tensordot(l, rhs_t, axes=((0, 1, 2, 3), (2, 3, 0, 1)))

    return energy


def Lagrangian_fun(T2, L2, F, W, o, v, np):
    """
    Eq [A8] in Kvaal - h_i^i + 0.5 u_{ij}^{ij} term
    """
    result = 0.25 * np.einsum(
        "lkdc,dclk->", L2, W[v, v, o, o], optimize=["einsum_path", (0, 1)]
    )
    result += 0.25 * np.einsum(
        "lkdc,lkdc->", T2, W[o, o, v, v], optimize=["einsum_path", (0, 1)]
    )
    result += np.einsum(
        "lkdc,kmec,dmel->",
        L2,
        T2,
        W[v, o, v, o],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "lkdc,kmdc,ml->",
        L2,
        T2,
        F[o, o],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "lkdc,lkec,de->",
        L2,
        T2,
        F[v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += 0.125 * np.einsum(
        "lkdc,mndc,mnlk->",
        L2,
        T2,
        W[o, o, o, o],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += 0.125 * np.einsum(
        "lkdc,lkef,dcef->",
        L2,
        T2,
        W[v, v, v, v],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "lkdc,kndf,lmec,mnef->",
        L2,
        T2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    )
    result += -0.25 * np.einsum(
        "lkdc,lkdf,mnec,mnef->",
        L2,
        T2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    )
    result += -0.125 * np.einsum(
        "lkdc,kndc,lmef,mnef->",
        L2,
        T2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    )
    result += -0.125 * np.einsum(
        "lkdc,lmdc,knef,mnef->",
        L2,
        T2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    )
    result += 0.0625 * np.einsum(
        "lkdc,mndc,lkef,mnef->",
        L2,
        T2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    )
    return result
