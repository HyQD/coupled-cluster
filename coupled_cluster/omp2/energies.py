from coupled_cluster.cc_helper import compute_reference_energy


def compute_ccd_ground_state_energy(f, u, t, o, v, np):
    energy = compute_reference_energy(f, u, o, v, np=np)
    energy += compute_ccd_ground_state_energy_correction(u, t, o, v, np=np)

    return energy


def compute_ccd_ground_state_energy_correction(u, t, o, v, np):
    r"""Ground state correlation energy for the coupled cluster doubles method

    \Delta E_{CCD} = 0.25 * t^{ab}_{ij} u^{ij}_{ab}.
    """

    return 0.25 * np.tensordot(
        t, u[o, o, v, v], axes=((0, 1, 2, 3), (2, 3, 0, 1))
    )
