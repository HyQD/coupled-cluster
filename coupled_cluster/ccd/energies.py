from coupled_cluster.cc import compute_reference_energy
from coupled_cluster.ccd.rhs_t import compute_t_2_amplitudes


def compute_ccd_ground_state_energy(f, u, t, o, v, np=None):
    if np is None:
        import numpy as np

    energy = compute_reference_energy(f, u, o, v, np=np)
    energy += compute_ground_state_energy_correction(u, t, o, v, np=np)

    return energy


def compute_ground_state_energy_correction(u, t, o, v, np=None):
    if np is None:
        import numpy as np

    return 0.25 * np.tensordot(
        u[v, v, o, o], t, axes=((0, 1, 2, 3), (0, 1, 2, 3))
    )
