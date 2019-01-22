def compute_ground_state_energy_correction(u, t, o, v, np=None):
    if np is None:
        import numpy as np

    return 0.25 * np.tensordot(
        u[v, v, o, o], t, axes=((0, 1, 2, 3), (0, 1, 2, 3))
    )
