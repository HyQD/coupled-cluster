def compute_time_dependent_overlap(l, t, l_t, t_t, np=None):
    """Compute the overlap between the time-evolved state and the ground state
    wavefunction using the bivariational view. That is, we compute

        P(t, t_0) = |<~Psi(t)|Psi(0)>|^2
                  = <~Psi(t)|Psi(0)><~Psi(0)|Psi(t)>,

    where
        <~Psi(t)| = <~Phi|(1 + Lambda(t)) e^{-T(t)}
        |Psi(t)> = e^{T(t)}|Phi>.

    We label the two terms:
        tilde_t = <~Psi(t)|Psi(0)>,
        tilde_0 = <~Psi(0)|Psi(t)>.
    """

    if np is None:
        import numpy as np

    tilde_t = 1
    tilde_t += 0.25 * np.tensordot(t, l_t, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    tilde_t -= 0.25 * np.tensordot(t_t, l_t, axes=((0, 1, 2, 3), (2, 3, 0, 1)))

    tilde_0 = 1
    tilde_0 -= 0.25 * np.tensordot(t, l, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    tilde_0 += 0.25 * np.tensordot(t_t, l, axes=((0, 1, 2, 3), (2, 3, 0, 1)))

    return tilde_t * tilde_0
