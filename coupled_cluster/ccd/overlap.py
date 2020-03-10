def compute_overlap(t_1, l_1, t_2, l_2, np):
    """Compute the overlap between two states or wavefunctions using the
    bivariational view. If the two states are the same state at different
    times, then we compute

        P(t_1, t_2) = |<~Psi(t_1)|Psi(t_2)>|^2
                  = <~Psi(t_2)|Psi(t_1)><~Psi(t_1)|Psi(t_2)>,

    where
        <~Psi(t)| = <~Phi|(1 + Lambda(t)) e^{-T(t)}
        |Psi(t)> = e^{T(t)}|Phi>.

    We label the two terms:
        tilde_1 = <~Psi(t_1)|Psi(t_2)>,
        tilde_2 = <~Psi(t_2)|Psi(t_1)>.
    """
    tilde_1 = 1
    tilde_1 += 0.25 * np.tensordot(t_1, l_2, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    tilde_1 -= 0.25 * np.tensordot(t_2, l_2, axes=((0, 1, 2, 3), (2, 3, 0, 1)))

    tilde_2 = 1
    tilde_2 -= 0.25 * np.tensordot(t_1, l_1, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    tilde_2 += 0.25 * np.tensordot(t_2, l_1, axes=((0, 1, 2, 3), (2, 3, 0, 1)))

    return tilde_2 * tilde_1


def compute_orbital_adaptive_overlap(t_1, l_1, t_2, l_2, np):
    raise NotImplementedError("Oh, boy...")
