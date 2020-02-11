def compute_time_dependent_overlap(t, l, t_t, l_t, np):
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

    raise NotImplementedError(
        "The CCS time-dependent overlap has not yet been implemented."
    )
