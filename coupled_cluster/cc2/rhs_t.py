# Labelling of the different terms comes from the book "Many-Body Methods in
# Chemistry and Physics" by I. Shavitt and R. J. Bartlett.


# Diagrams for CC2 amplitude equations

import coupled_cluster.ccd.rhs_t as ccd_t


def compute_t_1_amplitudes(
    f, f_transformed, u_transformed, t_1, t_2, o, v, np, out=None
):

    if out is None:
        out = np.zeros_like(t_1)

    add_s1_t(f_transformed, o, v, out, np=np)
    add_s2a_t(f_transformed, t_2, o, v, out, np=np)
    add_s2b_t(u_transformed, t_2, o, v, out, np=np)
    add_s2c_t(u_transformed, t_2, o, v, out, np=np)

    return out


def compute_t_2_amplitudes(
    f, f_transformed, u_transformed, t_1, t_2, o, v, np, out=None
):

    if out is None:
        out = np.zeros_like(t_2)

    add_d1_t(u_transformed, o, v, out, np=np)
    add_d2a_t(f, t_2, o, v, out, np=np)
    add_d2b_t(f, t_2, o, v, out, np=np)

    return out


def add_s1_t(f, o, v, out, np):
    """Function adding the S1 diagram

        g(f, u, t) <- f^{a}_{i}

    Number of FLOPS required: O(m n).
    """
    out += f[v, o]


def add_s2a_t(f, t_2, o, v, out, np):
    """Function adding the S2a diagram

        g(f, u, t) <- f^{k}_{c} t^{ac}_{ik}

    Numer of FLOPS required: O(m^2 n^2).
    """
    out += np.tensordot(f[o, v], t_2, axes=((0, 1), (3, 1)))


def add_s2b_t(u, t_2, o, v, out, np):
    """Function adding the S2b diagram

        g(f, u, t) <- 0.5 u^{ak}_{cd} t^{cd}_{ik}

    Number of FLOPS required: O(m^3 n^2).
    """
    out += 0.5 * np.tensordot(u[v, o, v, v], t_2, axes=((1, 2, 3), (3, 0, 1)))


def add_s2c_t(u, t_2, o, v, out, np):
    """Function adding the S2b diagram

        g(f, u, t) <- -0.5 u^{kl}_{ic} t^{ac}_{kl}

    Number of FLOPS required: O(m^2 n^3)
    """
    term = np.tensordot(u[o, o, o, v], t_2, axes=((0, 1), (2, 3)))
    out -= 0.5 * np.trace(term, axis1=1, axis2=3).swapaxes(0, 1)


def add_d1_t(u, o, v, out, np):
    """Function adding the D1 diagram

        g(f, u, t) <- u^{ab}_{ij}

    Number of FLOPS required: O(m^2 n^2).
    """
    out += u[v, v, o, o]


def add_d2a_t(f, t, o, v, out, np):
    """Function adding the D2a diagram

        g(f, u, t) <- f^{b}_{c} t^{ac}_{ij} P(ab)

    Number of FLOPS required: O(m^3 n^2).
    """
    term = np.tensordot(f[v, v], t, axes=((1), (1))).transpose(1, 0, 2, 3)
    term -= term.swapaxes(0, 1)
    out += term


def add_d2b_t(f, t, o, v, out, np):
    """Function adding the D2b diagram

        g(f, u, t) <- -f^{k}_{j} t^{ab}_{ik} P(ij)

    Number of FLOPS required: O(m^2 n^3).
    """
    term = np.tensordot(t, f[o, o], axes=((3), (0)))
    term -= term.swapaxes(2, 3)
    out -= term
