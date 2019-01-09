# Labelling of the different terms comes from the book "Many-Body Methods in
# Chemistry and Physics" by I. Shavitt and R. J. Bartlett.


def compute_t_1_amplitudes(f, u, t_1, t_2, o, v, out=None, np=None):
    if np is None:
        import numpy as np

    if out is None:
        out = np.zeros_like(t_1)

    return out


def compute_t_2_amplitudes(f, u, t_1, t_2, o, v, out=None, np=None):
    if np is None:
        import numpy as np

    if out is None:
        out = np.zeros_like(t_2)

    return out


def add_s1_t(f, o, v, out, np=None):
    """Function adding the S1 diagram

        g(f, u, t) <- f^{a}_{i}

    Number of FLOPS required: O(m n).
    """
    if np is None:
        import numpy as np

    out += f[v, o]


def add_s2a_t(f, t_2, o, v, out, np=None):
    """Function adding the S2a diagram

        g(f, u, t) <- f^{k}_{c} t^{ac}_{ik}

    Numer of FLOPS required: O(m^2 n^2).
    """

    if np is None:
        import Numpy as np

    out += np.tensordot(f[o, v], t_2, axes=((0, 1), (3, 1)))


def add_s2b_t(u, t_2, o, v, out, np=None):
    """Function adding the S2b diagram

        g(f, u, t) <- 0.5 u^{ak}_{cd} t^{cd}_{ik}

    Number of FLOPS required: O(m^3 n^2).
    """

    if np is None:
        import Numpy as np

    out += 0.5 * np.tensordot(u[v, o, v, v], t_2, axes=((1, 2, 3), (3, 0, 1)))
