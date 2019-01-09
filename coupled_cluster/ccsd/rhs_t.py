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
