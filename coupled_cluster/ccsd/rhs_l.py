def compute_l_1_amplitudes(f, u, t_1, t_2, l_1, l_2, o, v, out=None, np=None):
    if np is None:
        import numpy as np

    if out is None:
        out = np.zeros_like(l_1)

    return out


def compute_l_2_amplitudes(f, u, t_1, t_2, l_1, l_2, o, v, out=None, np=None):
    if np is None:
        import numpy as np

    if out is None:
        out = np.zeros_like(l_2)

    return out
