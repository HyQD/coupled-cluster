def compute_l_1_amplitudes(f, u, t_1, t_2, l_1, l_2, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(l_1)

    return out


def compute_l_2_amplitudes(f, u, t_1, t_2, l_1, l_2, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(l_2)

    return out


# Here begins the L_1 stuff
# Note to self: everything output is upside-down and mirrored.

# TODO: figure out naming conventions and,
# if things need to be Hermitian conjugated? Probably not.


def add_s2a_l(f, l_1, o, v, out, np):
    """Function for adding the S2a diagram

        g*(f, u, l, t) <- f^{b}_{a} l^{i}_{b}

    Number of FLOPS: O(m^2 n)
    """

    out += np.tensordot(f[v, v], l_1, axes=((0), (1))).transpose(1, 0)  # ia


def add_s2b_l(f, l_1, o, v, out, np):
    """Function for adding the S2b diagram
    
        g*(f, u, l, t) <- (-1) f^{i}_{j} l^{j}_{a}

    Number of FLOPS: O(m n^2)
    """

    out += (-1) * np.tensordot(f[o, o], l_1, axes=((1), (0)))  # ia
