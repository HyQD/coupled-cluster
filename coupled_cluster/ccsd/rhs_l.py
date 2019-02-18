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


def add_s1_l(f, o, v, out, np):
    """Function for adding the S1 diagram

        g*(f, u, l, t) <- f^{i}_{a}

    Number of FLOPS required: None?
    """

    out += f[o, v]  # ia


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


def add_s3a_l(u, l_1, o, v, out, np):
    """Function for adding the S3a diagram

        g*(f, u, l, t) <- l^{j}_{b} u^{ib}_{aj}

    Number of FLOPS required: O(m^2 n^2)
    """

    out += np.tensordot(l_1, u[o, v, v, o], axes=((0, 1), (3, 1)))  # ia


def add_s3b_l(u, t_1, o, v, out, np):
    """Function for adding the S3b diagram

        g*(f, u, l, t) <- (-1) t^{b}_{j} u^{ij}_{ab}

    Number of FLOPS required: O(m^2, n^2)
    """

    out += np.tensordot(t_1, u[o, o, v, v], axes=((0, 1), (3, 1)))  # ia


def add_s4a_l(u, l_2, o, v, out, np):
    """Function for adding the S4a diagram

        g*(f, u, l, t) <- (0.5) l^{ij}_{bc} u^{bc}_{aj}

    Number of FLOPS required: O()
    """

    out += (0.5) * np.tensordot(
        l_2, u[v, v, v, o], axes=((1, 2, 3), (3, 0, 1))
    )  # ia


def add_s4b_l(u, l_2, o, v, out, np):
    """Function for adding the S4b diagram

        g*(f, u, l, t) <- (-0.5) l^{jk}_{ab} u^{ib}_{jk}

    Number of FLOPS required: O()
    """

    out += (-0.5) * np.tensordot(
        l_2, u[o, v, o, o], axes=((0, 1, 3), (2, 3, 1))
    ).transpose(
        1, 0
    )  # ai -> ia


def add_s5a_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S5a diagram

        g*(f, u, l, t) <- l^{i}_{b} t^{c}_{j} u^{bj}_{ac}

    Number of FLOPS required: O()
    """

    term = np.tensordot(t_1, u[v, o, v, v], axes=((0, 1), (3, 1))) # ba
    out += np.tensordot(l_1, term, axes=((1), (0))) # ia

def add_s5b_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S5b diagram

        g*(f, u, l, t) <- l^{j}_{a} t^{b}_{k} u^{ik}_{bj}

    Number of FLOPS required: O()
    """

    term = np.tensordot(t_1, u[o, o, v, o], axes=((0, 1), (2, 1))) # ij
    out += np.tensordot(l_1, term, axes=((0), (1))).transpose(1, 0) # ai -> ia

def add_s5c_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S5c diagram

        g*(f, u, l, t) <- l^{j}_{b} t^{c}_{j} u^{ib}_{ac}

    Number of FLOPS required: O()
    """

    term = np.tensordot(l_1, t_1, axes=((0), (1))) # bc
    out += np.tensordot(term, u[o, v, v, v], axes=((0, 1), (1, 3))) # ia

def add_s5d_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S5d diagram

        g*(f, u, l, t) <- (-1) l^{j}_{b} t^{b}_{k} u^{ik}_{aj}

    Number of FLOPS required: O()
    """

    term = (-1) * np.tensordot(l_1, t_1, axes=((1), (0))) # jk
    out += np.tensordot(term, u[o, o, v, o], axes=((0, 1), (3, 1))) # ia