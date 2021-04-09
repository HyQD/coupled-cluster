def compute_l_1_amplitudes(f, f_t, u_t, t_1, t_2, l_1, l_2, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(l_1)

    add_s1_l(f_t, o, v, out, np=np)
    add_s2a_l(f_t, l_1, o, v, out, np=np)
    add_s2b_l(f_t, l_1, o, v, out, np=np)
    add_s3a_l(u_t, l_1, o, v, out, np=np)
    add_s4a_l(u_t, l_2, o, v, out, np=np)
    add_s4b_l(u_t, l_2, o, v, out, np=np)
    add_s7_l(u_t, l_1, t_2, o, v, out, np=np)
    add_s10c_l(u_t, l_1, t_2, o, v, out, np=np)
    add_s10d_l(u_t, l_1, t_2, o, v, out, np=np)

    return out


def compute_l_2_amplitudes(f, f_t, u_t, t_1, t_2, l_1, l_2, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(l_2)

    add_d1_l(u_t, o, v, out, np=np)
    add_d2c_l(f, l_2, o, v, out, np=np)
    add_d2d_l(f, l_2, o, v, out, np=np)
    add_d5a_l(u_t, l_1, o, v, out, np=np)
    add_d5b_l(u_t, l_1, o, v, out, np=np)
    add_d7a_l(f_t, l_1, o, v, out, np=np)

    return out


# Here begins the L_1 stuff
# Note to self: everything output is upside-down and mirrored.


def add_s1_l(f, o, v, out, np):
    """Function for adding the S1 diagram

        g*(f, u, l, t) <- f^{i}_{a}

    Number of FLOPS required: O(m n)
    """

    out += f[o, v]  # ia


def add_s2a_l(f, l_1, o, v, out, np):
    """Function for adding the S2a diagram

        g*(f, u, l, t) <- f^{b}_{a} l^{i}_{b}

    Number of FLOPS: O(m^2 n)
    """

    out += np.dot(l_1, f[v, v])


def add_s2b_l(f, l_1, o, v, out, np):
    """Function for adding the S2b diagram        g*(f, u, l, t) <- (-1) f^{i}_{j} l^{j}_{a}

    Number of FLOPS: O(m n^2)
    """

    out -= np.dot(f[o, o], l_1)


def add_s3a_l(u, l_1, o, v, out, np):
    """Function for adding the S3a diagram

        g*(f, u, l, t) <- l^{j}_{b} u^{ib}_{aj}

    Number of FLOPS required: O(m^2 n^2)
    """

    out += np.tensordot(l_1, u[o, v, v, o], axes=((0, 1), (3, 1)))  # ia


def add_s3b_l(u, t_1, o, v, out, np):
    """Function for adding the S3b diagram

        g*(f, u, l, t) <- (-1) t^{b}_{j} u^{ij}_{ab}

    Number of FLOPS required: O(m^2 n^2)
    """

    out += np.tensordot(t_1, u[o, o, v, v], axes=((0, 1), (3, 1)))  # ia


def add_s4a_l(u, l_2, o, v, out, np):
    """Function for adding the S4a diagram

        g*(f, u, l, t) <- (0.5) l^{ij}_{bc} u^{bc}_{aj}

    Number of FLOPS required: O(m^3 n^2)
    """

    out += (0.5) * np.tensordot(
        l_2, u[v, v, v, o], axes=((1, 2, 3), (3, 0, 1))
    )  # ia


def add_s4b_l(u, l_2, o, v, out, np):
    """Function for adding the S4b diagram

        g*(f, u, l, t) <- (-0.5) l^{jk}_{ab} u^{ib}_{jk}

    Number of FLOPS required: O(m^2 n^3)
    """

    out -= 0.5 * np.tensordot(u[o, v, o, o], l_2, axes=((2, 3, 1), (0, 1, 3)))


def add_s7_l(u, l_1, t_2, o, v, out, np):
    """Function for adding the S7 diagram (naming probably wrong)

        g*(f, u, l, t) <- l^{j}_{b} t^{bc}_{jk} u^{ik}_{ac}

    We do this in two steps

        W^{c}_{k} = l^{j}_{b} t^{bc}_{jk}
        g*(f, u, l, t) <- W^{c}_{k} u^{ik}_{ac}

    Number of FLOPS required: O(m^2 n^2)
    """

    term = np.tensordot(l_1, t_2, axes=((0, 1), (2, 0)))  # ck
    out += np.tensordot(term, u[o, o, v, v], axes=((0, 1), (3, 1)))  # ia


def add_s10c_l(u, l_1, t_2, o, v, out, np):
    """Function for adding the S10c diagram

        g*(f, u, l, t) <- (-0.5) l^{i}_{b} t^{bc}_{jk} u^{jk}_{ac}

    We do this in two steps

        W^{b}_{a} = -0.5 t^{bc}_{jk} u^{jk}_{ac}
        g*(f, u, l, t) <- l^{i}_{b} W^{b}_{a}

    Number of FLOPS required: O(m^3 n^2)
    """

    W_ba = -0.5 * np.tensordot(t_2, u[o, o, v, v], axes=((1, 2, 3), (3, 0, 1)))
    out += np.dot(l_1, W_ba)


def add_s10d_l(u, l_1, t_2, o, v, out, np):
    """Function for adding the S10d diagram

        g*(f, u, l, t) <- (-0.5) l^{j}_{a} t^{bc}_{jk} u^{ik}_{bc}

    We do this in two steps

        W^{i}_{j} = -0.5 u^{ik}_{bc} t^{bc}_{jk}
        g*(f, u, l, t) <- W^{i}_{j} l^{j}_{a}

    Number of FLOPS required: O(m^2 n^3)
    """

    W_ij = -0.5 * np.tensordot(u[o, o, v, v], t_2, axes=((1, 2, 3), (3, 0, 1)))
    out += np.dot(W_ij, l_1)


# You are now entering the land of L_2. Be cautious.
# I also don't understand the naming convention at all..


def add_d1_l(u, o, v, out, np):
    """Function adding the D1 diagram

        g(f, u, t, l) <- u^{ij}_{ab}

    Number of FLOPS required: O(m^2 n^2).
    """
    out += u[o, o, v, v]


def add_d2c_l(f, l, o, v, out, np):
    """Function adding the D2c diagram

        g(f, u, t, l) <- -f^{c}_{a} l^{ij}_{bc} P(ab)

    Number of FLOPS required: O(m^3 n^2).
    """
    temp = np.tensordot(l, f[v, v], axes=((3), (0))).transpose(0, 1, 3, 2)
    temp -= temp.swapaxes(2, 3)
    out -= temp


def add_d2d_l(f, l, o, v, out, np):
    """Function adding the D2d diagram

        g(f, u, t, l) <- f^{i}_{k} l^{jk}_{ab} P(ij)

    Number of FLOPS required: O(m^2 n^3).
    """
    temp = np.tensordot(f[o, o], l, axes=((1), (1)))
    temp -= temp.swapaxes(0, 1)
    out += temp


def add_d5a_l(u, l_1, o, v, out, np):
    """Function for adding the D5a diagram

        g*(f, u, l, t) <- l^{k}_{a} u^{ij}_{bk} P(ab)

    Number of FLOPS required: O(m^2 n^3)
    """

    term = np.tensordot(u[o, o, v, o], l_1, axes=((3), (0)))
    term -= term.swapaxes(2, 3)
    out -= term


def add_d5b_l(u, l_1, o, v, out, np):
    """Function for adding the D5b diagram

        g*(f, u, l, t) <- (-1) * l^{i}_{c} u^{jc}_{ab} P(ij)

    Number of FLOPS required: O(m^3 n^2)
    """

    term = (-1) * np.tensordot(l_1, u[o, v, v, v], axes=((1), (1)))  # ijab
    term -= term.swapaxes(0, 1)
    out += term


def add_d7a_l(f, l_1, o, v, out, np):
    """Function for adding the D7a diagram

        g*(f, u, l, t) <- f^{i}_{a} l^{j}_{b} P(ab) P(ij)

    Number of FLOPS required: O(m^2 n^2)
    """

    term = np.tensordot(f[o, v], l_1, axes=0).transpose(0, 2, 1, 3)  # ijab
    term -= term.swapaxes(2, 3)  # P(ab)
    term -= term.swapaxes(0, 1)  # P(ij)
    out += term
