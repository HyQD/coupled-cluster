def compute_l_1_amplitudes(f, u, t_1, l_1, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(l_1)

    add_s1_l(f, o, v, out, np=np)
    add_s2a_l(f, l_1, o, v, out, np=np)
    add_s2b_l(f, l_1, o, v, out, np=np)
    add_s3a_l(u, l_1, o, v, out, np=np)
    add_s3b_l(u, t_1, o, v, out, np=np)
    add_s5a_l(u, l_1, t_1, o, v, out, np=np)
    add_s5b_l(u, l_1, t_1, o, v, out, np=np)
    add_s5c_l(u, l_1, t_1, o, v, out, np=np)
    add_s5d_l(u, l_1, t_1, o, v, out, np=np)
    add_s8a_l(f, l_1, t_1, o, v, out, np=np)
    add_s8b_l(f, l_1, t_1, o, v, out, np=np)
    add_s11f_l(u, l_1, t_1, o, v, out, np=np)
    add_s11g_l(u, l_1, t_1, o, v, out, np=np)
    add_s11h_l(u, l_1, t_1, o, v, out, np=np)

    return out


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
    """Function for adding the S2b diagram

        g*(f, u, l, t) <- (-1) f^{i}_{j} l^{j}_{a}

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


def add_s5a_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S5a diagram

        g*(f, u, l, t) <- l^{i}_{b} t^{c}_{j} u^{bj}_{ac}

    We do this in two steps

        W^{b}_{a} = t^{c}_{j} u^{bj}_{ac}
        g*(f, u, l, t) <- l^{i}_{b} W^{b}_{a}

    Number of FLOPS required: O(m^3 n)
    """

    term = np.tensordot(t_1, u[v, o, v, v], axes=((0, 1), (3, 1)))  # ba
    out += np.tensordot(l_1, term, axes=((1), (0)))  # ia


def add_s5b_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S5b diagram

        g*(f, u, l, t) <- l^{j}_{a} t^{b}_{k} u^{ik}_{bj}

    We do in this two steps

        W^{i}_{j} = t^{b}_{k} u^{ik}_{bj}
        g*(f, u, l, t) <- W^{i}_{j} l^{j}_{a}

    Number of FLOPS required: O(m n^3)
    """

    term = np.tensordot(t_1, u[o, o, v, o], axes=((0, 1), (2, 1)))  # ij
    out += np.dot(term, l_1)


def add_s5c_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S5c diagram

        g*(f, u, l, t) <- l^{j}_{b} t^{c}_{j} u^{ib}_{ac}

    We do this in two steps

        W^{c}_{b} = t^{c}_{j} l^{j}_{b}
        g*(f, u, l, t) <- u^{ib}_{ac} W^{c}_{b}

    Number of FLOPS required: O(m^3 n)
    """

    term = np.dot(t_1, l_1)
    out += np.tensordot(u[o, v, v, v], term, axes=((1, 3), (1, 0)))


def add_s5d_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S5d diagram

        g*(f, u, l, t) <- (-1) l^{j}_{b} t^{b}_{k} u^{ik}_{aj}

    We do this in two steps

        W^{j}_{k} = (-1) l^{j}_{b} t^{b}_{k}
        g*(f, u, l, t) <- u^{ik}_{aj} W^{j}_{k}

    Number of FLOPS required: O(m n^3)
    """

    term = -np.dot(l_1, t_1)
    out += np.tensordot(term, u[o, o, v, o], axes=((0, 1), (3, 1)))  # ia


def add_s8a_l(f, l_1, t_1, o, v, out, np):
    """Function for adding the S8a diagram

        g*(f, u, l, t) <- (-1) f^{i}_{b} l^{j}_{a} t^{b}_{j}

    We do this in two steps

        W^{b}_{a} = (-1) t^{b}_{j} l^{j}_{a}
        g*(f, u, l, t) <- f^{i}_{b} W^{b}_{a}

    Number of FLOPS required: O(m^2 n)
    """

    term = -np.dot(t_1, l_1)
    out += np.dot(f[o, v], term)


def add_s8b_l(f, l_1, t_1, o, v, out, np):
    """Function for adding the S8b diagram

        g*(f, u, l, t) <- (-1) f^{j}_{a} l^{i}_{b} t^{b}_{j}

    We do this in two steps

        W^{i}_{j} = (-1) l^{i}_{b} t^{b}_{j}
        g*(f, u, l, t) <- W^{i}_{j} f^{j}_{a}

    Number of FLOPS required: O(m n^2)
    """

    term = -np.dot(l_1, t_1)
    out += np.dot(term, f[o, v])


def add_s11f_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S11f diagram

        g*(f, u, l, t) <- (-1) l^{i}_{b} t^{b}_{j} t^{c}_{k} u^{jk}_{ac}

    We do this in three steps

        W^{i}_{j} = (-1) l^{i}_{b} t^{b}_{j}
        Z^{j}_{a} = u^{jk}_{ac} t^{c}_{k}
        g*(f, u, l, t) <- W^{i}_{j} Z^{j}_{a}

    Number of FLOPS required: O(m^2 n^2)
    """

    W_ij = -np.dot(l_1, t_1)
    Z_ja = np.tensordot(u[o, o, v, v], t_1, axes=((1, 3), (1, 0)))
    out += np.dot(W_ij, Z_ja)


def add_s11g_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S11g diagram

        g*(f, u, l, t) <- (-1) l^{j}_{a} t^{b}_{j} t^{c}_{k} u^{ik}_{bc}

    We do this in three steps

        W^{i}_{b} = (-1) u^{ik}_{bc} t^{c}_{k}
        W^{i}_{j} = W^{i}_{b} t^{b}_{j}
        g*(f, u, l, t) <- W^{i}_{j} l^{j}_{a}

    Number of FLOPS required: O(m^2 n^2)
    """

    W_ib = -np.tensordot(u[o, o, v, v], t_1, axes=((1, 3), (1, 0)))
    W_ij = np.dot(W_ib, t_1)
    out += np.dot(W_ij, l_1)


def add_s11h_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S11h diagram

        g*(f, u, l, t) <- (-1) l^{j}_{b} t^{b}_{k} t^{c}_{j} u^{ik}_{ac}

    We do this in three steps

        W^{j}_{k} = (-1) l^{j}_{b} t^{b}_{k}
        W^{c}_{k} = t^{c}_{j} W^{j}_{k}
        g*(f, u, l, t) <- u^{ik}_{ac} W^{c}_{k}

    Number of FLOPS required: O(m^2 n^2)
    """

    W_jk = -np.dot(l_1, t_1)
    W_ck = np.dot(t_1, W_jk)
    out += np.tensordot(u[o, o, v, v], W_ck, axes=((1, 3), (1, 0)))
