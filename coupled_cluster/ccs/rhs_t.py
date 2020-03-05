def compute_t_1_amplitudes(f, u, t_1, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(t_1)

    add_s1_t(f, o, v, out, np=np)
    add_s3a_t(f, t_1, o, v, out, np=np)
    add_s3b_t(f, t_1, o, v, out, np=np)
    add_s3c_t(u, t_1, o, v, out, np=np)
    add_s5a_t(f, t_1, o, v, out, np=np)
    add_s5b_t(u, t_1, o, v, out, np=np)
    add_s5c_t(u, t_1, o, v, out, np=np)
    add_s6_t(u, t_1, o, v, out, np=np)

    return out


def add_s1_t(f, o, v, out, np):
    """Function adding the S1 diagram

        g(f, u, t) <- f^{a}_{i}

    Number of FLOPS required: O(m n).
    """
    out += f[v, o]


def add_s3a_t(f, t_1, o, v, out, np):
    """Function adding the S3a diagram

        g(f, u, t) <- f^{a}_{c} t^{c}_{i}

    Number of FLOPS required: O(m^2 n)
    """
    out += np.tensordot(f[v, v], t_1, axes=((1), (0)))


def add_s3b_t(f, t_1, o, v, out, np):
    """Function adding the S3b diagram

        g(f, u, t) <- -f^{k}_{i} t^{a}_{k}

    Number of FLOPS required: O(m n^2)
    """
    out += -np.tensordot(f[o, o], t_1, axes=((0), (1))).transpose(1, 0)


def add_s3c_t(u, t_1, o, v, out, np):
    """Function adding the S3c diagram

        g(f, u, t) <- u^{ak}_{ic} t^{c}_{k}

    Number of FLOPS required: O(m^2 n^2)
    """
    out += np.tensordot(u[v, o, o, v], t_1, axes=((1, 3), (1, 0)))


def add_s5a_t(f, t_1, o, v, out, np):
    """Function for adding the S5a diagram

        g(f, u, t) <- -f^{k}_{c} t^{c}_{i} t^{a}_{k}

    We do this in two steps

        W^{k}_{i} = -f^{k}_{c} t^{c}_{i}
        g(f, u, t) <- t^{a}_{k} W^{k}_{i}

    Number of FLOPS required: O(m n^2)
    """

    W_ki = -np.dot(f[o, v], t_1)
    out += np.dot(t_1, W_ki)


def add_s5b_t(u, t_1, o, v, out, np):
    """Function for adding the S5b diagram

        g(f, u, t) <- u^{ak}_{cd} t^{c}_{i} t^{d}_{k}

    We do this in two steps

        W^{ak}_{di} = u^{ak}_{cd} t^{c}_{i}
        g(f, u, t) <- W^{ak}_{di} t^{d}_{k}

    Number of FLOPS required: O(m^3 n^2)
    """
    W_akdi = np.tensordot(u[v, o, v, v], t_1, axes=((2), (0)))
    out += np.tensordot(W_akdi, t_1, axes=((1, 2), (1, 0)))


def add_s5c_t(u, t_1, o, v, out, np):
    """Function for adding the S5c diagram

        g(f, u, t) <- - u^{kl}_{ic} t^{a}_{k} t^{c}_{l}

    We do this in two steps

        W^{k}_{i} = - u^{kl}_{ic} t^{c}_{l}
        g(f, u, t) <- t^{a}_{k} W^{k}_{i}

    Number of FLOPS required: O(m n^3)
    """
    W_ki = -np.tensordot(u[o, o, o, v], t_1, axes=((1, 3), (1, 0)))
    out += np.dot(t_1, W_ki)


def add_s6_t(u, t_1, o, v, out, np):
    """Function for adding the S6 diagram

        g(f, u, t) <- (-1) * u ^{kl}_{cd} t^{c}_{i} t^{a}_{k} t^{d}_{l}

    We do this in three steps

        W^{k}_{c} = - u^{kl}_{cd} t^{d}_{l}
        W^{k}_{i} = W^{k}_{c} t^{c}_{i}
        g(f, u, t) <- t^{a}_{k} W^{k}_{i}

    Number of FLOPS required: O(m^2 n^2)
    """

    W_kc = -np.tensordot(u[o, o, v, v], t_1, axes=((1, 3), (1, 0)))
    W_ki = np.dot(W_kc, t_1)
    out += np.dot(t_1, W_ki)
