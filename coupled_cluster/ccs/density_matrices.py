def compute_one_body_density_matrix(t_1, l_1, o, v, np, out=None):
    if out is None:
        out = np.zeros((v.stop, v.stop), dtype=t_1.dtype)

    out.fill(0)

    add_rho_ba(t_1, l_1, o, v, out, np)
    add_rho_ia(l_1, o, v, out, np)
    add_rho_ai(t_1, l_1, o, v, out, np)
    add_rho_ji(t_1, l_1, o, v, out, np)

    return out


def compute_two_body_density_matrix(t, l, o, v, np, out=None):
    raise NotImplementedError(
        "The two-body density matrix for CCS has not yet been implemented."
    )


def add_rho_ba(t_1, l_1, o, v, out, np):
    """Function for adding v-v part of the CCS one-body density matrix

    rho^{b}_{a} = l^{i}_{a} t^{b}_{i}
    """

    out[v, v] += np.dot(t_1, l_1)


def add_rho_ia(l_1, o, v, out, np):
    """Function for adding the o-v part of the CCS one-body density matrix

    rho^{i}_{a} = l^{i}_{a}
    """

    out[o, v] += l_1


def add_rho_ai(t_1, l_1, o, v, out, np):
    """Function for adding the v-o part of the CCS one-body density matrix

    rho^{a}_{i} = t^{a}_{i}
    """

    out[v, o] += t_1


def add_rho_ji(t_1, l_1, o, v, out, np):
    """Function for adding the o-o part of the CCS one-body density matrix

    rho^{j}_{i} = delta^{i}_{j} - l^{i}_{a} t^{a}_{j}
        + (0.5) l^{ij}_{ab} t^{ab}_{jk}

    """

    delta = np.eye(o.stop)

    out[o, o] += delta - np.dot(l_1, t_1)  # ij
