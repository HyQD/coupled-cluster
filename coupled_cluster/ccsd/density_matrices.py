def compute_one_body_density_matrix(t_1, t_2, l_1, l_2, o, v, np, out=None):
    if out is None:
        out = np.zeros((v.stop, v.stop), dtype=t_1.dtype)

    out.fill(0)

    add_rho_ba(t_1, t_2, l_1, l_2, o, v, out, np)
    add_rho_ia(l_1, o, v, out, np)
    add_rho_ai(t_1, t_2, l_1, l_2, o, v, out, np)
    add_rho_ji(t_1, t_2, l_1, l_2, o, v, out, np)


def add_rho_ba(t_1, t_2, l_1, l_2, o, v, out, np):
    """Function for adding v-v part of the one-body density matrix

        rho^{b}_{a} = l^{i}_{a} t^{b}_{i} - (0.5) l^{ij}_{ab} t^{bc}_{ij}

    """

    term = np.tensordot(l_1, t_1, axes=((0), (1)))  # ab
    out += term - (0.5) * np.tensordot(
        l_2, t_2, axes=((0, 1, 3), (2, 3, 0))
    )  # ac


def add_rho_ia(l_1, o, v, out, np):
    """Function for adding the o-v part of the one-body density matrix

        rho^{i}_{a} = l^{i}_{a}

    """

    out += l_1


def add_rho_ai(t_1, t_2, l_1, l_2, o, v, out, np):
    """Function for adding the v-o part of the one-body density matrix

        rho^{a}_{i} = (-1) l^{i}_{a} t^{a}_{j} t^{b}_{i} + l^{i}_{a} t^{ab}_{ij}
            + (0.5) l^{ij}_{ab} t^{a}_{k} t^{bc}_{ij}
            + (0.5) l^{ij}_{ab} t^{c}_{i} t^{ab}_{jk} + t^{a}_{i}
    
    """

    term = (-1) * np.tensordot(l_1, t_1, axes=((1), (0)))  # ij
    term = np.tensordot(term, t_1, axes=((0), (1))).transpose(
        1, 0
    )  # jb -> bj (ai)

    term += np.tensordot(l_1, t_2, axes=((0, 1), (2, 0)))  # bi (ai)

    term2 = (0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijbk
    term2 = np.tensordot(term2, t_2, axes=((0, 1, 2), (2, 3, 1))).transpose(
        1, 0
    )  # kc -> ck (ai)
    term += term2

    term2 = (0.5) * np.tensordot(l_2, t_1, axes=((0), (1)))  # jabc
    term2 = np.tensordot(term2, t_2, axes=((0, 1, 2), (2, 0, 1)))  # ck (ai)
    term += term2

    out += term + t_1


def add_rho_ji(t_1, t_2, l_1, l_2, o, v, out, np):
    """Function for adding the o-o part of the one-body density matrix

        rho^{j}_{i} = delta^{i}_{j} - l^{i}_{a} t^{a}_{j}
            + (0.5) l^{ij}_{ab} t^{ab}_{jk}

    """

    delta = np.eye(o.stop)

    term = delta - np.tensordot(l_1, t_1, axes=((1), (0)))  # ij
    out += term + (0.5) * np.tensordot(
        l_2, t_2, axes=((1, 2, 3), (2, 0, 1))
    )  # ik (ij)
