from opt_einsum import contract


def compute_one_body_density_matrix(t1, t2, l1, l2, o, v, np, out=None):

    nocc = o.stop
    nvirt = v.stop - nocc

    rho = np.zeros((nocc + nvirt, nocc + nvirt), dtype=t1.dtype)

    rho[o, o] += 2 * np.eye(nocc)
    rho[o, o] -= contract("kjab,baik->ji", l2, t2)
    rho[o, o] -= contract("ja,ai->ji", l1, t1)

    rho[v, o] = 2 * t1
    rho[v, o] += 2 * contract("jb,abij->ai", l1, t2)
    rho[v, o] -= contract("jb,abji->ai", l1, t2)

    rho[v, o] -= contract("jb,aj,bi->ai", l1, t1, t1, optimize=True)
    rho[v, o] -= contract("ak,jkcb,bcij->ai", t1, l2, t2, optimize=True)
    rho[v, o] -= contract("ci,jkcb,abjk->ai", t1, l2, t2, optimize=True)

    rho[o, v] = l1

    rho[v, v] = contract("ia,bi->ba", l1, t1)
    rho[v, v] += contract("ijac,bcij->ba", l2, t2)

    return rho


def add_rho_ba(t_1, t_2, l_1, l_2, o, v, out, np):
    """Function for adding v-v part of the one-body density matrix

    rho^{b}_{a} = l^{i}_{a} t^{b}_{i} - (0.5) l^{ij}_{ab} t^{bc}_{ij}

    """

    out[v, v] += np.dot(t_1, l_1)  # ab -> ba
    out[v, v] += (0.5) * np.tensordot(
        l_2, t_2, axes=((0, 1, 3), (2, 3, 1))
    ).transpose()  # ab???


def add_rho_ia(l_1, o, v, out, np):
    """Function for adding the o-v part of the one-body density matrix

    rho^{i}_{a} = l^{i}_{a}

    """

    out[o, v] += l_1


def add_rho_ai(t_1, t_2, l_1, l_2, o, v, out, np):
    """Function for adding the v-o part of the one-body density matrix

    rho^{a}_{i} = (-1) l^{i}_{a} t^{a}_{j} t^{b}_{i} + l^{i}_{a} t^{ab}_{ij}
        + (0.5) l^{ij}_{ab} t^{a}_{k} t^{bc}_{ij}
        + (0.5) l^{ij}_{ab} t^{c}_{i} t^{ab}_{jk} + t^{a}_{i}

    alternative first line:
    l^{i}_{a} ( t^{ab}_{ij} - t^{b}_{i} t^{a}_{j} )

    """

    out[v, o] += t_1

    term = t_2 - contract("bi, aj->abij", t_1, t_1)
    out[v, o] += np.tensordot(l_1, term, axes=((0, 1), (3, 1)))

    term = (0.5) * np.tensordot(t_1, l_2, axes=((0), (3)))  # ikjc
    out[v, o] += np.tensordot(
        term, t_2, axes=((1, 2, 3), (2, 3, 1))
    ).transpose()  # ia->ai

    term = -(0.5) * np.tensordot(t_1, l_2, axes=((1), (1)))  # akcb
    out[v, o] += np.tensordot(term, t_2, axes=((1, 2, 3), (2, 0, 1)))  # ai


def add_rho_ji(t_1, t_2, l_1, l_2, o, v, out, np):
    """Function for adding the o-o part of the one-body density matrix

    rho^{j}_{i} = delta^{i}_{j} - l^{i}_{a} t^{a}_{j}
        + (0.5) l^{ij}_{ab} t^{ab}_{jk}

    """

    delta = np.eye(o.stop)

    term = delta - np.tensordot(l_1, t_1, axes=((1), (0)))  # ij
    out[o, o] += term + (0.5) * np.tensordot(
        l_2, t_2, axes=((1, 2, 3), (2, 0, 1))
    )  # ik (ij)
