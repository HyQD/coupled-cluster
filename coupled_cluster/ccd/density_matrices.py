from opt_einsum import contract


def compute_one_body_density_matrix(t, l, o, v, np, out=None):
    if out is None:
        out = np.zeros((v.stop, v.stop), dtype=t.dtype)

    out.fill(0)
    out[o, o] += np.eye(o.stop)
    out[o, o] -= 0.5 * np.tensordot(l, t, axes=((0, 2, 3), (2, 0, 1)))
    out[v, v] += 0.5 * np.tensordot(t, l, axes=((1, 2, 3), (3, 0, 1)))

    return out


def compute_two_body_density_matrix(t, l, o, v, np, out=None):
    """Two body density matrices from Kvaal (2012).

    The final two body density matrix should satisfy

        contract('pqpq->', rho_qspr) = N(N-1)

    where N is the number of electrons.
    """
    if out is None:
        out = np.zeros((v.stop, v.stop, v.stop, v.stop), dtype=t.dtype)

    out.fill(0)

    add_rho_klij(t, l, o, v, out, np)
    add_rho_abij(t, l, o, v, out, np)
    add_rho_jbia(t, l, o, v, out, np)
    add_rho_ijab(t, l, o, v, out, np)
    add_rho_cdab(t, l, o, v, out, np)

    return out


def add_rho_klij(t, l, o, v, out, np):
    """Function adding the o-o-o-o part of the two-body density matrix

        rho^{kl}_{ij} = P(ij) delta^{k}_{i} delta^{l}_{j}
            - 0.5 * P(ij) P(kl) delta^{k}_{i} l^{lm}_{cd} t^{cd}_{jm}
            + 0.5 * l^{kl}_{cd} t^{cd}_{ij}

    Number of FLOPS required for the most complex term: O(n^4 m^2).
    """
    delta = np.eye(o.stop)

    term = contract("ki, lj -> klij", delta, delta)
    term -= term.swapaxes(2, 3)
    out[o, o, o, o] += term

    term_lj = -0.5 * np.tensordot(l, t, axes=((1, 2, 3), (3, 0, 1)))
    term = contract("ki, lj -> klij", delta, term_lj)
    term -= term.swapaxes(0, 1)
    term -= term.swapaxes(2, 3)
    out[o, o, o, o] += term

    out[o, o, o, o] += 0.5 * np.tensordot(l, t, axes=((2, 3), (0, 1)))


def add_rho_abij(t, l, o, v, out, np):
    """Function adding the v-v-o-o part of the two-body density matrix

        rho^{ab}_{ij} = -0.5 * P(ab) l^{kl}_{cd} t^{ac}_{ij} t^{bd}_{kl}
            + P(ij) l^{kl}_{cd} t^{ac}_{ik} t^{bd}_{jl}
            + 0.5 * P(ij) l^{kl}_{cd} t^{ab}_{il} t^{cd}_{jk}
            + 0.25 l^{kl}_{cd} t^{ab}_{kl} t^{cd}_{ij}
            + t^{ab}_{ij}

    Number of FLOPS required for the most complex term:

        if n > l / 2:
            O(m^4 n^2)
        else:
            O(m^2 n^4)
    """
    # Complexity: O(m^3 n^2)
    W_bc = -0.5 * np.tensordot(t, l, axes=((1, 2, 3), (3, 0, 1)))
    term = np.tensordot(t, W_bc, axes=((1), (1))).transpose(0, 3, 1, 2)
    term -= term.swapaxes(0, 1)
    out[v, v, o, o] += term

    # Complexity: O(m^3 n^3)
    W_aild = np.tensordot(t, l, axes=((1, 3), (2, 0)))
    term = np.tensordot(W_aild, t, axes=((2, 3), (3, 1))).transpose(0, 2, 1, 3)
    term -= term.swapaxes(2, 3)
    out[v, v, o, o] += term

    # Complexity: O(m^2 n^3)
    W_lj = 0.5 * np.tensordot(l, t, axes=((0, 2, 3), (3, 0, 1)))
    term = np.tensordot(t, W_lj, axes=((3), (0)))
    term -= term.swapaxes(2, 3)
    out[v, v, o, o] += term

    if o.stop >= v.stop // 2:
        # Complexity: O(m^4 n^2)
        W_abcd = 0.25 * np.tensordot(t, l, axes=((2, 3), (0, 1)))
        out[v, v, o, o] += np.tensordot(W_abcd, t, axes=((2, 3), (0, 1)))
    else:
        # Complexity: O(m^2 n^4)
        W_klij = 0.25 * np.tensordot(l, t, axes=((2, 3), (0, 1)))
        out[v, v, o, o] += np.tensordot(t, W_klij, axes=((2, 3), (0, 1)))

    out[v, v, o, o] += t


def add_rho_jbia(t, l, o, v, out, np):
    """Function adding the o-v-o-v part of the two-body density matrix

        rho^{jb}_{ia} = -rho^{bj}_{ia} = -rho^{jb}_{ai} = rho^{bj}_{ai}
            = 0.5 * delta^{j}_{i} l^{kl}_{ac} t^{bc}_{kl}
                - l^{jk}_{ac} t^{bc}_{ik}


    Number of FLOPS required for the most complex term: O(m^3 n^3)
    """
    delta = np.eye(o.stop)

    # Complexity: O(n^2 m^3)
    W_ba = 0.5 * np.tensordot(t, l, axes=((1, 2, 3), (3, 0, 1)))
    term_jbia = contract("ji, ba -> jbia", delta, W_ba)

    # Complexity: O(m^3 n^3)
    term_jbia -= np.tensordot(l, t, axes=((1, 3), (3, 1))).transpose(0, 2, 3, 1)

    # rho^{jb}_{ia}
    out[o, v, o, v] += term_jbia
    # rho^{bj}_{ia}
    out[v, o, o, v] -= term_jbia.transpose(1, 0, 2, 3)
    # rho^{jb}_{ai}
    out[o, v, v, o] -= term_jbia.transpose(0, 1, 3, 2)
    # rho^{bj}_{ai}
    out[v, o, v, o] += term_jbia.transpose(1, 0, 3, 2)


def add_rho_ijab(t, l, o, v, out, np):
    """Function adding the o-o-v-v part of the two-body density matrix

        rho^{ij}_{ab} = l^{ij}_{ab}


    Number of FLOPS required for the most complex term: O(m^2 n^2)
    """
    out[o, o, v, v] += l


def add_rho_cdab(t, l, o, v, out, np):
    """Function adding the v-v-v-v part of the two-body density matrix

        rho^{cd}_{ab} = 0.5 * l^{ij}_{ab} t^{cd}_{ij}


    Number of FLOPS required for the most complex term: O(m^4 n^2)
    """
    out[v, v, v, v] += 0.5 * np.tensordot(t, l, axes=((2, 3), (0, 1)))
