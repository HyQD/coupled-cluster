def compute_one_body_density_matrix(t, l, o, v, np, rho=None):
    if rho is None:
        rho = np.zeros((v.stop, v.stop), dtype=t.dtype)

    rho.fill(0)
    rho[o, o] += np.eye(o.stop)
    rho[o, o] -= 0.5 * np.tensordot(l, t, axes=((0, 2, 3), (2, 0, 1)))
    rho[v, v] += 0.5 * np.tensordot(t, l, axes=((1, 2, 3), (3, 0, 1)))

    return rho


def compute_two_body_density_matrix(t, l, o, v, np, rho=None):
    """Two body density matrices from Kvaal (2012).

    The final two body density matrix should satisfy

        np.einsum('pqpq->',Dpqrs) = N(N-1)

    where N is the number of electrons.
    """
    if rho is None:
        rho = np.zeros((v.stop, v.stop, v.stop, v.stop), dtype=t.dtype)

    rho.fill(0)

    add_rho_klij(t, l, o, v, rho, np)
    add_rho_abij(t, l, o, v, rho, np)
    add_rho_jbia(t, l, o, v, rho, np)
    add_rho_ijab(t, l, o, v, rho, np)
    add_rho_cdab(t, l, o, v, rho, np)

    return rho


def add_rho_klij(t, l, o, v, out, np):
    """Function adding the o-o-o-o part of the two-body density matrix

        rho^{kl}_{ij} = P(ij) delta^{k}_{i} delta^{l}_{j}
            - 0.5 * P(ij) P(kl) delta^{k}_{i} l^{lm}_{cd} t^{cd}_{jm}
            + 0.5 * l^{kl}_{cd} t^{cd}_{ij}

    Number of FLOPS required for the most complex term: O(n^4 m^2).
    """
    delta = np.eye(o.stop)

    term = np.einsum("ki, lj -> klij", delta, delta)
    term -= term.swapaxes(2, 3)

    out[o, o, o, o] += term

    term = 0.5 * np.tensordot(l, t, axes=((1, 2, 3), (3, 0, 1)))
    term = np.einsum("ki, lj -> klij", delta, term)
    term -= term.swapaxes(0, 1)
    term -= term.swapaxes(2, 3)

    out[o, o, o, o] -= term

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
    term_jbia = np.einsum("ji, ba -> jbia", delta, W_ba)

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


# def compute_two_body_density_matrix(t2, l2, o, v, np, rho_pqrs=None):
#    """Two body density matrices from Kvaal (2012).
#    t2 is misshaped so it has to passed in as t2.transpose(2,3,0,1).
#    The final two body density matrix should satisfy
#    (which it does in my test implementation):
#        np.einsum('pqpq->',Dpqrs) = N(N-1)
#    where N is the number of electrons.
#    """
#    if rho_pqrs is None:
#        rho_pqrs = np.zeros((v.stop, v.stop, v.stop, v.stop), dtype=t2.dtype)
#
#    rho_pqrs.fill(0)
#    rho_pqrs[o, o, o, o] = rho_ijkl(l2, t2, np)
#    rho_pqrs[v, v, v, v] = rho_abcd(l2, t2, np)
#    rho_pqrs[o, v, o, v] = rho_iajb(l2, t2, np)
#    rho_pqrs[o, v, v, o] = -rho_pqrs[o, v, o, v].transpose(0, 1, 3, 2)
#    rho_pqrs[v, o, o, v] = -rho_pqrs[o, v, o, v].transpose(1, 0, 2, 3)
#    rho_pqrs[v, o, v, o] = rho_pqrs[o, v, o, v].transpose(1, 0, 3, 2)
#    rho_pqrs[o, o, v, v] = rho_ijab(l2, t2, np)
#    rho_pqrs[v, v, o, o] = rho_abij(l2, t2, np)
#    return rho_pqrs
#
#
# def rho_ijkl(l2, t2, np):
#    """
#    Compute rho_{ij}^{kl}
#    """
#    delta_ij = np.eye(l2.shape[0], dtype=np.complex128)
#    rho_ijkl = np.einsum("ik,jl->ijkl", delta_ij, delta_ij, optimize=True)
#    rho_ijkl -= rho_ijkl.swapaxes(0, 1)
#    Pijkl = 0.5 * np.einsum(
#        "ik,lmcd,jmcd->ijkl", delta_ij, l2, t2, optimize=True
#    )
#    rho_ijkl -= Pijkl
#    rho_ijkl += Pijkl.swapaxes(0, 1)
#    rho_ijkl += Pijkl.swapaxes(2, 3)
#    rho_ijkl -= Pijkl.swapaxes(0, 1).swapaxes(2, 3)
#    rho_ijkl += 0.5 * np.einsum("klcd,ijcd->ijkl", l2, t2, optimize=True)
#    return rho_ijkl
#
#
# def rho_abcd(l2, t2, np):
#    """
#    Compute rho_{ab}^{cd}
#    """
#    rho_abcd = 0.5 * np.einsum("ijab,ijcd->abcd", l2, t2, optimize=True)
#    return rho_abcd
#
#
# def rho_iajb(l2, t2, np):
#    """
#    Compute rho_{ia}^{jb}
#    """
#    rho_iajb = 0.5 * np.einsum(
#        "ij,klac,klbc->iajb", np.eye(l2.shape[0]), l2, t2, optimize=True
#    )
#    rho_iajb -= np.einsum("jkac,ikbc->iajb", l2, t2)
#    return rho_iajb
#
#
# def rho_abij(l2, t2, np):
#    """
#    Compute rho_{ab}^{ij}
#    """
#    return l2.transpose(2, 3, 0, 1).copy()
#
#
# def rho_ijab(l2, t2, np):
#    """
#    Compute rho_{ij}^{ab}
#    """
#    rho_ijab = -0.5 * np.einsum(
#        "klcd,ijac,klbd->ijab", l2, t2, t2, optimize=True
#    )
#    rho_ijab += rho_ijab.swapaxes(2, 3)
#    Pij = np.einsum("klcd,ikac,jlbd->ijab", l2, t2, t2, optimize=True)
#    Pij += 0.5 * np.einsum("klcd,ilab,jkcd->ijab", l2, t2, t2, optimize=True)
#    rho_ijab += Pij
#    rho_ijab -= Pij.swapaxes(0, 1)
#    rho_ijab += 0.25 * np.einsum(
#        "klcd,klab,ijcd->ijab", l2, t2, t2, optimize=True
#    )
#    rho_ijab += t2
#    return rho_ijab
