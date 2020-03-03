import coupled_cluster.ccs.rhs_l as ccs_l
import coupled_cluster.ccd.rhs_l as ccd_l


def compute_l_1_amplitudes(f, u, t_1, t_2, l_1, l_2, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(l_1)

    ccs_l.compute_l_1_amplitudes(f, u, t_1, l_1, o, v, np=np, out=out)
    add_s4a_l(u, l_2, o, v, out, np=np)
    add_s4b_l(u, l_2, o, v, out, np=np)
    add_s6a_l(u, l_2, t_1, o, v, out, np=np)
    add_s6b_l(u, l_2, t_1, o, v, out, np=np)
    add_s6c_l(u, l_2, t_1, o, v, out, np=np)
    add_s6d_l(u, l_2, t_2, o, v, out, np=np)
    add_s7_l(u, l_1, t_2, o, v, out, np=np)
    add_s9a_l(u, l_2, t_2, o, v, out, np=np)
    add_s9b_l(u, l_2, t_1, o, v, out, np=np)
    add_s9c_l(u, l_2, t_2, o, v, out, np=np)
    add_s10a_l(f, l_2, t_2, o, v, out, np=np)
    add_s10b_l(f, l_2, t_2, o, v, out, np=np)
    add_s10c_l(u, l_1, t_2, o, v, out, np=np)
    add_s10d_l(u, l_1, t_2, o, v, out, np=np)
    add_s10e_l(u, l_2, t_2, o, v, out, np=np)
    add_s10f_l(u, l_2, t_2, o, v, out, np=np)
    add_s10g_l(u, l_2, t_2, o, v, out, np=np)
    add_s11a_l(u, l_2, t_1, o, v, out, np=np)
    add_s11b_l(u, l_2, t_1, o, v, out, np=np)
    add_s11c_l(u, l_2, t_1, o, v, out, np=np)
    add_s11d_l(u, l_2, t_1, t_2, o, v, out, np=np)
    add_s11e_l(u, l_2, t_1, t_2, o, v, out, np=np)
    add_s11i_l(u, l_2, t_1, t_2, o, v, out, np=np)
    add_s11j_l(u, l_2, t_1, t_2, o, v, out, np=np)
    add_s11k_l(u, l_2, t_1, o, v, out, np=np)
    add_s11l_l(u, l_2, t_1, t_2, o, v, out, np=np)
    add_s11m_l(u, l_2, t_1, t_2, o, v, out, np=np)
    add_s11n_l(u, l_2, t_1, t_2, o, v, out, np=np)
    add_s11o_l(u, l_2, t_1, t_2, o, v, out, np=np)
    add_s12a_l(u, l_2, t_1, o, v, out, np=np)
    add_s12b_l(u, l_2, t_1, o, v, out, np=np)

    return out


def compute_l_2_amplitudes(f, u, t_1, t_2, l_1, l_2, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(l_2)

    ccd_l.compute_l_2_amplitudes(f, u, t_2, l_2, o, v, np=np, out=out)
    add_d4a_l(u, l_2, t_1, o, v, out, np=np)
    add_d4b_l(u, l_2, t_1, o, v, out, np=np)
    add_d5a_l(u, l_1, o, v, out, np=np)
    add_d5b_l(u, l_1, o, v, out, np=np)
    add_d7a_l(f, l_1, o, v, out, np=np)
    add_d7b_l(f, l_2, t_1, o, v, out, np=np)
    add_d7c_l(f, l_2, t_1, o, v, out, np=np)
    add_d8a_l(u, l_1, t_1, o, v, out, np=np)
    add_d8b_l(u, l_1, t_1, o, v, out, np=np)
    add_d8c_l(u, l_2, t_1, o, v, out, np=np)
    add_d8d_l(u, l_2, t_1, o, v, out, np=np)
    add_d10a_l(u, l_2, t_1, o, v, out, np=np)
    add_d10b_l(u, l_2, t_1, o, v, out, np=np)
    add_d11a_l(u, l_1, t_1, o, v, out, np=np)
    add_d11b_l(u, l_2, t_1, o, v, out, np=np)
    add_d11c_l(u, l_2, t_1, o, v, out, np=np)
    add_d12a_l(u, l_2, t_1, o, v, out, np=np)
    add_d12b_l(u, l_2, t_1, o, v, out, np=np)
    add_d12c_l(u, l_2, t_1, o, v, out, np=np)

    return out


# Here begins the L_1 stuff
# Note to self: everything output is upside-down and mirrored.


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


def add_s6a_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S6a diagram

        g*(f, u, l, t) <- l^{ij}_{bc} t^{b}_{k} u^{ck}_{aj}

    We do this in two steps

        W^{ij}_{ck} = l^{ij}_{bc} t^{b}_{k}
        g*(f, u, l, t) <- W^{ij}_{ck} u^{ck}_{aj}

    Number of FLOPS required: O(m^2 n^3)
    """

    term = np.tensordot(l_2, t_1, axes=((2), (0)))  # ijck
    out += np.tensordot(term, u[v, o, v, o], axes=((1, 2, 3), (3, 0, 1)))  # ia


def add_s6b_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S6b diagram

        g*(f, u, l, t) <- (0.5) l^{ij}_{bc} t^{d}_{j} u^{bc}_{ad}

    We do this in two steps

        W^{ij}_{ad} = (0.5) l^{ij}_{bc} u^{bc}_{ad}
        g*(f, u, l, t) <- W^{ij}_{ad} t^{d}_{j}

    Number of FLOPS required: O(m^4 n^2)
    """

    W_ijad = 0.5 * np.tensordot(l_2, u[v, v, v, v], axes=((2, 3), (0, 1)))
    out += np.tensordot(W_ijad, t_1, axes=((1, 3), (1, 0)))


def add_s6c_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S6c diagram

        g*(f, u, l, t) <- (0.5) l^{jk}_{ab} t^{b}_{l} u^{il}_{jk}

    We do this in two steps

        W^{jk}_{al} = 0.5 l^{jk}_{ab} t^{b}_{l}
        g*(f, u, l, t) <- u^{il}_{jk} W^{jk}_{al}

    Number of FLOPS required: O(m n^4)
    """

    term = (0.5) * np.tensordot(l_2, t_1, axes=((3), (0)))  # jkal
    out += np.tensordot(u[o, o, o, o], term, axes=((1, 2, 3), (3, 0, 1)))


def add_s6d_l(u, l_2, t_2, o, v, out, np):
    """Function for adding the S6d diagram

        g*(f, u, l, t) <- (0.5) l^{jk}_{bc} t^{bd}_{jk} u^{ic}_{ad}

    We do this in two steps

        W^{d}_{c} = 0.5 t^{bd}_{jk} l^{jk}_{bc}
        g*(f, u, l, t) <- u^{ic}_{ad} W^{d}_{c}

    Number of FLOPS required: O()
    """

    W_dc = (0.5) * np.tensordot(t_2, l_2, axes=((0, 2, 3), (2, 0, 1)))
    out += np.tensordot(u[o, v, v, v], W_dc, axes=((1, 3), (1, 0)))


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


def add_s9a_l(u, l_2, t_2, o, v, out, np):
    """Function for adding the S9a diagram

        g*(f, u, l, t) <- (-1) l^{ij}_{bc} t^{bd}_{jk} u^{ck}_{ad}

    We do this in two steps

        W^{ic}_{dk} = (-1) l^{ij}_{bc} t^{bd}_{jk}
        g*(f, u, l, t) <- W^{ic}_{dk} u^{ck}_{ad}

    Number of FLOPS required: O(m^3 n^3)
    """

    term = (-1) * np.tensordot(l_2, t_2, axes=((1, 2), (2, 0)))  # icdk
    out += np.tensordot(term, u[v, o, v, v], axes=((1, 2, 3), (0, 3, 1)))  # ia


def add_s9b_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S9b diagram

        g*(f, u, l, t) <- (-1) l^{jk}_{ab} t^{c}_{j} u^{ib}_{ck}

    We do this in two steps

        W^{ck}_{ab} = (-1) t^{c}_{j} l^{jk}_{ab}
        g*(f, u, l, t) <- u^{ib}_{ck} W^{ck}_{ab}

    Number of FLOPS required: O(m^3 n^2)
    """

    W_ckab = -np.tensordot(t_1, l_2, axes=((1), (0)))
    out += np.tensordot(u[o, v, v, o], W_ckab, axes=((1, 2, 3), (3, 0, 1)))


def add_s9c_l(u, l_2, t_2, o, v, out, np):
    """Function for adding the S9c diagram

        g*(f, u, l, t) <- (-1) l^{jk}_{ab} t^{bc}_{jl} u^{il}_{ck}

    We do this in two steps

        W^{ka}_{cl} = (-1) l^{jk}_{ab} t^{bc}_{jl}
        g*(f, u, l, t) <- u^{il}_{ck} W^{ka}_{cl}

    Number of FLOPS required: O(m^3 n^3)
    """

    term = (-1) * np.tensordot(l_2, t_2, axes=((0, 3), (2, 0)))  # kacl
    out += np.tensordot(u[o, o, v, o], term, axes=((3, 2, 1), (0, 2, 3)))


def add_s10a_l(f, l_2, t_2, o, v, out, np):
    """Function for adding the S10a diagram

        g*(f, u, l, t) <- (-0.5) f^{i}_{b} l^{jk}_{ac} t^{bc}_{jk}

    We do this in two steps

        W^{b}_{a} = (-0.5) t^{bc}_{jk} l^{jk}_{ac}
        g*(f, u, l, t) <- f^{i}_{b} W^{b}_{a}

    Number of FLOPS required: O(m^3 n^2)
    """

    W_ba = -0.5 * np.tensordot(t_2, l_2, axes=((1, 2, 3), (3, 0, 1)))
    out += np.dot(f[o, v], W_ba)


def add_s10b_l(f, l_2, t_2, o, v, out, np):
    """Function for adding the S10b diagram

        g*(f, u, l, t) <- (-0.5) f^{j}_{a} l^{ik}_{bc} t^{bc}_{jk}

    We do this in two steps

        W^{i}_{j} = (-0.5) l^{ik}_{bc} t^{bc}_{jk}
        g*(f, u, l, t) <- W^{i}_{j} f^{j}_{a}

    Number of FLOPS required: O(m^2 n^3)
    """

    term = (-0.5) * np.tensordot(l_2, t_2, axes=((1, 2, 3), (3, 0, 1)))  # ij
    out += np.dot(term, f[o, v])


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


def add_s10e_l(u, l_2, t_2, o, v, out, np):
    """Function for adding the S10e diagram

        g*(f, u, l, t) <- (-0.5) l^{jk}_{bc} t^{bc}_{jl} u^{il}_{ak}

    We do this in two steps

        W^{k}_{l} = -0.5 l^{jk}_{bc} t^{bc}_{jl}
        g*(f, u, l, t) <- W^{k}_{l} u^{il}_{ak}

    Number of FLOPS required: O(m^2 n^3)
    """

    term = (-0.5) * np.tensordot(l_2, t_2, axes=((0, 2, 3), (2, 0, 1)))  # kl
    out += np.tensordot(term, u[o, o, v, o], axes=((0, 1), (3, 1)))  # ia


def add_s10f_l(u, l_2, t_2, o, v, out, np):
    """Function for adding the S10f diagram

        g*(f, u, l, t) <- (-0.25) l^{jk}_{ab} t^{cd}_{jk} u^{ib}_{cd}

    We do this in two steps

        W^{ib}_{jk} = -0.25 u^{ib}_{cd} t^{cd}_{jk}
        g*(f, u, l, t) <- W^{ib}_{jk} l^{jk}_{ab}

    Number of FLOPS required: O(m^3 n^3)
    """

    W_ibjk = -0.25 * np.tensordot(u[o, v, v, v], t_2, axes=((2, 3), (0, 1)))
    out += np.tensordot(W_ibjk, l_2, axes=((1, 2, 3), (3, 0, 1)))


def add_s10g_l(u, l_2, t_2, o, v, out, np):
    """Function for adding the S10g diagram

        g*(f, u, l, t) <- (0.25) l^{ij}_{bc} t^{bc}_{kl} u^{kl}_{aj}

    We do this in two steps

        W^{ij}_{kl} = 0.25 l^{ij}_{bc} t^{bc}_{kl}
        g*(f, u, l, t) <- W^{ij}_{kl} u^{kl}_{aj}

    Number of FLOPS required: O(m^2 n^4)
    """

    term = (0.25) * np.tensordot(l_2, t_2, axes=((2, 3), (0, 1)))  # ijkl
    out += np.tensordot(term, u[o, o, v, o], axes=((1, 2, 3), (3, 0, 1)))  # ia


def add_s11a_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S11a diagram

        g*(f, u, l, t) <- l^{ij}_{bc} t^{b}_{k} t^{d}_{j} u^{ck}_{ad}

    We do this in three steps

        W^{ij}_{ck} = l^{ij}_{bc} t^{b}_{k}
        W^{ic}_{kd} = W^{ij}_{ck} t^{d}_{j}
        g*(f, u, l, t) <- W^{ic}_{kd} u^{ck}_{ad}

    Number of FLOPS required: O(m^3 n^2)
    """

    term = np.tensordot(l_2, t_1, axes=((2), (0)))  # ijck
    term = np.tensordot(term, t_1, axes=((1), (1)))  # ickd
    out += np.tensordot(term, u[v, o, v, v], axes=((1, 2, 3), (0, 1, 3)))  # ia


def add_s11b_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S11b diagram

        g*(f, u, l, t) <- l^{jk}_{ab} t^{b}_{l} t^{c}_{j} u^{il}_{ck}

    We do this in three steps

        W^{jk}_{al} = l^{jk}_{ab} t^{b}_{l}
        W^{ic}_{ja} = u^{il}_{ck} W^{jk}_{al}
        g*(f, u, l, t) <- W^{ic}_{ja} t^{c}_{j}

    Number of FLOPS required: O(m^2 n^4)
    """

    term = np.tensordot(l_2, t_1, axes=((3), (0)))  # jkal
    term = np.tensordot(u[o, o, v, o], term, axes=((3, 1), (1, 3)))
    out += np.tensordot(term, t_1, axes=((1, 2), (0, 1)))


def add_s11c_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S11c diagram

        g*(f, u, l, t) <- (0.5) l^{jk}_{ab} t^{c}_{k} t^{d}_{j} u^{ib}_{cd}

    We do this in three steps

        W^{ib}_{cj} = 0.5 u^{ib}_{cd} t^{d}_{j}
        W^{ib}_{jk} = W^{ib}_{cj} t^{c}_{k}
        g*(f, u, l, t) <- W^{ib}_{jk} l^{jk}_{ab}

    Number of FLOPS required: O(m^3 n^2)
    """

    term = 0.5 * np.tensordot(u[o, v, v, v], t_1, axes=((3), (0)))  # ibcj
    term = np.tensordot(term, t_1, axes=((2), (0)))  # ibjk
    out += np.tensordot(term, l_2, axes=((1, 2, 3), (3, 0, 1)))


def add_s11d_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11d diagram

        g*(f, u, l, t) <- (0.5) l^{jk}_{bc} t^{b}_{l} t^{cd}_{jk} u^{il}_{ad}

    We do this in three steps

        W^{jk}_{cl} = 0.5 l^{jk}_{bc} t^{b}_{l}
        W^{l}_{d} = W^{jk}_{cl} t^{cd}_{jk}
        g*(f, u, l, t) <- W^{l}_{d} u^{il}_{ad}

    Number of FLOPS required: O(m^2 n^3)
    """

    term = (0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # jkcl
    term = np.tensordot(term, t_2, axes=((0, 1, 2), (2, 3, 0)))  # ld
    out += np.tensordot(term, u[o, o, v, v], axes=((0, 1), (1, 3)))  # ia


def add_s11e_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11e diagram

        g*(f, u, l, t) <- (0.5) * l^{jk}_{bc} t^{d}_{j} t^{bc}_{kl} u^{il}_{ad}

    We do this in three steps

        W^{kb}_{cd} = 0.5 l^{jk}_{bc} t^{d}_{j}
        W^{d}_{l} = W^{kb}_{cd} t^{bc}_{kl}
        g*(f, u, l, t) <- W^{d}_{l} u^{il}_{ad}

    Number of FLOPS required: O(m^3 n^2)
    """

    term = (0.5) * np.tensordot(l_2, t_1, axes=((0), (1)))  # kbcd
    term = np.tensordot(term, t_2, axes=((0, 1, 2), (2, 0, 1)))  # dl
    out += np.tensordot(term, u[o, o, v, v], axes=((0, 1), (3, 1)))  # ia


def add_s11i_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11i diagram

        g*(f, u, l, t) <- (-1) l^{ij}_{bc} t^{b}_{k} t^{cd}_{jl} u^{kl}_{ad}

    We do this in three steps

        W^{ij}_{ck} = (-1) l^{ij}_{bc} t^{b}_{k}
        W^{ik}_{dl} = W^{ij}_{ck} t^{cd}_{jl}
        g*(f, u, l, t) <- W^{ik}_{dl} u^{kl}_{ad}

    Number of FLOPS required: O(m^2 n^3)
    """

    term = (-1) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijck
    term = np.tensordot(term, t_2, axes=((1, 2), (2, 0)))  # ikdl
    out += np.tensordot(term, u[o, o, v, v], axes=((1, 2, 3), (0, 3, 1)))  # ia


def add_s11j_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11j diagram

        g*(f, u, l, t) <- (-1) l^{jk}_{ab} t^{c}_{j} t^{bd}_{kl} u^{il}_{cd}

    We do this in three steps

        W^{ka}_{bc} = (-1) l^{jk}_{ab} t^{c}_{j}
        W^{ac}_{dl} = W^{ka}_{bc} t^{bd}_{kl}
        g*(f, u, l, t) <- u^{il}_{cd} W^{ac}_{dl}

    Number of FLOPS required: O(m^4 n^2)
    """

    term = (-1) * np.tensordot(l_2, t_1, axes=((0), (1)))  # kabc
    term = np.tensordot(term, t_2, axes=((0, 2), (2, 0)))  # acdl
    out += np.tensordot(u[o, o, v, v], term, axes=((2, 3, 1), (1, 2, 3)))


def add_s11k_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S11k diagram

        g*(f, u, l, t) <- (-0.5) l^{ij}_{bc} t^{b}_{l} t^{c}_{k} u^{kl}_{aj}

    We do this in three steps

        W^{ij}_{cl} = (-0.5) l^{ij}_{bc} t^{b}_{l}
        w^{ij}_{lk} = w^{ij}_{cl} t^{c}_{k}
        g*(f, u, l, t) <- W^{ij}_{lk} u^{kl}_{aj}

    Number of FLOPS required: O(m^2 n^3)
    """

    term = (-0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijcl
    term = np.tensordot(term, t_1, axes=((2), (0)))  # ijlk
    out += np.tensordot(term, u[o, o, v, o], axes=((1, 2, 3), (3, 1, 0)))  # ia


def add_s11l_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11k diagram

        g*(f, u, l, t) <- (-0.5) l^{ij}_{bc} t^{d}_{k} t^{bc}_{jl} u^{kl}_{ad}

    We do this in three steps

        W^{i}_{l} = (-0.5) l^{ij}_{bc} t^{bc}_{jl}
        Z^{l}_{a} = t^{d}_{k} u^{kl}_{ad}
        g*(f, u, l, t) <- W^{i}_{l} Z^{l}_{a}

    Number of FLOPS required: O(m^2 n^3)
    """

    W_il = -0.5 * np.tensordot(l_2, t_2, axes=((1, 2, 3), (2, 0, 1)))
    Z_la = np.tensordot(t_1, u[o, o, v, v], axes=((0, 1), (3, 0)))
    out += np.dot(W_il, Z_la)


def add_s11m_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11m diagram

        g*(f, u, l, t) <- (-0.5) l^{jk}_{ab} t^{c}_{l} t^{bd}_{jk} u^{il}_{cd}

    We do this in three steps

        W^{d}_{a} = (-0.5) t^{bd}_{jk} l^{jk}_{ab}
        Z^{i}_{d} = t^{c}_{l} u^{il}_{cd}
        g*(f, u, l, t) <- Z^{i}_{d} W^{d}_{a}

    Number of FLOPS required: O(m^3 n^2)
    """

    W_da = -0.5 * np.tensordot(t_2, l_2, axes=((0, 2, 3), (3, 0, 1)))
    Z_id = np.tensordot(t_1, u[o, o, v, v], axes=((0, 1), (2, 1)))
    out += np.dot(Z_id, W_da)


def add_s11n_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11n diagram

        g*(f, u, l t) <- (0.25) l^{ij}_{bc} t^{d}_{j} t^{bc}_{kl} u^{kl}_{ad}

    We do this in three steps

        W^{ij}_{kl} = 0.25 l^{ij}_{bc} t^{bc}_{kl}
        Z^{kl}_{aj} = u^{kl}_{ad} t^{d}_{j}
        g*(f, u, l, t) <- W^{ij}_{kl} Z^{kl}_{aj}

    Number of FLOPS required: O(m^2 n^4)
    """

    W_ijkl = 0.25 * np.tensordot(l_2, t_2, axes=((2, 3), (0, 1)))
    Z_klaj = np.tensordot(u[o, o, v, v], t_1, axes=((3), (0)))
    out += np.tensordot(W_ijkl, Z_klaj, axes=((1, 2, 3), (3, 0, 1)))


def add_s11o_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11o diagram

        g*(f, u, l, t) <- (0.25) l^{jk}_{ab} t^{b}_{l} t^{cd}_{jk} u^{il}_{cd}

    We do this in three steps

        W^{il}_{jk} = 0.25 u^{il}_{cd} t^{cd}_{jk}
        W^{il}_{ab} = W^{il}_{jk} l^{jk}_{ab}
        g*(f, u, l, t) <- W^{il}_{ab} t^{b}_{l}

    Number of FLOPS required: O(m^2 n^4)
    """

    W_iljk = 0.25 * np.tensordot(u[o, o, v, v], t_2, axes=((2, 3), (0, 1)))
    W_ilab = np.tensordot(W_iljk, l_2, axes=((2, 3), (0, 1)))
    out += np.tensordot(W_ilab, t_1, axes=((1, 3), (1, 0)))


def add_s12a_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S12a diagram

        g*(f, u, l, t) <- (-0.5) l^{ij}_{bc} t^{b}_{l} t^{c}_{k} t^{d}_{j} u^{kl}_{ad}

    We do this in four steps

        W^{ij}_{cl} = (-0.5) l^{ij}_{bc} t^{b}_{l}
        W^{ij}_{lk} = W^{ij}_{cl} t^{c}_{k}
        W^{il}_{kd} = W^{ij}_{lk} t^{d}_{j}
        g*(f, u, l, t) <- W^{il}_{kd} u^{kl}_{ad}

    Number of FLOPS required: O(m^2 n^3)
    """

    term = (-0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijcl
    term = np.tensordot(term, t_1, axes=((2), (0)))  # ijlk
    term = np.tensordot(term, t_1, axes=((1), (1)))  # ilkd
    out += np.tensordot(term, u[o, o, v, v], axes=((1, 2, 3), (1, 0, 3)))  # ia


def add_s12b_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S12b diagram

        g*(f, u, l, t) <- (-0.5) * l^{jk}_{ab} t^{b}_{l} t^{c}_{k} t^{d}_{j} u^{il}_{cd}

    We do this in four steps

        W^{jk}_{al} = (-0.5) l^{jk}_{ab} t^{b}_{l}
        W^{ja}_{lc} = W^{jk}_{al} t^{c}_{k}
        W^{al}_{cd} = W^{ja}_{lc} t^{d}_{j}
        g*(f, u, l, t) <- u^{il}_{cd} W^{al}_{cd}

    Number of FLOPS required: O(m^3 n^2)
    """

    term = (-0.5) * np.tensordot(l_2, t_1, axes=((3), (0)))  # jkal
    term = np.tensordot(term, t_1, axes=((1), (1)))  # jalc
    term = np.tensordot(term, t_1, axes=((0), (1)))  # alcd
    out += np.tensordot(u[o, o, v, v], term, axes=((1, 2, 3), (1, 2, 3)))


# You are now entering the land of L_2. Be cautious.
# I also don't understand the naming convention at all..


def add_d4a_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the D4a diagram

        g*(f, u, l, t) <- l^{ij}_{cd} t^{c}_{k} u^{dk}_{ab}

    We do this in two steps

        W^{ij}_{dk} = l^{ij}_{cd} t^{c}_{k}
        g*(f, u, l, t) <- W^{ij}_{dk} u^{dk}_{ab}

    Number of FLOPS required: O(m^3 n^3)
    """

    term = np.tensordot(l_2, t_1, axes=((2), (0)))  # ijdk
    out += np.tensordot(term, u[v, o, v, v], axes=((2, 3), (0, 1)))  # ijab


def add_d4b_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the D4b diagram

        g*(f, u, l, t) <- l^{kl}_{ab} t^{c}_{k} u^{ij}_{cl}

    We do this in two steps

        W^{ij}_{kl} = t^{c}_{k} u^{ij}_{cl}
        g*(f, u, l, t) <- W^{ij}_{kl} l^{kl}_{ab}

    Number of FLOPS required: O(m^2 n^4)
    """

    W_ijkl = np.tensordot(u[o, o, v, o], t_1, axes=((2), (0))).transpose(
        0, 1, 3, 2
    )
    out += np.tensordot(W_ijkl, l_2, axes=((2, 3), (0, 1)))


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


def add_d7b_l(f, l_2, t_1, o, v, out, np):
    """Function for adding the D7b diagram

        g*(f, u, l, t) <- f^{i}_{c} l^{jk}_{ab} t^{c}_{k} P(ij)

    We do this in two steps

        W^{i}_{k} = f^{i}_{c} t^{c}_{k}
        g*(f, u, l, t) <- W^{i}_{k} l^{jk}_{ab} P(ij)

    Number of FLOPS required: O(M^2 N^3)
    """

    # Must first do terms 1 and 3
    term = np.tensordot(f[o, v], t_1, axes=((1), (0)))  # ik
    term = np.tensordot(term, l_2, axes=((1), (1)))  # ijab
    term -= term.swapaxes(0, 1)
    out += term


def add_d7c_l(f, l_2, t_1, o, v, out, np):
    """Function for adding the D7c diagram

        g*(f, u, l, t) <- f^{k}_{a} l^{ij}_{bc} t^{c}_{k} P(ab)

    We do this in two steps

        W^{c}_{a} = t^{c}_{k} f^{k}_{a}
        g*(f, u, l, t) <- l^{ij}_{bc} W^{c}_{a} P(ab)

    Number of FLOPS required: O(m^3 n^2)
    """

    W_ca = np.tensordot(t_1, f[o, v], axes=((1), (0)))
    term = np.tensordot(l_2, W_ca, axes=((3), (0)))
    term -= term.swapaxes(2, 3)
    out -= term


def add_d8a_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the D8a diagram

        g*(f, u, l, t) <- l^{i}_{c} t^{c}_{k} u^{jk}_{ab} P(ij)

    We do this in two steps

        W^{i}_{k} = l^{i}_{c} t^{c}_{k}
        g*(f, u, l, t) <- W^{i}_{k} u^{jk}_{ab} P(ij)

    Number of FLOPS required: O(m^2 n^3)
    """

    term = np.tensordot(l_1, t_1, axes=((1), (0)))  # ik
    term = np.tensordot(term, u[o, o, v, v], axes=((1), (1)))  # ijab
    term -= term.swapaxes(0, 1)
    out += term


def add_d8b_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the D8b diagram

        g*(f, u, l, t) <- l^{k}_{a} t^{c}_{k} u^{ij}_{bc} P(ab)

    We do this in two steps

        W^{c}_{a} = t^{c}_{k} l^{k}_{a}
        g*(f, u, l, t) <- u^{ij}_{bc} W^{c}_{a} P(ab)

    Number of FLOPS required: O(m^3 N^2)
    """

    W_ca = np.dot(t_1, l_1)
    term = np.tensordot(u[o, o, v, v], W_ca, axes=((3), (0)))
    term -= term.swapaxes(2, 3)
    out -= term


def add_d8c_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the D8c diagram

        g*(f, u, l, t) <- l^{ij}_{ac} t^{d}_{k} u^{ck}_{bd} P(ab)

    We do this in two steps

        W^{c}_{b} = t^{d}_{k} u^{ck}_{bd}
        g*(f, u, l, t) <- l^{ij}_{ac} W^{c}_{b} P(ab)

    Number of FLOPS required: O(m^3 n^2)
    """

    # Starting with terms 2 and 3
    term = np.tensordot(t_1, u[v, o, v, v], axes=((0, 1), (3, 1)))  # cb
    term = np.tensordot(l_2, term, axes=((3), (0)))  # ijab
    term -= term.swapaxes(2, 3)
    out += term


def add_d8d_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the D8d diagram

        g*(f, u, l, t) <- l^{ik}_{ab} t^{c}_{l} u^{jl}_{ck} P(ij)

    We do this in two steps

        W^{j}_{k} = t^{c}_{l} u^{jl}_{ck}
        g*(f, u, l, t) <- W^{j}_{k} l^{ik}_{ab} P(ij)

    Number of FLOPS required: O(m^2 n^3)
    """

    # Starting with terms 2 and 3
    W_jk = np.tensordot(t_1, u[o, o, v, o], axes=((0, 1), (2, 1)))  # jk
    term = np.tensordot(W_jk, l_2, axes=((1), (1)))
    term -= term.swapaxes(0, 1)
    out -= term


def add_d10a_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the D10a diagram

        g*(f, u, l, t) <- (-0.5) l^{kl}_{ab} t^{c}_{l} t^{d}_{k} u^{ij}_{cd}

    We do this in three steps

        W^{dl}_{ab} = (-0.5) t^{d}_{k} l^{kl}_{ab}
        W^{cd}_{ab} = t^{c}_{l} W^{dl}_{ab}
        g*(f, u, l, t) <- u^{ij}_{cd} W^{cd}_{ab}

    Number of FLOPS required: O(m^4 n^2)
    """

    W_dlab = -0.5 * np.tensordot(t_1, l_2, axes=((1), (0)))
    W_cdab = np.tensordot(t_1, W_dlab, axes=((1), (1)))
    out += np.tensordot(u[o, o, v, v], W_cdab, axes=((2, 3), (0, 1)))


def add_d10b_l(u, l_2, t_1, o, v, out, np):
    """function for adding the D10b diagram

        g*(f, u, l, t) <- (-0.5) l^{ij}_{cd} t^{c}_{l} t^{d}_{k} u^{kl}_{ab}

    We do this in three steps

        W^{ij}_{dl} = (-0.5) l^{ij}_{cd} t^{c}_{l}
        W^{ij}_{lk} = W^{ij}_{dl} t^{d}_{k}
        g*(f, u, l, t) <- W^{ij}_{lk} u^{kl}_{ab}

    Number of FLOPS required: O(m^2 n^4)
    """

    term = (-0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijdl
    term = np.tensordot(term, t_1, axes=((2), (0)))  # ijlk
    out += np.tensordot(term, u[o, o, v, v], axes=((2, 3), (1, 0)))  # ijab


def add_d11a_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the D11a diagram

        g*(f, u, l, t) <- l^{i}_{a} t^{c}_{k} u^{jk}_{bc} P(ab) P(ij)

    We do this in two steps

        W^{j}_{b} = t^{c}_{k} u^{jk}_{bc}
        g*(f, u, l, t) <- l^{i}_{a} W^{j}_{b} P(ab) P(ij)

    Number of FLOPS required: O(m^2 n^3)
    """

    # Starting with terms 2 and 3
    term = np.tensordot(t_1, u[o, o, v, v], axes=((0, 1), (3, 1)))  # jb
    term = np.tensordot(l_1, term, axes=0).transpose(0, 2, 1, 3)  # iajb -> ijab
    term -= term.swapaxes(2, 3)
    term -= term.swapaxes(0, 1)
    out += term


def add_d11b_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the D11b diagram

        g*(f, u, l, t) <- l^{ik}_{ac} t^{d}_{k} u^{jc}_{bd} P(ab) P(ij)

    We do this in two steps

        W^{di}_{ac} = t^{d}_{k} l^{ik}_{ac}
        g*(f, u, l, t) <- W^{di}_{ac} u^{jc}_{bd} P(ab) P(ij)

    Number of FLOPS required: O(m^4 n^2)
    """

    term = np.tensordot(l_2, t_1, axes=((1), (1)))  # iacd
    term = np.tensordot(term, u[o, v, v, v], axes=((2, 3), (1, 3))).transpose(
        0, 2, 1, 3
    )  # iajb -> ijab
    term -= term.swapaxes(2, 3)
    term -= term.swapaxes(0, 1)
    out += term


def add_d11c_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the D11c diagram

        g*(f, u, l, t) <- (-1) l^{ik}_{ac} t^{c}_{l} u^{jl}_{bk} P(ab) P(ij)

    We do this in two steps

        W^{ik}_{al} = (-1) l^{ik}_{ac} t^{c}_{l}
        g*(f, u, l, t) <- W^{ik}_{al} u^{jl}_{bk} P(ab) P(ij)

    Number of FLOPS required: O(m^2 n^4)
    """

    term = (-1) * np.tensordot(l_2, t_1, axes=((3), (0)))  # ikal
    term = np.tensordot(term, u[o, o, v, o], axes=((1, 3), (3, 1))).transpose(
        0, 2, 1, 3
    )  # iajb -> ijab
    term -= term.swapaxes(2, 3)
    term -= term.swapaxes(0, 1)
    out += term


def add_d12a_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the D12a diagram

        g*(f, u, l, t) <- (-1) l^{ij}_{ac} t^{c}_{k} t^{d}_{l} u^{kl}_{bd} P(ab)

    We do this in three steps

        W^{k}_{b} = (-1) t^{d}_{l} u^{kl}_{bd}
        W^{c}_{b} = t^{c}_{k} W^{k}_{b}
        g*(f, u, l, t) <- l^{ij}_{ac} W^{c}_{b} P(ab)

    Number of FLOPS required: O(m^3 n^2)
    """

    # Start from the back
    term = (-1) * np.tensordot(t_1, u[o, o, v, v], axes=((0, 1), (3, 1)))  # kb
    term = np.dot(t_1, term)
    term = np.tensordot(l_2, term, axes=((3), (0)))  # ijab
    term -= term.swapaxes(2, 3)
    out += term


def add_d12b_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the D12b diagram

        g*(f, u, l, t) <- (-1) l^{ik}_{ab} t^{c}_{k} t^{d}_{l} u^{jl}_{cd} P(ij)

    We do this in three steps

        W^{j}_{c} = (-1) t^{d}_{l} u^{jl}_{cd}
        W^{j}_{k} = W^{j}_{c} t^{c}_{k}
        g*(f, u, l, t) <- l^{ik}_{ab} W^{j}_{k} P(ij)

    Number of FLOPS required: O(m^2 n^3)
    """

    # Starting from the back again
    term = (-1) * np.tensordot(t_1, u[o, o, v, v], axes=((0, 1), (3, 1)))  # jc
    term = np.dot(term, t_1)
    term = np.tensordot(term, l_2, axes=((1), (1)))
    term -= term.swapaxes(0, 1)
    out -= term


def add_d12c_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the D12c diagram

        g*(f, u, l, t) <- (-1) l^{ik}_{ac} t^{c}_{l} t^{d}_{k} u^{jl}_{bd} P(ab) P(ij)

    We do this in three steps

        W^{jl}_{bk} = (-1) u^{jl}_{bd} t^{d}_{k}
        W^{cj}_{bk} = t^{c}_{l} W^{jl}_{bk}
        g*(f, u, l, t) <- l^{ik}_{ac} W^{cj}_{bk} P(ab) P(ij)

    Number of FLOPS required: O(m^3 n^3)
    """

    # From the back
    term = (-1) * np.tensordot(t_1, u[o, o, v, v], axes=((0), (3)))  # kjlb
    term = np.tensordot(t_1, term, axes=((1), (2)))  # ckjb
    term = np.tensordot(l_2, term, axes=((1, 3), (1, 0))).transpose(
        0, 2, 1, 3
    )  # iajb -> ijab
    term -= term.swapaxes(2, 3)
    term -= term.swapaxes(0, 1)
    out += term
