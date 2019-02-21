def compute_l_1_amplitudes(f, u, t_1, t_2, l_1, l_2, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(l_1)

    add_s1_l(f, o, v, out, np=np)
    add_s2a_l(f, l_1, o, v, out, np=np)
    add_s2b_l(f, l_1, o, v, out, np=np)
    add_s3a_l(u, l_1, o, v, out, np=np)
    add_s3b_l(u, t_1, o, v, out, np=np)
    add_s4a_l(u, l_2, o, v, out, np=np)
    add_s4b_l(u, l_2, o, v, out, np=np)
    add_s5a_l(u, l_1, t_1, o, v, out, np=np)
    add_s5b_l(u, l_1, t_1, o, v, out, np=np)
    add_s5c_l(u, l_1, t_1, o, v, out, np=np)
    add_s5d_l(u, l_1, t_1, o, v, out, np=np)
    add_s6a_l(u, l_2, t_1, o, v, out, np=np)
    add_s6b_l(u, l_2, t_1, o, v, out, np=np)
    add_s6c_l(u, l_2, t_1, o, v, out, np=np)
    add_s6d_l(u, l_2, t_2, o, v, out, np=np)
    add_s7_l(u, l_1, t_2, o, v, out, np=np)
    add_s8a_l(f, l_1, t_1, o, v, out, np=np)
    add_s8b_l(f, l_1, t_1, o, v, out, np=np)
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
    add_s11f_l(u, l_1, t_1, o, v, out, np=np)
    add_s11g_l(u, l_1, t_1, o, v, out, np=np)
    add_s11h_l(u, l_1, t_1, o, v, out, np=np)
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

    return out


# Here begins the L_1 stuff
# Note to self: everything output is upside-down and mirrored.


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

    term = np.tensordot(t_1, u[v, o, v, v], axes=((0, 1), (3, 1)))  # ba
    out += np.tensordot(l_1, term, axes=((1), (0)))  # ia


def add_s5b_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S5b diagram

        g*(f, u, l, t) <- l^{j}_{a} t^{b}_{k} u^{ik}_{bj}

    Number of FLOPS required: O()
    """

    term = np.tensordot(t_1, u[o, o, v, o], axes=((0, 1), (2, 1)))  # ij
    out += np.tensordot(l_1, term, axes=((0), (1))).transpose(1, 0)  # ai -> ia


def add_s5c_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S5c diagram

        g*(f, u, l, t) <- l^{j}_{b} t^{c}_{j} u^{ib}_{ac}

    Number of FLOPS required: O()
    """

    term = np.tensordot(l_1, t_1, axes=((0), (1)))  # bc
    out += np.tensordot(term, u[o, v, v, v], axes=((0, 1), (1, 3)))  # ia


def add_s5d_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S5d diagram

        g*(f, u, l, t) <- (-1) l^{j}_{b} t^{b}_{k} u^{ik}_{aj}

    Number of FLOPS required: O()
    """

    term = (-1) * np.tensordot(l_1, t_1, axes=((1), (0)))  # jk
    out += np.tensordot(term, u[o, o, v, o], axes=((0, 1), (3, 1)))  # ia


def add_s6a_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S6a diagram

        g*(f, u, l, t) <- l^{ij}_{bc} t^{b}_{k} u^{ck}_{aj}

    Number of FLOPS required: O()
    """

    term = np.tensordot(l_2, t_1, axes=((2), (0)))  # ijck
    out += np.tensordot(term, u[v, o, v, o], axes=((1, 2, 3), (3, 0, 1)))  # ia


def add_s6b_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S6b diagram

        g*(f, u, l, t) <- (0.5) l^{ij}_{bc} t^{d}_{j} u^{bc}_{ad}

    Number of FLOPS required: O()
    """

    term = (0.5) * np.tensordot(l_2, t_1, axes=((1), (1)))  # ibcd
    out += np.tensordot(term, u[v, v, v, v], axes=((1, 2, 3), (0, 1, 3)))  # ia


def add_s6c_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S6c diagram

        g*(f, u, l, t) <- (0.5) l^{jk}_{ab} t^{b}_{l} u^{il}_{jk}

    Number of FLOPS required: O()
    """

    term = (0.5) * np.tensordot(l_2, t_1, axes=((3), (0)))  # jkal
    out += np.tensordot(
        term, u[o, o, o, o], axes=((0, 1, 3), (2, 3, 1))
    ).transpose(
        1, 0
    )  # ai -> ia


def add_s6d_l(u, l_2, t_2, o, v, out, np):
    """Function for adding the S6d diagram

        g*(f, u, l, t) <- (0.5) l^{jk}_{bc} t^{bd}_{jk} u^{ic}_{ad}

    Number of FLOPS required: O()
    """

    term = (0.5) * np.tensordot(l_2, t_2, axes=((0, 1, 2), (2, 3, 0)))  # cd
    out += np.tensordot(term, u[o, v, v, v], axes=((0, 1), (1, 3)))  # ia


def add_s7_l(u, l_1, t_2, o, v, out, np):
    """Function for adding the S7 diagram (naming probably wrong)

        g*(f, u, l, t) <- l^{j}_{b} t^{bc}_{jk} u^{ik}_{ac}

    Number of FLOPS required: O()
    """

    term = np.tensordot(l_1, t_2, axes=((0, 1), (2, 0)))  # ck
    out += np.tensordot(term, u[o, o, v, v], axes=((0, 1), (3, 1)))  # ia


def add_s8a_l(f, l_1, t_1, o, v, out, np):
    """Function for adding the S8a diagram

        g*(f, u, l, t) <- (-1) f^{i}_{b} l^{j}_{a} t^{b}_{j}

    Number of FLOPS required: O()
    """

    term = (-1) * np.tensordot(l_1, t_1, axes=((0), (1)))  # ab
    out += np.tensordot(f[o, v], term, axes=((1), (1)))  # ia


def add_s8b_l(f, l_1, t_1, o, v, out, np):
    """Function for adding the S8b diagram

        g*(f, u, l, t) <- (-1) f^{j}_{a} l^{i}_{b} t^{b}_{j}

    Number of FLOPS required: O()
    """

    term = (-1) * np.tensordot(l_1, t_1, axes=((1), (0)))  # ij
    out += np.tensordot(f[o, v], term, axes=((0), (1))).transpose(
        1, 0
    )  # ai -> ia


def add_s9a_l(u, l_2, t_2, o, v, out, np):
    """Function for adding the S9a diagram

        g*(f, u, l, t) <- (-1) l^{ij}_{bc} t^{bd}_{jk} u^{ck}_{ad}
    
    Number of FLOPS required: O()
    """

    term = (-1) * np.tensordot(l_2, t_2, axes=((1, 2), (2, 0)))  # icdk
    out += np.tensordot(term, u[v, o, v, v], axes=((1, 2, 3), (0, 3, 1)))  # ia


def add_s9b_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S9b diagram

        g*(f, u, l, t) <- (-1) l^{jk}_{ab} t^{c}_{j} u^{ib}_{ck}

    Number of FLOPS required: O()
    """

    term = (-1) * np.tensordot(l_2, t_1, axes=((0), (1)))  # kabc
    out += np.tensordot(
        term, u[o, v, v, o], axes=((0, 2, 3), (3, 1, 2))
    ).transpose(
        1, 0
    )  # ai -> ia


def add_s9c_l(u, l_2, t_2, o, v, out, np):
    """Function for adding the S9c diagram

        g*(f, u, l, t) <- (-1) l^{jk}_{ab} t^{bc}_{jl} u^{il}_{ck}

    Number of FLOPS required: O()
    """

    term = (-1) * np.tensordot(l_2, t_2, axes=((0, 3), (2, 0)))  # kacl
    out += np.tensordot(
        term, u[o, o, v, o], axes=((0, 2, 3), (3, 2, 1))
    ).transpose(
        1, 0
    )  # ai -> ia


def add_s10a_l(f, l_2, t_2, o, v, out, np):
    """Function for adding the S10a diagram

        g*(f, u, l, t) <- (-0.5) f^{i}_{b} l^{jk}_{ac} t^{bc}_{jk}

    Number of FLOPS required: O()
    """

    term = (-0.5) * np.tensordot(l_2, t_2, axes=((0, 1, 3), (2, 3, 1)))  # ab
    out += np.tensordot(f[o, v], term, axes=((1), (1)))  # ia


def add_s10b_l(f, l_2, t_2, o, v, out, np):
    """Function for adding the S10b diagram

        g*(f, u, l, t) <- (-0.5) f*{j}_{a} l^{ik}_{bc} t^{bc}_{jk}

    Number of FLOPS required: O()
    """

    term = (-0.5) * np.tensordot(l_2, t_2, axes=((1, 2, 3), (3, 0, 1)))  # ij
    out += np.tensordot(f[o, v], term, axes=((0), (1))).transpose(
        1, 0
    )  # ai - > ia


def add_s10c_l(u, l_1, t_2, o, v, out, np):
    """Function for adding the S10c diagram

        g*(f, u, l, t) <- (-0.5) l^{i}_{b} t^{bc}_{jk} u^{jk}_{ac}

    Number of FLOPS required: O()
    """

    term = (-0.5) * np.tensordot(l_1, t_2, axes=((1), (0)))  # icjk
    out += np.tensordot(term, u[o, o, v, v], axes=((1, 2, 3), (3, 0, 1)))  # ia


def add_s10d_l(u, l_1, t_2, o, v, out, np):
    """Function for adding the S10d diagram

        g*(f, u, l, t) <- (-0.5) l^{j}_{a} t^{bc}_{jk} u^{ik}_{bc}

    Number of FLOPS required: O()
    """

    term = (-0.5) * np.tensordot(l_1, t_2, axes=((0), (2)))  # abck
    out += np.tensordot(
        term, u[o, o, v, v], axes=((1, 2, 3), (2, 3, 1))
    ).transpose(
        1, 0
    )  # ai -> ia


def add_s10e_l(u, l_2, t_2, o, v, out, np):
    """Function for adding the S10e diagram

        g*(f, u, l, t) <- (-0.5) l^{jk}_{bc} t^{bc}_{jl} u^{il}_{ak}

    Number of FLOPS required: O()
    """

    term = (-0.5) * np.tensordot(l_2, t_2, axes=((0, 2, 3), (2, 0, 1)))  # kl
    out += np.tensordot(term, u[o, o, v, o], axes=((0, 1), (3, 1)))  # ia


def add_s10f_l(u, l_2, t_2, o, v, out, np):
    """Function for adding the S10f diagram

        g*(f, u, l, t) <- (-0.25) l^{jk}_{ab} t^{cd}_{jk} u^{ib}_{cd}

    Number of FLOPS required: O()
    """

    term = (-0.25) * np.tensordot(l_2, t_2, axes=((0, 1), (2, 3)))  # abcd
    out += np.tensordot(
        term, u[o, v, v, v], axes=((1, 2, 3), (1, 2, 3))
    ).transpose(
        1, 0
    )  # ai -> ia


def add_s10g_l(u, l_2, t_2, o, v, out, np):
    """Function for adding the S10g diagram

        g*(f, u, l, t) <- (0.25) l^{ij}_{bc} t^{bc}_{kl} u^{kl}_{aj}

    Number of FLOPS required: O()
    """

    term = (0.25) * np.tensordot(l_2, t_2, axes=((2, 3), (0, 1)))  # ijkl
    out += np.tensordot(term, u[o, o, v, o], axes=((1, 2, 3), (3, 0, 1)))  # ia


def add_s11a_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S11a diagram

        g*(f, u, l, t) <- l^{ij}_{bc} t^{b}_{k} t^{d}_{j} u^{ck}_{ad}

    Number of FLOPS required: O()
    """

    term = np.tensordot(l_2, t_1, axes=((2), (0)))  # ijck
    term = np.tensordot(term, t_1, axes=((1), (1)))  # ickd
    out += np.tensordot(term, u[v, o, v, v], axes=((1, 2, 3), (0, 1, 3)))  # ia


def add_s11b_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S11b diagram

        g*(f, u, l, t) <- l^{jk}_{ab} t^{b}_{l} t^{c}_{j} u^{il}_{ck}

    Number of FLOPS required: O()
    """

    term = np.tensordot(l_2, t_1, axes=((3), (0)))  # jkal
    term = np.tensordot(term, u[o, o, v, o], axes=((1, 3), (3, 1)))  # jaic
    out += np.tensordot(t_1, term, axes=((0, 1), (3, 0))).transpose(
        1, 0
    )  # ai -> ia


def add_s11c_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S11c diagram

        g*(f, u, l, t) <- (0.5) l^{jk}_{ab} t^{c}_{k} t^{d}_{j} u^{ib}_{cd}

    Number of FLOPS required: O()
    """

    term = (0.5) * np.tensordot(l_2, t_1, axes=((1), (1)))  # jabc
    term = np.tensordot(term, u[o, v, v, v], axes=((2, 3), (1, 2)))  # jaid
    out += np.tensordot(t_1, term, axes=((0, 1), (3, 0))).transpose(
        1, 0
    )  # ai -> ia


def add_s11d_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11d diagram

        g*(f, u, l, t) <- (0.5) l^{jk}_{bc} t^{b}_{l} t^{cd}_{jk} u^{il}_{ad}

    Number of FLOPS required: O()
    """

    term = (0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # jkcl
    term = np.tensordot(term, t_2, axes=((0, 1, 2), (2, 3, 0)))  # ld
    out += np.tensordot(term, u[o, o, v, v], axes=((0, 1), (1, 3)))  # ia


def add_s11e_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11e diagram

        g*(f, u, l, t) <- (0.5) * l^{jk}_{bc} t^{d}_{j} t^{bc}_{kl} u^{il}_{ad}

    Number of FLOPS required: O()
    """

    term = (0.5) * np.tensordot(l_2, t_1, axes=((0), (1)))  # kbcd
    term = np.tensordot(term, t_2, axes=((0, 1, 2), (2, 0, 1)))  # dl
    out += np.tensordot(term, u[o, o, v, v], axes=((0, 1), (3, 1)))  # ia


def add_s11f_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S11f diagram

        g*(f, u, l, t) <- l^{i}_{b} t^{b}_{j} t^{c}_{k} u^{jk}_{ac}

    Number of FLOPS required: O()
    """

    term = (-1) * np.tensordot(l_1, t_1, axes=((1), (0)))  # ij
    term = np.tensordot(term, u[o, o, v, v], axes=((1), (0)))  # ikac
    out += np.tensordot(t_1, term, axes=((0, 1), (3, 1)))  # ia


def add_s11g_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S11g diagram

        g*(f, u, l, t) <- (-1) l^{j}_{a} t^{b}_{j} t^{c}_{k} u^{ik}_{bc}

    Number of FLOPS required: O()
    """

    term = (-1) * np.tensordot(l_1, t_1, axes=((0), (1)))  # ab
    term = np.tensordot(term, u[o, o, v, v], axes=((1), (2)))  # aikc
    out += np.tensordot(t_1, term, axes=((0, 1), (3, 2))).transpose(
        1, 0
    )  # ai -> ia


def add_s11h_l(u, l_1, t_1, o, v, out, np):
    """Function for adding the S11h diagram

        g*(f, u, l, t) <- (-1) l^{j}_{b} t^{b}_{k} t^{c}_{j} u^{ik}_{ac}

    Number of FLOPS required: O()
    """

    term = (-1) * np.tensordot(l_1, t_1, axes=((1), (0)))  # jk
    term = np.tensordot(term, t_1, axes=((0), (1)))  # kc
    out += np.tensordot(term, u[o, o, v, v], axes=((0, 1), (1, 3)))  # ia


def add_s11i_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11i diagram

        g*(f, u, l, t) <- (-1) l^{ij}_{bc} t^{b}_{k} t^{cd}_{jl} u^{kl}_{ad}

    Number of FLOPS required: O()
    """

    term = (-1) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijck
    term = np.tensordot(term, t_2, axes=((1, 2), (2, 0)))  # ikdl
    out += np.tensordot(term, u[o, o, v, v], axes=((1, 2, 3), (0, 3, 1)))  # ia


def add_s11j_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11j diagram

        g*(f, u, l, t) <- (-1) l^{jk}_{ab} t^{c}_{j} t^{bd}_{kl} u^{il}_{cd}

    Number of FLOPS required: O
    """

    term = (-1) * np.tensordot(l_2, t_1, axes=((0), (1)))  # kabc
    term = np.tensordot(term, t_2, axes=((0, 2), (2, 0)))  # acdl
    out += np.tensordot(
        term, u[o, o, v, v], axes=((1, 2, 3), (2, 3, 1))
    ).transpose(
        1, 0
    )  # ai -> ia


def add_s11k_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S11k diagram

        g*(f, u, l, t) <- (-0.5) l^{ij}_{bc} t^{b}_{l} t^{c}_{k} u^{kl}_{aj}

    Number of FLOPS required: O()
    """

    term = (-0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijcl
    term = np.tensordot(term, t_1, axes=((2), (0)))  # ijlk
    out += np.tensordot(term, u[o, o, v, o], axes=((1, 2, 3), (3, 1, 0)))  # ia


def add_s11l_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11k diagram

        g*(f, u, l, t) <- (-0.5) l^{ij}_{bc} t^{d}_{k} t^{bc}_{jl} u^{kl}_{ad}

    Number of FLOPS required: O()
    """

    # Starting with term 1 (l_2) and term 3 (t_2)
    term = (-0.5) * np.tensordot(l_2, t_2, axes=((1, 2, 3), (2, 0, 1)))  # il
    # Then term 4 (u)
    term = np.tensordot(term, u[o, o, v, v], axes=((1), (1)))  # ikad
    # Lastly, term 2 (t_1)
    out += np.tensordot(t_1, term, axes=((0, 1), (3, 1)))  # ia


def add_s11m_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11m diagram

        g*(f, u, l, t) <- (-0.5) l^{jk}_{ab} t^{c}_{l} t^{bd}_{jk} u^{il}_{cd}

    Number of FLOPS required: O()
    """

    # Same as over,
    # Starting with term 1 (l_2) and term 3 (t_2)
    term = (-0.5) * np.tensordot(l_2, t_2, axes=((0, 1, 3), (2, 3, 0)))  # ad
    # Then term 4 (u)
    term = np.tensordot(term, u[o, o, v, v], axes=((1), (3)))  # ailc
    # Lastly, term 2 (t_1)
    out += np.tensordot(t_1, term, axes=((0, 1), (3, 2))).transpose(
        1, 0
    )  # ai -> ia


def add_s11n_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11n diagram

        g*(f, u, l t) <- (0.25) l^{ij}_{bc} t^{d}_{j} t^{bc}_{kl} u^{kl}_{ad}

    Number of FLOPS required: O()
    """

    term = (0.25) * np.tensordot(l_2, t_1, axes=((1), (1)))  # ibcd
    term = np.tensordot(term, t_2, axes=((1, 2), (0, 1)))  # idkl
    out += np.tensordot(term, u[o, o, v, v], axes=((1, 2, 3), (3, 0, 1)))  # ia


def add_s11o_l(u, l_2, t_1, t_2, o, v, out, np):
    """Function for adding the S11o diagram

        g*(f, u, l, t) <- (0.25) l^{jk}_{ab} t^{b}_{l} t^{cd}_{jk} u^{il}_{cd}

    Number of FLOPS required: O()
    """

    term = (0.25) * np.tensordot(l_2, t_1, axes=((3), (0)))  # jkal
    term = np.tensordot(term, t_2, axes=((0, 1), (2, 3)))  # alcd
    out += np.tensordot(
        term, u[o, o, v, v], axes=((1, 2, 3), (1, 2, 3))
    ).transpose(
        1, 0
    )  # ai -> ia


def add_s12a_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S12a diagram

        g*(f, u, l, t) <- (-0.5) l^{ij}_{bc} t^{b}_{l} t^{c}_{k} t^{d}_{j} u^{kl}_{ad}

    Number of FLOPS required: O()
    """

    term = (-0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijcl
    term = np.tensordot(term, t_1, axes=((2), (0)))  # ijlk
    term = np.tensordot(term, t_1, axes=((1), (1)))  # ilkd
    out += np.tensordot(term, u[o, o, v, v], axes=((1, 2, 3), (1, 0, 3)))  # ia


def add_s12b_l(u, l_2, t_1, o, v, out, np):
    """Function for adding the S12b diagram

        g*(f, u, l, t) <- (-0.5) * l^{jk}_{ab} t^{b}_{l} t^{c}_{k} t^{d}_{j} u^{il}_{cd}

    Number of FLOPS required: O()
    """

    term = (-0.5) * np.tensordot(l_2, t_1, axes=((3), (0)))  # jkal
    term = np.tensordot(term, t_1, axes=((1), (1)))  # jalc
    term = np.tensordot(term, t_1, axes=((0), (1)))  # alcd
    out += np.tensordot(
        term, u[o, o, v, v], axes=((1, 2, 3), (1, 2, 3))
    ).transpose(
        1, 0
    )  # ai -> ia
