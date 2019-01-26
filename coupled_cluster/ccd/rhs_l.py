# We use equation A9b in Simen Kvaal's article "Ab initio quantum dynamics using
# coupled-cluster". The labelling of the diagrams are based on the labelling
# done in the book "Many-Body Methods in Chemistry and Physics" by I. Shavitt
# and R. J. Bartlett.


def compute_l_2_amplitudes(f, u, t, l, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(l)

    add_d1_l(u, o, v, out, np=np)
    add_d2a_l(u, l, o, v, out, np=np)
    add_d2b_l(u, l, o, v, out, np=np)
    add_d2c_l(f, l, o, v, out, np=np)
    add_d2d_l(f, l, o, v, out, np=np)
    add_d2e_l(u, l, o, v, out, np=np)
    add_d3a_l(u, t, l, o, v, out, np=np)
    add_d3b_l(u, t, l, o, v, out, np=np)
    add_d3c_l(u, t, l, o, v, out, np=np)
    add_d3d_l(u, t, l, o, v, out, np=np)
    add_d3e_l(u, t, l, o, v, out, np=np)
    add_d3f_l(u, t, l, o, v, out, np=np)
    add_d3g_l(u, t, l, o, v, out, np=np)

    return out


def add_d1_l(u, o, v, out, np):
    """Function adding the D1 diagram

        g(f, u, t, l) <- u^{ij}_{ab}

    Number of FLOPS required: O(m^2 n^2).
    """
    out += u[o, o, v, v]


def add_d2a_l(u, l, o, v, out, np):
    """Function adding the D2a diagram

        g(f, u, t, l) <- 0.5 * l^{kl}_{ab} u^{ij}_{kl}

    Number of FLOPS required: O(m^2 n^4).
    """
    out += 0.5 * np.tensordot(u[o, o, o, o], l, axes=((2, 3), (0, 1)))


def add_d2b_l(u, l, o, v, out, np):
    """Function adding the D2b diagram

        g(f, u, t, l) <- 0.5 * l^{ij}_{dc} u^{dc}_{ab}

    Number of FLOPS required: O(m^4 n^2).
    """
    out += 0.5 * np.tensordot(l, u[v, v, v, v], axes=((2, 3), (0, 1)))


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


def add_d2e_l(u, l, o, v, out, np):
    """Function adding the D2e diagram

        g(f, u, t, l) <- l^{jk}_{bc} u^{ic}_{ak} P(ab) P(ij)

    Number of FLOPS required: O(m^3 n^3).
    """
    temp_abij = np.tensordot(l, u[o, v, v, o], axes=((1, 3), (3, 1))).transpose(
        2, 0, 3, 1
    )
    temp_abij -= temp_abij.swapaxes(0, 1)
    temp_abij -= temp_abij.swapaxes(2, 3)
    out += temp_abij


def add_d3a_l(u, t, l, o, v, out, np):
    """Function adding the D3a diagram

        g(f, u, t, l) <- -0.5 l^{ij}_{bc} t^{dc}_{kl} u^{kl}_{ad} P(ab)

    We do this in two steps

        W^{c}_{a} = 0.5 * t^{dc}_{kl} u^{kl}_{ad}
        g(f, u, t, l) <- -l^{ij}_{bc} W^{c}_{a} P(ab)

    Number of FLOPS required: O(m^3 n^2).
    """
    W_ca = 0.5 * np.tensordot(t, u[o, o, v, v], axes=((0, 2, 3), (3, 0, 1)))
    temp = np.tensordot(l, W_ca, axes=((3), (0))).transpose(0, 1, 3, 2)
    temp -= temp.swapaxes(2, 3)
    out -= temp


def add_d3b_l(u, t, l, o, v, out, np):
    """Function adding the D3b diagram

        g(f, u, t, l) <- 0.25 * l^{ij}_{dc} t^{dc}_{kl} u^{kl}_{ab}

    We do this in two steps and in one of two ways depending on the number of
    occupied indices.

    1) If half, or more, of the basis functions are occupied, we
    precompute

        W^{dc}_{ab} = 0.25 * t^{dc}_{kl} u^{kl}_{ab}
        g(f, u, t, l) <- l^{ij}_{dc} W^{dc}_{ab}

    Number of FLOPS required: O(m^4 n^2).

    2) If less than half of the basis functions are occupied, we precompute

        W^{ij}_{kl} = 0.25 * l^{ij}_{dc} t^{dc}_{kl}
        g(f, u, t, l) <- W^{ij}_{kl} u^{kl}_{ab}

    Number of FLOPS required: O(m^2 n^4).
    """
    if o.stop >= v.stop // 2:
        # Case 1
        W_dcab = 0.25 * np.tensordot(t, u[o, o, v, v], axes=((2, 3), (0, 1)))
        out += np.tensordot(l, W_dcab, axes=((2, 3), (0, 1)))
    else:
        # Case 2
        W_ijkl = 0.25 * np.tensordot(l, t, axes=((2, 3), (0, 1)))
        out += np.tensordot(W_ijkl, u[o, o, v, v], axes=((2, 3), (0, 1)))


def add_d3c_l(u, t, l, o, v, out, np):
    """Function adding the D3c diagram

        g(f, u, t, l) <- 0.5 * l^{jk}_{ab} t^{dc}_{kl} u^{il}_{dc} P(ij)

    We do this in two steps

        W^{i}_{k} = 0.5 * t^{dc}_{kl} u^{il}_{dc}
        g(f, u, t, l) <- l^{jk}_{ab} W^{i}_{k} P(ij)

    Number of FLOPS required: O(m^2 n^3).
    """
    W_ik = 0.5 * np.tensordot(u[o, o, v, v], t, axes=((1, 2, 3), (3, 0, 1)))
    temp_abij = np.tensordot(W_ik, l, axes=((1), (1)))
    temp_abij -= temp_abij.swapaxes(0, 1)
    out += temp_abij


def add_d3d_l(u, t, l, o, v, out, np):
    """Function adding the D3d diagram

        g(f, u, t, l) <- -l^{jk}_{bc} t^{dc}_{kl} u^{il}_{ad} P(ab) P(ij)

    We do this in two steps

        W^{jd}_{bl} = l^{jk}_{bc} t^{dc}_{kl}
        g(f, u, t, l) <- -W^{jd}_{bl} u^{il}_{ad} P(ab) P(ij)

    Number of FLOPS required: O(m^3 n^3).
    """
    W_jdbl = np.tensordot(l, t, axes=((1, 3), (2, 1))).transpose(0, 2, 1, 3)
    term_abij = np.tensordot(
        u[o, o, v, v], W_jdbl, axes=((1, 3), (3, 1))
    ).transpose(0, 2, 1, 3)
    term_abij -= term_abij.swapaxes(0, 1)
    term_abij -= term_abij.swapaxes(2, 3)
    out -= term_abij


def add_d3e_l(u, t, l, o, v, out, np):
    """Function adding the D3e diagram

        g(f, u, t, l) <- 0.5 * l^{jk}_{dc} t^{dc}_{kl} u^{il}_{ab} P(ij)

    We do this in two steps

        W^{j}_{l} = 0.5 * l^{jk}_{dc} t^{dc}_{kl}
        g(f, u, t, l) <- W^{j}_{l} u^{il}_{ab} P(ij)

    Number of FLOPS required: O(m^2 n^3).
    """
    W_jl = 0.5 * np.tensordot(l, t, axes=((1, 2, 3), (2, 0, 1)))
    term_abij = np.tensordot(W_jl, u[o, o, v, v], axes=((1), (1))).transpose(
        1, 0, 2, 3
    )
    term_abij -= term_abij.swapaxes(0, 1)
    out += term_abij


def add_d3f_l(u, t, l, o, v, out, np):
    """Function adding the D3f diagram

        g(f, u, t, l) <- 0.25 * l^{kl}_{ab} t^{dc}_{kl} u^{ij}_{dc}

    We do this in two steps and in one of two ways depending on the number of
    occupied indices.

    1) If half, or more, of the basis functions are occupied, we
    precompute

        W^{dc}_{ab} = 0.25 * l^{kl}_{ab} t^{dc}_{kl}
        g(f, u, t, l) <- W^{dc}_{ab} u^{ij}_{dc}

    Number of FLOPS required: O(m^4 n^2).

    2) If less than half of the basis functions are occupied, we precompute

        W^{ij}_{kl} = 0.25 * t^{dc}_{kl} u^{ij}_{dc}
        g(f, u, t, l) <- l^{kl}_{ab} W^{ij}_{kl}

    Number of FLOPS required: O(m^2 n^4).
    """
    if o.stop >= v.stop // 2:
        # Case 1
        W_dcab = 0.25 * np.tensordot(t, l, axes=((2, 3), (0, 1)))
        out += np.tensordot(u[o, o, v, v], W_dcab, axes=((2, 3), (0, 1)))
    else:
        # Case 2
        W_ijkl = 0.25 * np.tensordot(u[o, o, v, v], t, axes=((2, 3), (0, 1)))
        out += np.tensordot(W_ijkl, l, axes=((2, 3), (0, 1)))


def add_d3g_l(u, t, l, o, v, out, np):
    """Function adding the D3g diagram

        g(f, u, t, l) <- -0.5 * l^{kl}_{bc} t^{dc}_{kl} u^{ij}_{ad} P(ab)

    We do this in two steps

        W^{d}_{b} = 0.5 * l^{kl}_{bc} t^{dc}_{kl}
        g(f, u, t, l) <- -W^{d}_{b} u^{ij}_{ad} P(ab)

    Number of FLOPS required: O(m^3 n^2).
    """
    W_db = 0.5 * np.tensordot(t, l, axes=((1, 2, 3), (3, 0, 1)))
    term_abij = np.tensordot(u[o, o, v, v], W_db, axes=((3), (0)))
    term_abij -= term_abij.swapaxes(2, 3)
    out -= term_abij
