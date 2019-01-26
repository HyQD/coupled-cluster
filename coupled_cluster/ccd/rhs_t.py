# Labelling of the different terms comes from the book "Many-Body Methods in
# Chemistry and Physics" by I. Shavitt and R. J. Bartlett.


def compute_t_2_amplitudes(f, u, t, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(t)

    add_d1_t(u, o, v, out, np=np)
    add_d2a_t(f, t, o, v, out, np=np)
    add_d2b_t(f, t, o, v, out, np=np)
    add_d2c_t(u, t, o, v, out, np=np)
    add_d2d_t(u, t, o, v, out, np=np)
    add_d2e_t(u, t, o, v, out, np=np)
    add_d3a_t(u, t, o, v, out, np=np)
    add_d3b_t(u, t, o, v, out, np=np)
    add_d3c_t(u, t, o, v, out, np=np)
    add_d3d_t(u, t, o, v, out, np=np)

    return out


def add_d1_t(u, o, v, out, np):
    """Function adding the D1 diagram

        g(f, u, t) <- u^{ab}_{ij}

    Number of FLOPS required: O(m^2 n^2).
    """
    out += u[v, v, o, o]


def add_d2a_t(f, t, o, v, out, np):
    """Function adding the D2a diagram

        g(f, u, t) <- f^{b}_{c} t^{ac}_{ij} P(ab)

    Number of FLOPS required: O(m^3 n^2).
    """
    term = np.tensordot(f[v, v], t, axes=((1), (1))).transpose(1, 0, 2, 3)
    term -= term.swapaxes(0, 1)
    out += term


def add_d2b_t(f, t, o, v, out, np):
    """Function adding the D2b diagram

        g(f, u, t) <- -f^{k}_{j} t^{ab}_{ik} P(ij)

    Number of FLOPS required: O(m^2 n^3).
    """
    term = np.tensordot(t, f[o, o], axes=((3), (0)))
    term -= term.swapaxes(2, 3)
    out -= term


def add_d2c_t(u, t, o, v, out, np):
    """Function adding the D2c diagram

        g(f, u, t) <- 0.5 * t^{cd}_{ij} u^{ab}_{cd}

    Number of FLOPS required: O(m^4 n^2).
    """
    out += 0.5 * np.tensordot(u[v, v, v, v], t, axes=((2, 3), (0, 1)))


def add_d2d_t(u, t, o, v, out, np):
    """Function adding the D2d diagram

        g(f, u, t) <- 0.5 * t^{ab}_{kl} u^{kl}_{ij}

    Number of FLOPS required: O(m^2 n^4).
    """
    out += 0.5 * np.tensordot(t, u[o, o, o, o], axes=((2, 3), (0, 1)))


def add_d2e_t(u, t, o, v, out, np):
    """Function adding the D2e diagram

        g(f, u, t) <- t^{ac}_{ik} u^{kb}_{cj} P(ab) P(ij)

    Number of FLOPS required: O(m^3 n^3).
    """
    term = np.tensordot(t, u[o, v, v, o], axes=((1, 3), (2, 0))).transpose(
        0, 2, 1, 3
    )
    term -= term.swapaxes(0, 1)
    term -= term.swapaxes(2, 3)
    out += term


def add_d3a_t(u, t, o, v, out, np):
    """Function adding the D3a diagram

        g(f, u, t) <- 0.25 * t^{cd}_{ij} t^{ab}_{kl} u^{kl}_{cd}

    We do this in two steps and in one of two ways depending on the number of
    occupied indices.

    1) If half, or more, of the basis functions are occupied, we
    precompute

        W^{ab}_{cd} = 0.25 * t^{ab}_{kl} u^{kl}_{cd}
        g(f, u, t) <- t^{cd}_{ij} W^{ab}_{cd}

    Number of FLOPS required: O(m^4 n^2).

    2) If less than half of the basis functions are occupied, we precompute

        W^{kl}_{ij} = 0.25 * t^{cd}_{ij} u^{kl}_{cd}
        g(f, u, t) <- t^{ab}_{kl} W^{kl}_{ij}

    Number of FLOPS required: O(m^2 n^4).
    """
    if o.stop >= v.stop // 2:
        # Case 1
        W_abcd = 0.25 * np.tensordot(t, u[o, o, v, v], axes=((2, 3), (0, 1)))
        out += np.tensordot(W_abcd, t, axes=((2, 3), (0, 1)))
    else:
        # Case 2
        W_klij = 0.25 * np.tensordot(u[o, o, v, v], t, axes=((2, 3), (0, 1)))
        out += np.tensordot(t, W_klij, axes=((2, 3), (0, 1)))


def add_d3b_t(u, t, o, v, out, np):
    """Function adding the D3b diagram

        g(f, u, t) <- t^{ac}_{ik} t^{bd}_{jl} u^{kl}_{cd} P(ij)

    We do this in two steps

        W^{bk}_{jc} = t^{bd}_{jl} u^{kl}_{cd}
        g(f, u, t) <- t^{ac}_{ik} W^{bk}_{jc} P(ij)

    Number of FLOPS required: O(m^3 n^3).
    """
    W_bkjc = np.tensordot(t, u[o, o, v, v], axes=((1, 3), (3, 1))).transpose(
        0, 2, 1, 3
    )
    term = np.tensordot(t, W_bkjc, axes=((1, 3), (3, 1))).transpose(0, 2, 1, 3)
    term -= term.swapaxes(2, 3)
    out += term


def add_d3c_t(u, t, o, v, out, np):
    """Function adding the D3c diagram

        g(f, u, t) <- -0.5 t^{ab}_{lj} t^{dc}_{ik} u^{kl}_{cd} P(ij)

    We do this in two steps

        W^{l}_{i} = 0.5 * t^{dc}_{ik} u^{kl}_{cd}
        g(f, u, t) <- - t^{ab}_{lj} W^{l}_{i} P(ij)

    Number of FLOPS required: O(m^2 n^3).
    """
    W_li = 0.5 * np.tensordot(u[o, o, v, v], t, axes=((0, 2, 3), (3, 1, 0)))
    term = -np.tensordot(t, W_li, axes=((2), (0))).transpose(0, 1, 3, 2)
    term -= term.swapaxes(2, 3)
    out += term


def add_d3d_t(u, t, o, v, out, np):
    """Function adding the D3d diagram

        g(f, u, t) <- -0.5 t^{ac}_{lk} t^{db}_{ij} u^{kl}_{cd} P(ab)

    We do this in two steps

        W^{a}_{d} = 0.5 * t^{ac}_{lk} u^{kl}_{cd}
        g(f, u, t) <- - t^{db}_{ij} W^{a}_{d} P(ab)

    Number of FLOPS required: O(m^3 n^2).
    """
    W_ad = 0.5 * np.tensordot(t, u[o, o, v, v], axes=((1, 2, 3), (2, 1, 0)))
    term = -np.tensordot(W_ad, t, axes=((1), (0)))
    term -= term.swapaxes(0, 1)
    out += term
