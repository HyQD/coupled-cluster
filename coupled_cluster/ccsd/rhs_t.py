# Labelling of the different terms comes from the book "Many-Body Methods in
# Chemistry and Physics" by I. Shavitt and R. J. Bartlett.


# Diagrams for CCSD T_1 amplitude equations

import coupled_cluster.ccs.rhs_t as ccs_t
import coupled_cluster.ccd.rhs_t as ccd_t


def compute_t_1_amplitudes(f, u, t_1, t_2, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(t_1)

    ccs_t.compute_t_1_amplitudes(f, u, t_1, o, v, np=np, out=out)

    add_s2a_t(f, t_2, o, v, out, np=np)
    add_s2b_t(u, t_2, o, v, out, np=np)
    add_s2c_t(u, t_2, o, v, out, np=np)
    add_s4a_t(u, t_1, t_2, o, v, out, np=np)
    add_s4b_t(u, t_1, t_2, o, v, out, np=np)
    add_s4c_t(u, t_1, t_2, o, v, out, np=np)

    return out


def compute_t_2_amplitudes(f, u, t_1, t_2, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(t_2)

    ccd_t.compute_t_2_amplitudes(f, u, t_2, o, v, np=np, out=out)

    add_d4a_t(u, t_1, o, v, out, np=np)
    add_d4b_t(u, t_1, o, v, out, np=np)
    add_d5a_t(f, t_1, t_2, o, v, out, np=np)
    add_d5b_t(f, t_1, t_2, o, v, out, np=np)
    add_d5c_t(u, t_1, t_2, o, v, out, np=np)
    add_d5d_t(u, t_1, t_2, o, v, out, np=np)
    add_d5e_t(u, t_1, t_2, o, v, out, np=np)
    add_d5f_t(u, t_1, t_2, o, v, out, np=np)
    add_d5g_t(u, t_1, t_2, o, v, out, np=np)
    add_d5h_t(u, t_1, t_2, o, v, out, np=np)
    add_d6a_t(u, t_1, o, v, out, np=np)
    add_d6b_t(u, t_1, o, v, out, np=np)
    add_d6c_t(u, t_1, o, v, out, np=np)
    add_d7a_t(u, t_1, t_2, o, v, out, np=np)
    add_d7b_t(u, t_1, t_2, o, v, out, np=np)
    add_d7c_t(u, t_1, t_2, o, v, out, np=np)
    add_d7d_t(u, t_1, t_2, o, v, out, np=np)
    add_d7e_t(u, t_1, t_2, o, v, out, np=np)
    add_d8a_t(u, t_1, o, v, out, np=np)
    add_d8b_t(u, t_1, o, v, out, np=np)
    add_d9_t(u, t_1, o, v, out, np=np)

    return out


# def add_s1_t(f, o, v, out, np):
#     """Function adding the S1 diagram
#
#         g(f, u, t) <- f^{a}_{i}
#
#     Number of FLOPS required: O(m n).
#     """
#     out += f[v, o]


def add_s2a_t(f, t_2, o, v, out, np):
    """Function adding the S2a diagram

        g(f, u, t) <- f^{k}_{c} t^{ac}_{ik}

    Numer of FLOPS required: O(m^2 n^2).
    """
    out += np.tensordot(f[o, v], t_2, axes=((0, 1), (3, 1)))


def add_s2b_t(u, t_2, o, v, out, np):
    """Function adding the S2b diagram

        g(f, u, t) <- 0.5 u^{ak}_{cd} t^{cd}_{ik}

    Number of FLOPS required: O(m^3 n^2).
    """
    out += 0.5 * np.tensordot(u[v, o, v, v], t_2, axes=((1, 2, 3), (3, 0, 1)))


def add_s2c_t(u, t_2, o, v, out, np):
    """Function adding the S2b diagram

        g(f, u, t) <- -0.5 u^{kl}_{ic} t^{ac}_{kl}

    Number of FLOPS required: O(m^2 n^3)
    """
    term = np.tensordot(u[o, o, o, v], t_2, axes=((0, 1), (2, 3)))
    out -= 0.5 * np.trace(term, axis1=1, axis2=3).swapaxes(0, 1)


# def add_s3a_t(f, t_1, o, v, out, np):
#     """Function adding the S3a diagram
#
#         g(f, u, t) <- f^{a}_{c} t^{c}_{i}
#
#     Number of FLOPS required: O(m^2 n)
#     """
#     out += np.tensordot(f[v, v], t_1, axes=((1), (0)))


# def add_s3b_t(f, t_1, o, v, out, np):
#     """Function adding the S3b diagram
#
#         g(f, u, t) <- -f^{k}_{i} t^{a}_{k}
#
#     Number of FLOPS required: O(m n^2)
#     """
#     out += -np.tensordot(f[o, o], t_1, axes=((0), (1))).transpose(1, 0)


# def add_s3c_t(u, t_1, o, v, out, np):
#     """Function adding the S3c diagram
#
#         g(f, u, t) <- u^{ak}_{ic} t^{c}_{k}
#
#     Number of FLOPS required: O(m^2 n^2)
#     """
#     out += np.tensordot(u[v, o, o, v], t_1, axes=((1, 3), (1, 0)))


def add_s4a_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the S4a diagram

        g(f, u, t) <- -0.5 * u^{kl}_{cd} t^{c}_{i} t^{ad}_{kl}

    We do this in two steps

        W^{kl}_{di} = -0.5 * u^{kl}_{cd} t^{c}_{i}
        g(f, u, t) <- t^{ad}_{kl} W^{kl}_{di}

    Number of FLOPS required: O(m^2 n^3)
    """
    W_kldi = -0.5 * np.tensordot(u[o, o, v, v], t_1, axes=((2), (0)))
    out += np.tensordot(t_2, W_kldi, axes=((1, 2, 3), (2, 0, 1)))


def add_s4b_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the S4b diagram

        g(f, u, t) <- -0.5 * u^{kl}_{cd} t^{a}_{k} t^{cd}_{il}

    We do this in two steps

        W^{k}_{i} = -0.5 u^{kl}_{cd} t^{cd}_{il}
        g(f, u, t) <- t^{a}_{k} W^{k}_{i}

    Number of FLOPS required: O(m^2 n^3)
    """

    W_ki = -0.5 * np.tensordot(u[o, o, v, v], t_2, axes=((1, 2, 3), (3, 0, 1)))
    out += np.dot(t_1, W_ki)


def add_s4c_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the S4c diagram

        g(f, u, t) <- u^{kl}_{cd} t^{c}_{k} t^{da}_{li}

    We do this in two steps

        W^{l}_{d} = u^{kl}_{cd} t^{c}_{k}
        g(f, u, t) <- W^{l}_{d} t^{da}_{li}

    Number of FLOPS required: O(m^2 n^2)
    """

    temp_ld = np.tensordot(u[o, o, v, v], t_1, axes=((0, 2), (1, 0)))
    out += np.tensordot(temp_ld, t_2, axes=((0, 1), (2, 0)))


# def add_s5a_t(f, t_1, o, v, out, np):
#     """Function for adding the S5a diagram
#
#         g(f, u, t) <- -f^{k}_{c} t^{c}_{i} t^{a}_{k}
#
#     We do this in two steps
#
#         W^{k}_{i} = -f^{k}_{c} t^{c}_{i}
#         g(f, u, t) <- t^{a}_{k} W^{k}_{i}
#
#     Number of FLOPS required: O(m n^2)
#     """
#
#     W_ki = -np.dot(f[o, v], t_1)
#     out += np.dot(t_1, W_ki)


# def add_s5b_t(u, t_1, o, v, out, np):
#     """Function for adding the S5b diagram
#
#         g(f, u, t) <- u^{ak}_{cd} t^{c}_{i} t^{d}_{k}
#
#     We do this in two steps
#
#         W^{ak}_{di} = u^{ak}_{cd} t^{c}_{i}
#         g(f, u, t) <- W^{ak}_{di} t^{d}_{k}
#
#     Number of FLOPS required: O(m^3 n^2)
#     """
#     W_akdi = np.tensordot(u[v, o, v, v], t_1, axes=((2), (0)))
#     out += np.tensordot(W_akdi, t_1, axes=((1, 2), (1, 0)))


# def add_s5c_t(u, t_1, o, v, out, np):
#     """Function for adding the S5c diagram
#
#         g(f, u, t) <- - u^{kl}_{ic} t^{a}_{k} t^{c}_{l}
#
#     We do this in two steps
#
#         W^{k}_{i} = - u^{kl}_{ic} t^{c}_{l}
#         g(f, u, t) <- t^{a}_{k} W^{k}_{i}
#
#     Number of FLOPS required: O(m n^3)
#     """
#     W_ki = -np.tensordot(u[o, o, o, v], t_1, axes=((1, 3), (1, 0)))
#     out += np.dot(t_1, W_ki)


# def add_s6_t(u, t_1, o, v, out, np):
#     """Function for adding the S6 diagram
#
#         g(f, u, t) <- (-1) * u ^{kl}_{cd} t^{c}_{i} t^{a}_{k} t^{d}_{l}
#
#     We do this in three steps
#
#         W^{k}_{c} = - u^{kl}_{cd} t^{d}_{l}
#         W^{k}_{i} = W^{k}_{c} t^{c}_{i}
#         g(f, u, t) <- t^{a}_{k} W^{k}_{i}
#
#     Number of FLOPS required: O(m^2 n^2)
#     """
#
#     W_kc = -np.tensordot(u[o, o, v, v], t_1, axes=((1, 3), (1, 0)))
#     W_ki = np.dot(W_kc, t_1)
#     out += np.dot(t_1, W_ki)


# Diagrams for T_1 contributions to CCSD T_2 equations.


def add_d4a_t(u, t_1, o, v, out, np):
    """Function for adding the D4a diagram

        g(f, u, t) <- u^{ab}_{cj} t^{c}_{i} P(ij)

    Number of FLOPS required: O(m^3 n^2)
    """

    # Get abji want abij
    term = np.tensordot(u[v, v, v, o], t_1, axes=((2), (0))).transpose(
        0, 1, 3, 2
    )
    term -= term.swapaxes(2, 3)
    out += term


def add_d4b_t(u, t_1, o, v, out, np):
    """Function for adding the D4b diagram

        g(f, u, t) <-  (-1) * u^{kb}_{ij} t^{a}_{k} P(ab)

    Number of FLOPS required: O(m^2 n^3)
    """
    term = np.tensordot(t_1, u[o, v, o, o], axes=((1), (0)))
    term -= term.swapaxes(0, 1)
    out -= term


def add_d5a_t(f, t_1, t_2, o, v, out, np):
    """Function for adding the D5a diagram

        g(f, u, t) <- (-1) * f^{k}_{c} t^{c}_{i} t^{ab}_{kj} P(ij)

    We do this in two steps

        W^{k}_{i} = f^{k}_{c} t^{c}_{i}
        g(f, u, t) <- (-1) * t^{ab}_{kj} W^{k}_{i} P(ij)

    Number of FLOPS required: O(m^2 n^3)
    """

    term_ki = np.dot(f[o, v], t_1)
    # Note that we add an extra minus sign thus avoiding the transpose in the
    # two last axes
    term = -np.tensordot(t_2, term_ki, axes=((2), (0)))
    term -= term.swapaxes(2, 3)
    out -= term


def add_d5b_t(f, t_1, t_2, o, v, out, np):
    """Function for adding the D5b diagram

        g(f, u, t) <- (-1) * f^{k}_{c} t^{a}_{k} t^{cb}_{ij} P(ab)

    We do this in two steps

        W^{a}_{c} = t^{a}_{k} f^{k}_{c}
        g(f, u, t) <- (-1) W^{a}_{c} t^{cb}_{ij} P(ab)

    Number of FLOPS required: O(m^3 n^2)
    """

    term_ac = np.dot(t_1, f[o, v])
    term = np.tensordot(term_ac, t_2, axes=((1), (0)))
    term -= term.swapaxes(0, 1)

    out -= term


def add_d5c_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D5c diagram

        g(f, u, t) <- u^{ak}_{cd} t^{c}_{i} t^{db}_{kj} P(ab) P(ij)

    We do this in two steps

        W^{ak}_{di} = u^{ak}_{cd} t^{c}_{i}
        g(f, u, t) <- W^{ak}_{di} t^{db}_{kj} P(ab) P(ij)

    Number of FLOPS required: O(m^3 n^3)
    """

    term = np.tensordot(u[v, o, v, v], t_1, axes=((2), (0)))  # akdi
    # Get aibj want abij
    term = np.tensordot(term, t_2, axes=((1, 2), (2, 0))).transpose(0, 2, 1, 3)
    term -= term.swapaxes(0, 1)
    term -= term.swapaxes(2, 3)

    out += term


def add_d5d_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D5d diagram

        g(f, u, t) <- (-1) * u^{kl}_{ic} t^{a}_{k} t^{cb}_{lj} P(ab) P(ij)

    We do this in two steps

        W^{al}_{ic} = t^{a}_{k} u^{kl}_{ic}
        g(f, u, t) <- (-1) * W^{al}_{ic} t^{cb}_{lj} P(ab) P(ij)

    Number of FLOPS required: O(m^3 n^3)
    """

    term_alic = np.tensordot(t_1, u[o, o, o, v], axes=((1), (0)))
    term = -1 * np.tensordot(term_alic, t_2, axes=((1, 3), (2, 0))).transpose(
        0, 2, 1, 3
    )
    term -= term.swapaxes(0, 1)
    term -= term.swapaxes(2, 3)

    out += term


def add_d5e_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D5e diagram

        g(f, u, t) <- (-0.5) * u^{kb}_{cd} t^{a}_{k} t^{cd}_{ij} P(ab)

    We do this in two steps

        W^{kb}_{ij} = u^{kb}_{cd} t^{cd}_{ij}
        g(f, u, t) <- (-0.5) * t^{a}_{k} W^{kb}_{ij} P(ab)

    Number of FLOPS required: O(m^3 n^3)
    """

    term_kbij = np.tensordot(u[o, v, v, v], t_2, axes=((2, 3), (0, 1)))
    term = -0.5 * np.tensordot(t_1, term_kbij, axes=((1), (0)))
    term -= term.swapaxes(0, 1)

    out += term


def add_d5f_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D5f diagram

        g(f, u, t) <- (0.5) * u^{kl}_{cj} t^{c}_{i} t^{ab}_{kl} P(ij)

    We do this in two steps

        W^{kl}_{ji} = u^{kl}_{cj} t^{c}_{i}
        g(f, u, t) <- 0.5 * t^{ab}_{kl} W^{kl}_{ji} P(ij)

    Number of FLOPS required O(m^2, n^4)
    """

    term_klji = np.tensordot(u[o, o, v, o], t_1, axes=((2), (0)))
    # Note that we add an extra minus sign thus avoiding the need for a
    # transpose in the two last axes of term_klji
    term = -0.5 * np.tensordot(t_2, term_klji, axes=((2, 3), (0, 1)))
    term -= term.swapaxes(2, 3)

    out += term


def add_d5g_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D5g diagram

        g(f, u, t) <- u^{ka}_{cd} t^{c}_{k} t^{db}_{ij} P(ab)

    We do this in two steps

        W^{a}_{d} = u^{ka}_{cd} t^{c}_{k}
        g(f, u, t) <- W^{a}_{d} t^{db}_{ij} P(ab)

    Number of FLOPS required: O(m^3, n^2)
    """

    term_ad = np.tensordot(u[o, v, v, v], t_1, axes=((0, 2), (1, 0)))
    term = np.tensordot(term_ad, t_2, axes=((1), (0)))

    term -= term.swapaxes(0, 1)

    out += term


def add_d5h_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D5h diagram

        g(f, u, t) <- (-1) * u^{kl}_{ci} t^{c}_{k} t^{ab}_{lj} P(ij)

    We do this in two steps

        W^{l}_{i} = u^{kl}_{ci} t^{c}_{k}
        g(f, u, t) <- (-1) * t^{ab}_{lj} W^{l}_{i} P(ij)

    Number of FLOPS required O(m^2, n^3)
    """

    term_li = np.tensordot(u[o, o, v, o], t_1, axes=((0, 2), (1, 0)))
    # Note that we include a sign change to avoid transposing the two last axes
    term = np.tensordot(t_2, term_li, axes=((2), (0)))
    term -= term.swapaxes(2, 3)

    out += term


def add_d6a_t(u, t_1, o, v, out, np):
    """Function for adding the D6a diagram

        g(f, u, t) <- u^{ab}_{cd} t^{c}_{i} t^{d}_{j}

    We do this in two steps

        W^{ab}_{di} = u^{ab}_{cd} t^{c}_{i}
        g(f, u, t) <- W^{ab}_{di} t^{d}_{j}

    Number of FLOPS required O(m^4 n)
    """

    term = np.tensordot(u[v, v, v, v], t_1, axes=((2), (0)))  # abdi
    term = np.tensordot(term, t_1, axes=((2), (0)))  # abij

    out += term


def add_d6b_t(u, t_1, o, v, out, np):
    """Function for adding the D6b diagram

        g(f, u, t) <- u^{kl}_{ij} t^{a}_{k} t^{b}_{l}

    We do this in two steps

        W^{bk}_{ij} = t^{b}_{l} u^{kl}_{ij}
        g(f, u, t) <- t^{a}_{k} W^{bk}_{ij}

    Number of FLOPS required O(m^2, n^3)
    """

    term_bkij = np.tensordot(t_1, u[o, o, o, o], axes=((1), (1)))
    term = np.tensordot(t_1, term_bkij, axes=((1), (1)))

    out += term


def add_d6c_t(u, t_1, o, v, out, np):
    """Function for adding the D6c diagram

        g(f, u, t) <- (-1) * u^{kb}_{cj} t^{c}_{i} t^{a}_{k} P(ab) P(ij)

    We do this in two steps

        W^{kb}_{ji} = (-1) * u^{kb}_{cj} t^{c}_{i}
        g(f, u, t) <- t^{a}_{k} W^{kb}_{ji} P(ab) P(ij)

    Number of FLOPS required O(m^2 n^3)
    """

    # Note that we remove the sign change to avoid transposing the last two axes
    term_kbji = np.tensordot(u[o, v, v, o], t_1, axes=((2), (0)))  # kbji
    term = np.tensordot(t_1, term_kbji, axes=((1), (0)))
    term -= term.swapaxes(0, 1)
    term -= term.swapaxes(2, 3)

    out += term


def add_d7a_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D7a diagram

        g(f, u, t) <- (0.5) * u^{kl}_{cd} t^{c}_{i} t^{ab}_{kl} t^{d}_{j}

    We do this in three steps

        W^{kl}_{di} = 0.5 * u^{kl}_{cd} t^{c}_{i}
        W^{kl}_{ij} = W^{kl}_{di} t^{d}_{j}
        g(f, u, t) <- t^{ab}_{kl} W^{kl}_{ij}

    Number of FLOPS required O(m^2 n^4)
    """

    term_kldi = (0.5) * np.tensordot(
        u[o, o, v, v], t_1, axes=((2), (0))
    )  # kldi
    term_klij = np.tensordot(term_kldi, t_1, axes=((2), (0)))
    term = np.tensordot(t_2, term_klij, axes=((2, 3), (0, 1)))

    out += term


def add_d7b_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D7b diagram

        g(f, u , t) <- (0.5) * u^{kl}_{cd} t^{a}_{k} t^{cd}_{ij} t^{b}_{l}

    We do this in three steps

        W^{kl}_{ij} = 0.5 * u^{kl}_{cd} t^{cd}_{ij}
        W^{bk}_{ij} = t^{b}_{l} W^{kl}_{ij}
        g(f, u, t) <- t^{a}_{k} W^{bk}_{ij}

    Number of FLOPS required O(m^2 n^4)
    """

    # klij
    term = 0.5 * np.tensordot(u[o, o, v, v], t_2, axes=((2, 3), (0, 1)))
    # bkij
    term = np.tensordot(t_1, term, axes=((1), (1)))
    term = np.tensordot(t_1, term, axes=((1), (1)))

    out += term


def add_d7c_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D7c diagram

        g(f, u, t) <- (-1) u^{kl}_{cd} t^{c}_{i} t^{a}_{k} t^{db}_{lj} P(ij) P(ab)

    We do this in three steps

        W^{al}_{cd} = (-1) t^{a}_{k} u^{kl}_{cd}
        W^{al}_{di} = W^{al}_{cd} t^{c}_{i}
        g(f, u, t) <- W^{al}_{di} t^{db}_{lj} P(ij) P(ab)

    Number of FLOPS required O(m^3 n^3)
    """

    # alcd
    term = -np.tensordot(t_1, u[o, o, v, v], axes=((1), (0)))
    # aldi
    term = np.tensordot(term, t_1, axes=((2), (0)))
    term = np.tensordot(term, t_2, axes=((1, 2), (2, 0))).transpose(0, 2, 1, 3)

    term -= term.swapaxes(0, 1)
    term -= term.swapaxes(2, 3)

    out += term


def add_d7d_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D7d diagram

        g(f, u, t) <- (-1) * u^{kl}_{cd} t^{c}_{k} t^{d}_{i} t^{ab}_{lj} P(ij)

    We do this in three steps

        W^{l}_{d} = (-1) * u^{kl}_{cd} t^{c}_{k}
        W^{l}_{i} = W^{l}_{d} t^{d}_{i}
        g(f, u, t) <- t^{ab}_{lj} W^{l}_{i} P(ij)

    Number of FLOPS required O(m^2 n^3)
    """

    # ld
    term = (-1) * np.tensordot(u[o, o, v, v], t_1, axes=((0, 2), (1, 0)))
    # li
    term = np.tensordot(term, t_1, axes=((1), (0)))
    # Note that we include a sign change to avoid transposing the last two axes
    term = -np.tensordot(t_2, term, axes=((2), (0)))
    term -= term.swapaxes(2, 3)

    out += term


def add_d7e_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D7e diagram

        g(f, u, t) <- (-1) * u^{kl}_{cd} t^{c}_{k} t^{a}_{l} t^{db}_{ij} P(ab)

    We do this in three steps

        W^{l}_{d} = (-1) * u^{kl}_{cd} t^{c}_{k}
        W^{a}_{d} = t^{a}_{l} W^{l}_{d}
        g(f, u, t) <- W^{a}_{d} t^{db}_{ij} P(ab)

    Number of FLOPS required O(m^3 n^2)
    """

    term = (-1) * np.tensordot(u[o, o, v, v], t_1, axes=((0, 2), (1, 0)))  # ld
    term = np.tensordot(t_1, term, axes=((1), (0)))  # ad
    term = np.tensordot(term, t_2, axes=((1), (0)))  # abij
    term -= term.swapaxes(0, 1)

    out += term


def add_d8a_t(u, t_1, o, v, out, np):
    """Function for adding the D8a diagram

        g(f, u, t) <- (-1) * u^{kb}_{cd} t^{c}_{i} t^{a}_{k} t^{d}_{j} P(ab)

    We do this in three steps

        W^{kb}_{di} = (-1) * u^{kb}_{cd} t^{c}_{i}
        W^{kb}_{ij} = W^{kb}_{di} t^{d}_{j}
        g(f, u, t) <- t^{a}_{k} W^{kb}_{ij} P(ab)

    Number of FLOPS required O(m^3 n^2)

    Note: The minus sign in this expression is in disagreement with the
    definition in Shavitt & Bartlett page 306. We believe this is an error in
    Shavitt & Bartlett as SymPy and Gauss & Stanton's article are in agreement.
    """

    term = np.tensordot(u[o, v, v, v], t_1, axes=((2), (0)))  # kbdi
    term = np.tensordot(term, t_1, axes=((2), (0)))  # kbij
    term = np.tensordot(t_1, term, axes=((1), (0)))
    term -= term.swapaxes(0, 1)

    out -= term


def add_d8b_t(u, t_1, o, v, out, np):
    """Function for adding the D8b diagram

        g(f, u, t) <- u^{kl}_{cj} t^{c}_{i} t^{a}_{k} t^{b}_{l} P(ij)

    We do this in three steps

        W^{kl}_{ji} = u^{kl}_{cj} t^{c}_{i}
        W^{bk}_{ji} = t^{b}_{l} W^{kl}_{ji}
        g(f, u, t) <- t^{a}_{k} W^{bk}_{ji} P(ij)

    Number of FLOPS required O(m^2 n^3)
    """

    term = np.tensordot(u[o, o, v, o], t_1, axes=((2), (0)))  # klji
    term = np.tensordot(t_1, term, axes=((1), (1)))  # bkji
    # Note that the sign change removes the need for transposing the two last
    # axes
    term = -np.tensordot(t_1, term, axes=((1), (1)))
    term -= term.swapaxes(2, 3)

    out += term


def add_d9_t(u, t_1, o, v, out, np):
    """Function for adding the D9 diagram

        g(f, u, t) <- u^{kl}_{cd} t^{c}_{i} t^{d}_{j} t^{a}_{k} t^{b}_{l}

    We do this in four steps

        W^{kl}_{di} = u^{kl}_{cd} t^{c}_{i}
        W^{kl}_{ij} = W^{kl}_{di} t^{d}_{j}
        W^{bk}_{ij} = t^{b}_{l} W^{kl}_{ij}
        g(f, u, t) <- t^{a}_{k} W^{bk}_{ij}

    Number of FLOPS required O(m^2 n^3)
    """

    term = np.tensordot(u[o, o, v, v], t_1, axes=((2), (0)))  # kldi
    term = np.tensordot(term, t_1, axes=((2), (0)))  # klij
    term = np.tensordot(t_1, term, axes=((1), (1)))  # bkij
    term = np.tensordot(t_1, term, axes=((1), (1)))

    out += term
