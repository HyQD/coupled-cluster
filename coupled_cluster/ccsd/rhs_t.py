# Labelling of the different terms comes from the book "Many-Body Methods in
# Chemistry and Physics" by I. Shavitt and R. J. Bartlett.


# Diagrams for CCSD T_1 amplitude equations

import coupled_cluster.ccd.rhs_t as ccd_t


def compute_t_1_amplitudes(f, u, t_1, t_2, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(t_1)

    add_s1_t(f, o, v, out, np=np)
    add_s2a_t(f, t_2, o, v, out, np=np)
    add_s2b_t(u, t_2, o, v, out, np=np)
    add_s2c_t(u, t_2, o, v, out, np=np)
    add_s3a_t(f, t_1, o, v, out, np=np)
    add_s3b_t(f, t_1, o, v, out, np=np)
    add_s3c_t(u, t_1, o, v, out, np=np)
    add_s4a_t(u, t_1, t_2, o, v, out, np=np)
    add_s4b_t(u, t_1, t_2, o, v, out, np=np)
    add_s4c_t(u, t_1, t_2, o, v, out, np=np)
    add_s5a_t(f, t_1, o, v, out, np=np)
    add_s5b_t(u, t_1, o, v, out, np=np)
    add_s5c_t(u, t_1, o, v, out, np=np)
    add_s6_t(u, t_1, o, v, out, np=np)

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


def add_s1_t(f, o, v, out, np):
    """Function adding the S1 diagram

        g(f, u, t) <- f^{a}_{i}

    Number of FLOPS required: O(m n).
    """
    out += f[v, o]


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


def add_s3a_t(f, t_1, o, v, out, np):
    """Function adding the S3a diagram

        g(f, u, t) <- f^{a}_{c} t^{c}_{i}

    Number of FLOPS required: O(m^2 n)
    """
    out += np.tensordot(f[v, v], t_1, axes=((1), (0)))


def add_s3b_t(f, t_1, o, v, out, np):
    """Function adding the S3b diagram

        g(f, u, t) <- -f^{k}_{i} t^{a}_{k}

    Number of FLOPS required: O(m, n^2)
    """
    out += -np.tensordot(f[o, o], t_1, axes=((0), (1))).transpose(1, 0)


def add_s3c_t(u, t_1, o, v, out, np):
    """Function adding the S3c diagram

        g(f, u, t) <- u^{ak}_{ic} t^{c}_{k}

    Number of FLOPS required: O(m^2, n^2)
    """
    out += np.tensordot(u[v, o, o, v], t_1, axes=((1, 3), (1, 0)))


def add_s4a_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the S4a diagram

        g(f, u, t) <- -0.5 * u^{kl}_{cd} t^{c}_{i} t^{ad}_{kl}

    Number of FLOPS required: O(m^3 n^3)
    """
    W_kldi = -0.5 * np.tensordot(u[o, o, v, v], t_1, axes=((2), (0)))
    out += np.tensordot(W_kldi, t_2, axes=((0, 1, 2), (2, 3, 1))).swapaxes(0, 1)


def add_s4b_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the S4b diagram

        g(f, u, t) <- -0.5 * u^{kl}_{cd} t^{a}_{k} t^{cd}_{il}

    Number of FLOPS required: O(m^3 n^3)
    """
    W_lcda = -0.5 * np.tensordot(u[o, o, v, v], t_1, axes=((0), (1)))
    out += np.tensordot(W_lcda, t_2, axes=((1, 2, 0), (0, 1, 3)))


def add_s4c_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the S4c diagram

        g(f, u, t) <- u^{kl}_{cd} t^{c}_{k} t^{da}_{li}

    Number of FLOPS required: O(m^3 n^3)
    """
    temp_ld = np.tensordot(u[o, o, v, v], t_1, axes=((0, 2), (1, 0)))
    out += np.tensordot(temp_ld, t_2, axes=((0, 1), (2, 0)))


def add_s5a_t(f, t_1, o, v, out, np):
    """Function for adding the S5a diagram

        g(f, u, t) <- f^{k}_{c} t^{c}_{i} t^{a}_{k}

    Number of FLOPS required: O(m^2, n^2)
    """
    temp_ki = -np.tensordot(f[o, v], t_1, axes=((1), (0)))
    out += np.tensordot(temp_ki, t_1, axes=((0), (1))).swapaxes(0, 1)


def add_s5b_t(u, t_1, o, v, out, np):
    """Function for adding the S5b diagram

        g(f, u, t) <- u^{ak}_{cd} t^{c}_{i} t^{d}_{k}

    Number of FLOPS required: O(m^2 n^3)
    """
    W_akdi = np.tensordot(u[v, o, v, v], t_1, axes=((2), (0)))
    out += np.tensordot(W_akdi, t_1, axes=((1, 2), (1, 0)))


def add_s5c_t(u, t_1, o, v, out, np):
    """Function for adding the S5c diagram

        g(f, u, t) <- - u^{kl}_{ic} t^{a}_{k} t^{c}_{l}

    Number of FLOPS required: O(m^2, n^3)
    """
    W_lica = -np.tensordot(u[o, o, o, v], t_1, axes=((0), (1)))
    out += np.tensordot(W_lica, t_1, axes=((0, 2), (1, 0))).swapaxes(0, 1)


def add_s6_t(u, t_1, o, v, out, np):
    """Function for adding the S6 diagram

        g(f, u, t) <- (-1) * u ^{kl}_{cd} t^{c}_{i} t^{a}_{k} t^{d}_{l}

    Number of FLOPS required: O(m^3 n^3)
    """
    W_kldi = -np.tensordot(u[o, o, v, v], t_1, axes=((2), (0)))
    W_ldia = np.tensordot(W_kldi, t_1, axes=((0), (1)))
    out += np.tensordot(W_ldia, t_1, axes=((0, 1), (1, 0))).swapaxes(0, 1)


# Diagrams for T_1 contributions to CCSD T_2 equations.


def add_d4a_t(u, t_1, o, v, out, np):
    """Function for adding the D4a diagram

        g(f, u, t) <- u^{ab}_{cj} t^{c}_{i} P(ij)

    Number of FLOPS required: O(m^3, n^2)
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
    term = np.tensordot(u[o, v, o, o], t_1, axes=((0), (1))).transpose(
        3, 0, 1, 2
    )
    term -= term.swapaxes(0, 1)
    out -= term


def add_d5a_t(f, t_1, t_2, o, v, out, np):
    """Function for adding the D5a diagram

        g(f, u, t) <- (-1) * f^{k}_{c} t^{c}_{i} t^{ab}_{kj} P(ij)

    Number of FLOPS required: O(m^3, n^3)
    """

    term_ki = np.tensordot(f[o, v], t_1, axes=((1), (0)))
    # Get iabj want abij
    term = np.tensordot(term_ki, t_2, axes=((0), (2))).transpose(1, 2, 0, 3)
    term -= term.swapaxes(2, 3)
    out -= term


def add_d5b_t(f, t_1, t_2, o, v, out, np):
    """Function for adding the D5b diagram

        g(f, u, t) <- (-1) * f^{k}_{c} t^{a}_{k} t^{cb}_{ij} P(ab)
    
    Number of FLOPS required: O(m^3, n^3)
    """

    term = np.tensordot(f[o, v], t_1, axes=((0), (1))) # ca
    # Get abij
    term = np.tensordot(term, t_2, axes=((0), (0)))
    term -= term.swapaxes(0, 1)

    out -= term


def add_d5c_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D5c diagram

        g(f, u, t) <- u^{ak}_{cd} t^{c}_{i} t^{db}_{kj} P(ab) P(ij)

    Number of FLOPS required: O(m^4, n^3) wrong
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

    Number of FLOPS required: O(m^3 n^4)
    """

    term = (-1) * np.tensordot(u[o, o, o, v], t_1, axes=((0), (1)))  # lica
    # Get iabj want abij
    term = np.tensordot(term, t_2, axes=((0, 2), (2, 0))).transpose(1, 2, 0, 3)
    term -= term.swapaxes(0, 1)
    term -= term.swapaxes(2, 3)

    out += term


def add_d5e_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D5e diagram

        g(f, u, t) <- (-0.5) * u^{kb}_{cd} t^{a}_{k} t^{cd}_{ij} P(ab)

    Number of FLOPS required: O(m^4, n^3)
    """

    term = (-0.5) * np.tensordot(u[o, v, v, v], t_1, axes=((0), (1)))  # bcda
    # Get baij want abij
    term = np.tensordot(term, t_2, axes=((1, 2), (0, 1))).transpose(1, 0, 2, 3)
    term -= term.swapaxes(0, 1)

    out += term


def add_d5f_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D5f diagram

        g(f, u, t) <- (0.5) * u^{kl}_{cj} t^{c}_{i} t^{ab}_{kl} P(ij)

    Number of FLOPS required O(m^3, n^4)
    """

    term = (0.5) * np.tensordot(u[o, o, v, o], t_1, axes=((2), (0)))  # klji
    # Get jiab want abij
    term = np.tensordot(term, t_2, axes=((0, 1), (2, 3))).transpose(2, 3, 1, 0)
    term -= term.swapaxes(2, 3)

    out += term


def add_d5g_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D5g diagram

        g(f, u, t) <- u^{ka}_{cd} t^{c}_{k} t^{db}_{ij} P(ab)
    
    Number of FLOPS required: O(m^4, n^3)
    """

    term = np.tensordot(u[o, v, v, v], t_1, axes=((0, 2), (1, 0)))  # ad
    term = np.tensordot(term, t_2, axes=((1), (0)))
    term -= term.swapaxes(0, 1)

    out += term


def add_d5h_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D5h diagram

        g(f, u, t) <- (-1) * u^{kl}_{ci} t^{c}_{k} t^{ab}_{lj}

    Number of FLOPS required O(m^3, n^4)
    """

    term = (-1) * np.tensordot(u[o, o, v, o], t_1, axes=((0, 2), (1, 0)))  # li
    # Get iabj want abij
    term = np.tensordot(term, t_2, axes=((0), (2))).transpose(1, 2, 0, 3)
    term -= term.swapaxes(2, 3)

    out += term

def add_d6a_t(u, t_1, o, v, out, np):
    """Function for adding the D6a diagram

        g(f, u, t) <- u^{ab}_{cd} t^{c}_{i} t^{d}_{j}

    Number of FLOPS required O()
    """

    term = np.tensordot(u[v, v, v, v], t_1, axes=((2), (0))) # abdi
    term = np.tensordot(term, t_1, axes=((2), (0))) # abij

    out += term

def add_d6b_t(u, t_1, o, v, out, np):
    """Function for adding the D6b diagram

        g(f, u, t) <- u^{kl}_{ij} t^{a}_{k} t^{b}_{l}

    Number of FLOPS required O(m, n)
    """

    term = np.tensordot(u[o, o, o, o], t_1, axes=((0), (1))) # lija
    term = np.tensordot(term, t_1, axes=((0), (1))).transpose(2, 3, 0, 1) # ijab -> abij
    
    out += term

def add_d6c_t(u, t_1, o, v, out, np):
    """Function for adding the D6c diagram

        g(f, u, t) <- (-1) * u^{kb}_{cj} t^{c}_{i} t^{a}_{k} P(ab) P(ij)

    Number of FLOPS required O(m, n)
    """

    term = (-1) * np.tensordot(u[o, v, v, o], t_1, axes=((2), (0))) # kbji
    term = np.tensordot(term, t_1, axes=((0), (1))).transpose(3, 0, 2, 1) # bjia -> abij
    term -= term.swapaxes(0, 1)
    term -= term.swapaxes(2, 3)

    out += term

def add_d7a_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D7a diagram

        g(f, u, t) <- (0.5) * u^{kl}_{cd} t^{c}_{i} t^{ab}_{kl} t^{d}_{j}

    Number of FLOPS required O()
    """

    term = (0.5) * np.tensordot(u[o, o, v, v], t_1, axes=((2), (0))) # kldi
    term = np.tensordot(term, t_2, axes=((0, 1), (2, 3))) # diab
    term = np.tensordot(term, t_1, axes=((0), (0))).transpose(1, 2, 0, 3) # iabj -> abij

    out += term

def add_d7b_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D7b diagram

        g(f, u , t) <- (0.5) * u^{kl}_{cd} t^{a}_{k} t^{cd}_{ij} t^{b}_{l}

    Number of FLOPS required O()
    """
    
    term = (0.5) * np.tensordot(u[o, o, v, v], t_1, axes=((0), (1))) # lcda
    term = np.tensordot(term, t_2, axes=((1, 2), (0, 1))) # laij
    term = np.tensordot(term, t_1, axes=((0), (1))).transpose(0, 3, 1, 2) # aijb -> abij

    out += term

def add_d7c_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D7c diagram

        g(f, u, t) <- (-1) u^{kl}_{cd} t^{c}_{i} t^{a}_{k} t^{db}_{lj} P(ij) P(ab)

    Number of FLOPS required O()
    """

    term = (-1) * np.tensordot(u[o, o, v, v], t_1, axes=((2), (0))) # kldi
    term = np.tensordot(term, t_1, axes=((0), (1))) # ldia
    term = np.tensordot(term, t_2, axes=((0, 1), (2, 0))).transpose(1, 2, 0, 3) # iabj -> abij
    term -= term.swapaxes(0, 1)
    term -= term.swapaxes(2, 3)

    out += term

def add_d7d_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D7d diagram

        g(f, u, t) <- (-1) * u^{kl}_{cd} t^{c}_{k} t^{d}_{i} t^{ab}_{lj} P(ij)

    Number of FLOPS required O()
    """

    term = (-1) * np.tensordot(u[o, o, v, v], t_1, axes=((0, 2), (1, 0))) # ld 
    term = np.tensordot(term, t_1, axes=((1), (0))) # li
    term = np.tensordot(term, t_2, axes=((0), (2))).transpose(1, 2, 0, 3) # iabj -> abij
    term -= term.swapaxes(2, 3)

    out += term

def add_d7e_t(u, t_1, t_2, o, v, out, np):
    """Function for adding the D7e diagram

        g(f, u, t) <- (-1) * u^{kl}_{cd} t^{c}_{k} t^{a}_{l} t^{db}_{ij} P(ab)

    Number of FLOPS required O()
    """

    term = (-1) * np.tensordot(u[o, o, v, v], t_1, axes=((0, 2), (1, 0))) # ld
    term = np.tensordot(term, t_1, axes=((0), (1))) # da
    term = np.tensordot(term, t_2, axes=((0), (0))) # abij
    term -= term.swapaxes(0, 1)

    out += term

def add_d8a_t(u, t_1, o, v, out, np):
    """Function for adding the D8a diagram

        g(f, u, t) <- u^{kb}_{cd} t^{c}_{i} t^{a}_{k} t^{d}_{j} P(ab)
    
    Number of FLOPS required O()
    """

    term = np.tensordot(u[o, v, v, v], t_1, axes=((2), (0))) # kbdi
    term = np.tensordot(term, t_1, axes=((0), (1))) # bdia
    term = np.tensordot(term, t_1, axes=((1), (0))).transpose(2, 0, 1, 3) # biaj -> abij
    term -= term.swapaxes(0, 1)

    out += term

def add_d8b_t(u, t_1, o, v, out, np):
    """Function for adding the D8b diagram

        g(f, u, t) <- u^{kl}_{cj} t^{c}_{i} t^{a}_{k} t^{b}_{l}

    Number of FLOPS required O()
    """

    term = np.tensordot(u[o, o, v, o], t_1, axes=((2), (0))) # klji
    term = np.tensordot(term, t_1, axes=((0), (1))) # ljia
    term = np.tensordot(term, t_1, axes=((0), (1))).transpose(2, 3, 1, 0) # jiab -> abij
    term -= term.swapaxes(2, 3)

    out += term

def add_d9_t(u, t_1, o, v, out, np):
    """Function for adding the D9 diagram

        g(f, u, t) <- u^{kl}_{cd} t^{c}_{i} t^{d}_{j} t^{a}_{k} t^{b}_{l}

    Number of FLOPS required O()
    """

    term = np.tensordot(u[o, o, v, v], t_1, axes=((2), (0))) # kldi
    term = np.tensordot(term, t_1, axes=((2), (0))) # klij
    term = np.tensordot(term, t_1, axes=((0), (1))) # lija
    term = np.tensordot(term, t_1, axes=((0), (1))).transpose(2, 3, 0, 1) # ijab -> abij
    
    out += term
