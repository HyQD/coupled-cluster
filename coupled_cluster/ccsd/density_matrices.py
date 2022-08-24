from opt_einsum import contract


def compute_one_body_density_matrix(t_1, t_2, l_1, l_2, o, v, np, out=None):
    if out is None:
        out = np.zeros((v.stop, v.stop), dtype=t_1.dtype)

    out.fill(0)

    add_rho_ba(t_1, t_2, l_1, l_2, o, v, out, np)
    add_rho_ia(l_1, o, v, out, np)
    add_rho_ai(t_1, t_2, l_1, l_2, o, v, out, np)
    add_rho_ji(t_1, t_2, l_1, l_2, o, v, out, np)

    return out


def compute_two_body_density_matrix(t_1, t_2, l_1, l_2, o, v, np, out=None):
    """Two body density matrices from Kvaal (2012).

    The final two body density matrix should satisfy

        contract('pqpq->', rho_qspr) = N(N-1)

    where N is the number of electrons.
    """
    if out is None:
        out = np.zeros((v.stop, v.stop, v.stop, v.stop), dtype=t_1.dtype)

    out.fill(0)

    add_rho_klij(t_1, t_2, l_1, l_2, o, v, out, np)

    add_rho_jkia(t_1, t_2, l_1, l_2, o, v, out, np)
    add_rho_akij(t_1, t_2, l_1, l_2, o, v, out, np)

    add_rho_ijab(t_1, t_2, l_1, l_2, o, v, out, np)
    add_rho_jbia(t_1, t_2, l_1, l_2, o, v, out, np)
    add_rho_abij(t_1, t_2, l_1, l_2, o, v, out, np)

    add_rho_ciab(t_1, t_2, l_1, l_2, o, v, out, np)
    add_rho_bcai(t_1, t_2, l_1, l_2, o, v, out, np)

    add_rho_cdab(t_1, t_2, l_1, l_2, o, v, out, np)
    return out


def add_rho_klij(t_1, t_2, l_1, l_2, o, v, out, np):
    """
    Function adding the o-o-o-o part of the two-body density matrix
    rho^{kl}_{ij} = delta_{i k} delta_{j l} P(ij)
    - delta_{j l} l^{k}_{e} t^{e}_{i} P(ij) P(kl)-0.5*delta_{j l} l^{km}_{ef} t^{ef}_{im} P(ij) P(kl)}
    + l^{kl}_{ef} t^{e}_{i} t^{f}_{j}
    + 0.5*l^{kl}_{ef} t^{ef}_{ij}
    """

    delta = np.eye(o.stop)

    Pij = contract("ki, lj -> klij", delta, delta)
    out[o, o, o, o] += Pij
    out[o, o, o, o] -= Pij.swapaxes(2, 3)

    Pijkl = -contract("lj, ke, ei->klij", delta, l_1, t_1) - 0.5 * contract(
        "lj, kmef, efim->ijkl", delta, l_2, t_2
    )
    out[o, o, o, o] += Pijkl
    out[o, o, o, o] -= Pijkl.swapaxes(0, 1)
    out[o, o, o, o] -= Pijkl.swapaxes(2, 3)
    out[o, o, o, o] += Pijkl.swapaxes(0, 1).swapaxes(2, 3)

    out[o, o, o, o] += contract("klef, ei, fj->klij", l_2, t_1, t_1)
    out[o, o, o, o] += 0.5 * contract("klef, efij->klij", l_2, t_2)


def add_rho_jkia(t_1, t_2, l_1, l_2, o, v, out, np):
    """
    rho^{jk}_{ia} = - delta_{i k} l^{j}_{a} P(jk) + l^{jk}_{ae} t^{e}_{i}
    """

    delta = np.eye(o.stop)
    Pjk = -contract("ki, ja->jkia", delta, l_1)
    rho_jkia = Pjk
    rho_jkia -= Pjk.swapaxes(0, 1)
    rho_jkia += contract("jkae, ei->jkia", l_2, t_1)

    out[o, o, o, v] += rho_jkia
    out[o, o, v, o] -= rho_jkia.swapaxes(2, 3)


def add_rho_akij(t_1, t_2, l_1, l_2, o, v, out, np):
    """
    rho^{ak}_{ij} =
    - delta_{j k} l^{m}_{e} t^{e}_{i} t^{a}_{m} P(ij) + delta_{j k} l^{m}_{e} t^{ae}_{im} P(ij)
    - 0.5*delta_{j k} l^{mn}_{ef} t^{e}_{i} t^{af}_{mn} P(ij) + 0.5*delta_{j k} l^{mn}_{ef} t^{a}_{n} t^{ef}_{im} P(ij)
    + delta_{j k} t^{a}_{i} P(ij) - l^{k}_{e} t^{e}_{j} t^{a}_{i} P(ij)
    - 0.5*l^{km}_{ef} t^{a}_{i} t^{ef}_{jm} P(ij) + l^{km}_{ef} t^{e}_{i} t^{af}_{jm} P(ij)
    -l^{k}_{e} t^{ae}_{ij} - l^{km}_{ef} t^{e}_{i} t^{f}_{j} t^{a}_{m}  - 0.5*l^{km}_{ef} t^{a}_{m} t^{ef}_{ij}
    """

    delta = np.eye(o.stop)

    Pij = -contract("kj, me, ei, am->akij", delta, l_1, t_1, t_1)
    Pij += contract("kj, me, aeim->akij", delta, l_1, t_2)
    Pij -= 0.5 * contract("kj, mnef, ei, afmn->akij", delta, l_2, t_1, t_2)
    Pij += 0.5 * contract("kj, mnef, an, efim->akij", delta, l_2, t_1, t_2)
    Pij += contract("kj, ai->akij", delta, t_1)
    Pij -= contract("ke, ej, ai->akij", l_1, t_1, t_1)
    Pij -= 0.5 * contract("kmef, ai, efjm->akij", l_2, t_1, t_2)
    Pij += contract("kmef, ei, afjm->akij", l_2, t_1, t_2)
    rho_akij = Pij
    rho_akij -= Pij.swapaxes(2, 3)

    rho_akij -= contract("ke, aeij->akij", l_1, t_2)
    rho_akij -= contract("kmef, ei, fj, am->akij", l_2, t_1, t_1, t_1)
    rho_akij -= 0.5 * contract("kmef, am, efij->akij", l_2, t_1, t_2)

    out[v, o, o, o] += rho_akij
    out[o, v, o, o] -= rho_akij.swapaxes(0, 1)


def add_rho_abij(t_1, t_2, l_1, l_2, o, v, out, np):
    """
    rho^{ab}_{ij} =

    l^{m}_{e} t^{e}_{i} t^{ab}_{jm} P(ij)
    + 0.5*l^{mn}_{ef} t^{ef}_{jm} t^{ab}_{in} P(ij)

    - l^{m}_{e} t^{e}_{j} t^{a}_{i} t^{b}_{m} P(ab) P(ij) + l^{m}_{e} t^{a}_{i} t^{be}_{jm} P(ab) P(ij)
    - l^{mn}_{ef} t^{e}_{i} t^{a}_{m} t^{bf}_{jn} P(ab) P(ij) - 0.5*l^{mn}_{ef} t^{e}_{j} t^{a}_{i} t^{bf}_{mn} P(ab) P(ij)

    + l^{m}_{e} t^{a}_{m} t^{be}_{ij} P(ab) + 0.5*l^{mn}_{ef} t^{a}_{n} t^{b}_{j} t^{ef}_{im} P(ab)
    + 0.5*l^{mn}_{ef} t^{a}_{i} t^{b}_{n} t^{ef}_{jm} P(ab) - 0.5*l^{mn}_{ef} t^{ae}_{mn} t^{bf}_{ij} P(ab)
    + l^{mn}_{ef} t^{ae}_{im} t^{bf}_{jn} P(ab) + t^{a}_{i} t^{b}_{j} P(ab)

    + l^{mn}_{ef} t^{e}_{i} t^{f}_{j} t^{a}_{m} t^{b}_{n} + 0.5*l^{mn}_{ef} t^{e}_{i} t^{f}_{j} t^{ab}_{mn}

    + 0.5*l^{mn}_{ef} t^{a}_{m} t^{b}_{n} t^{ef}_{ij}
    + 0.25*l^{mn}_{ef} t^{ef}_{ij} t^{ab}_{mn}
    + t^{ab}_{ij}
    """
    Pij = contract("me, ei, abjm->abij", l_1, t_1, t_2)
    Pij += 0.5 * contract("mnef, efjm, abin->abij", l_2, t_2, t_2)
    out[v, v, o, o] += Pij
    out[v, v, o, o] -= Pij.swapaxes(2, 3)

    Pabij = -contract("me, ej, ai, bm->abij", l_1, t_1, t_1, t_1)
    Pabij += contract("me, ai, bejm->abij", l_1, t_1, t_2)
    Pabij -= contract("mnef, ei, am, bfjn->abij", l_2, t_1, t_1, t_2)
    Pabij -= 0.5 * contract("mnef, ej, ai, bfmn->abij", l_2, t_1, t_1, t_2)

    out[v, v, o, o] += Pabij
    out[v, v, o, o] -= Pabij.swapaxes(0, 1)
    out[v, v, o, o] -= Pabij.swapaxes(2, 3)
    out[v, v, o, o] += Pabij.swapaxes(0, 1).swapaxes(2, 3)

    Pab = contract("me, am, beij->abij", l_1, t_1, t_2)
    Pab += 0.5 * contract("mnef, an, bj, efim->abij", l_2, t_1, t_1, t_2)
    Pab += 0.5 * contract("mnef, ai, bn, efjm->abij", l_2, t_1, t_1, t_2)
    Pab -= 0.5 * contract("mnef, aemn, bfij->abij", l_2, t_2, t_2)
    Pab += contract("mnef, aeim, bfjn->abij", l_2, t_2, t_2)
    Pab += contract("ai,bj->abij", t_1, t_1)

    out[v, v, o, o] += Pab
    out[v, v, o, o] -= Pab.swapaxes(0, 1)

    out[v, v, o, o] += contract(
        "mnef, ei, fj, am, bn->abij", l_2, t_1, t_1, t_1, t_1
    )
    out[v, v, o, o] += 0.5 * contract(
        "mnef, ei, fj, abmn->abij", l_2, t_1, t_1, t_2
    )

    out[v, v, o, o] += 0.5 * contract(
        "mnef, am, bn, efij->abij", l_2, t_1, t_1, t_2
    )
    out[v, v, o, o] += 0.25 * contract("mnef, efij, abmn->abij", l_2, t_2, t_2)
    out[v, v, o, o] += t_2


def add_rho_ijab(t_1, t_2, l_1, l_2, o, v, out, np):
    """
    rho^{ij}_{ab} = l^{ij}_{ab}
    """
    out[o, o, v, v] += l_2


def add_rho_jbia(t_1, t_2, l_1, l_2, o, v, out, np):
    """
    rho^{jb}_{ia} = delta_{i j} l^{m}_{a} t^{b}_{m} + 0.5*delta_{i j} l^{mn}_{ae} t^{be}_{mn}
    - l^{j}_{a} t^{b}_{i} + l^{jm}_{ae} t^{e}_{i} t^{b}_{m}
    - l^{jm}_{ae} t^{be}_{im}
    """

    delta = np.eye(o.stop)

    rho_jbia = contract("ji, ma, bm->jbia", delta, l_1, t_1)
    rho_jbia += 0.5 * contract("ji, mnae, bemn->jbia", delta, l_2, t_2)
    rho_jbia -= contract("ja, bi->jbia", l_1, t_1)
    rho_jbia += contract("jmae, ei, bm->jbia", l_2, t_1, t_1)
    rho_jbia -= contract("jmae, beim->jbia", l_2, t_2)
    out[o, v, o, v] += rho_jbia
    out[v, o, o, v] += -rho_jbia.swapaxes(0, 1)
    out[o, v, v, o] += -rho_jbia.swapaxes(2, 3)
    out[v, o, v, o] += rho_jbia.swapaxes(0, 1).swapaxes(2, 3)


def add_rho_bcai(t_1, t_2, l_1, l_2, o, v, out, np):
    """
    rho^{bc}_{ai} =
    l^{m}_{a} t^{b}_{m} t^{c}_{i} P(bc) + l^{mn}_{ae} t^{b}_{m} t^{ce}_{in} P(bc)- 0.5*l^{mn}_{ae} t^{b}_{i} t^{ce}_{mn} P(bc)
    - l^{m}_{a} t^{bc}_{im} - l^{mn}_{ae} t^{e}_{i} t^{b}_{m} t^{c}_{n}
    - 0.5*l^{mn}_{ae} t^{e}_{i} t^{bc}_{mn}
    """

    Pbc = (
        contract("ma, bm, ci->bcai", l_1, t_1, t_1)
        + contract("mnae, bm, cein->bcai", l_2, t_1, t_2)
        - 0.5 * contract("mnae, bi, cemn->bcai", l_2, t_1, t_2)
    )
    out[v, v, v, o] += Pbc
    out[v, v, v, o] -= Pbc.swapaxes(0, 1)

    out[v, v, v, o] -= contract("ma, bcim->bcai", l_1, t_2)
    out[v, v, v, o] -= contract("mnae, ei, bm, cn->bcai", l_2, t_1, t_1, t_1)
    out[v, v, v, o] -= 0.5 * contract("mnae, ei, bcmn->bcai", l_2, t_1, t_2)

    out[v, v, o, v] -= out[v, v, v, o].swapaxes(2, 3)


def add_rho_ciab(t_1, t_2, l_1, l_2, o, v, out, np):
    """
    rho^{ci}_{ab} = - l^{im}_{ab} t^{c}_{m}
    rho^{ic}_{ab} = - rho^{ci}_{ab}
    """
    rho_ciab = -contract("imab, cm->ciab", l_2, t_1)
    out[v, o, v, v] += rho_ciab
    out[o, v, v, v] -= rho_ciab.transpose(1, 0, 2, 3)


def add_rho_cdab(t_1, t_2, l_1, l_2, o, v, out, np):
    """
    rho^{cd}_{ab} = l^{mn}_{ab} t^{c}_{m} t^{d}_{n} + 0.5*l^{mn}_{ab} t^{cd}_{mn}
    """
    out[v, v, v, v] += contract("mnab, cm, dn->cdab", l_2, t_1, t_1)
    out[v, v, v, v] += 0.5 * contract("mnab, cdmn->cdab", l_2, t_2)


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
