from opt_einsum import contract


def compute_one_body_density_matrix(t2, l2, o, v, np, out=None):

    nocc = o.stop
    nvirt = v.stop - nocc

    rho = np.zeros((nocc + nvirt, nocc + nvirt), dtype=t2.dtype)

    rho[o, o] += 2 * np.eye(nocc)
    rho[o, o] -= contract("kjab,baik->ji", l2, t2)

    rho[v, v] += contract("ijac,bcij->ba", l2, t2)

    return rho


def compute_two_body_density_matrix(t, l, o, v, np, out=None):
    """
    The final two body density matrix should satisfy

        contract('pqpq->', rho_qspr) = N(N-1)

    where N is the number of electrons.
    """
    if out is None:
        out = np.zeros((v.stop, v.stop, v.stop, v.stop), dtype=t.dtype)
    out.fill(0)

    add_rho_klij(t, l, o, v, np, out)
    add_rho_abij(t, l, o, v, np, out)
    add_rho_jbia(t, l, o, v, np, out)
    add_rho_bjia(t, l, o, v, np, out)
    add_rho_ijab(t, l, o, v, np, out)
    # add_rho_cdab(t, l, o, v, out, np)

    return out


def add_rho_klij(t, l, o, v, np, out):

    no = o.stop
    nv = v.stop - no

    delta = np.eye(o.stop)

    rho = np.zeros((no, no, no, no), dtype=t.dtype)
    rho += 4 * contract("ik,jl->klij", delta, delta)
    rho -= 2 * contract("il,jk->klij", delta, delta)
    # rho += contract("klab,abij->klij", l, t)

    I0 = contract("mkab,abmi->ki", l, t)

    rho -= 2 * contract("ik,lj->klij", delta, I0)

    rho += contract("il,kj->klij", delta, I0)

    rho += contract("jk,li->klij", delta, I0)

    rho -= 2 * contract("jl,ki->klij", delta, I0)

    out[o, o, o, o] += rho


def add_rho_abij(t, l, o, v, np, out):

    no = o.stop
    nv = v.stop - no

    tt = 2 * t - t.transpose(0, 1, 3, 2)

    out[v, v, o, o] += 2 * tt


def add_rho_jbia(t, l, o, v, np, out):

    no = o.stop
    nv = v.stop - no

    delta = np.eye(o.stop)

    rho = np.zeros((no, nv, no, nv), dtype=t.dtype)

    tmp = contract("ijac,bcij->ba", l, t)
    rho += 2 * contract("ij,ba->jbia", delta, tmp)
    out[o, v, o, v] += rho
    out[v, o, v, o] += rho.transpose(1, 0, 3, 2)


def add_rho_bjia(t, l, o, v, np, out):

    no = o.stop
    nv = v.stop - no

    delta = np.eye(o.stop)

    rho = np.zeros((nv, no, no, nv), dtype=t.dtype)

    tmp = contract("ijac,bcij->ba", l, t)
    rho -= contract("ij,ba->bjia", delta, tmp)

    out[v, o, o, v] += rho
    out[o, v, v, o] += rho.transpose(1, 0, 3, 2)


def add_rho_ijab(t, l, o, v, np, out):
    out[o, o, v, v] += l
