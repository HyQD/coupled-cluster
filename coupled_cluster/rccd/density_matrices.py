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

    add_rho_klij(t, l, o, v, out, np)
    add_rho_abij(t, l, o, v, out, np)
    add_rho_jbia(t, l, o, v, out, np)
    add_rho_bjia(t, l, o, v, out, np)
    add_rho_ijab(t, l, o, v, out, np)
    add_rho_cdab(t, l, o, v, out, np)

    return out


def add_rho_klij(t, l, o, v, out, np):

    no = o.stop
    nv = v.stop - no

    delta = np.eye(o.stop)

    rho = np.zeros((no, no, no, no), dtype=t.dtype)
    rho += 4 * contract("ik,jl->klij", delta, delta)
    rho -= 2 * contract("il,jk->klij", delta, delta)
    rho += contract("klab,abij->klij", l, t)

    I0 = contract("mkab,abmi->ki", l, t)

    rho -= 2 * contract("ik,lj->klij", delta, I0)

    rho += contract("il,kj->klij", delta, I0)

    rho += contract("jk,li->klij", delta, I0)

    rho -= 2 * contract("jl,ki->klij", delta, I0)

    out[o, o, o, o] += rho


def add_rho_abij(t, l, o, v, out, np):

    tt = 2 * t - t.swapaxes(2, 3)

    tmp_oo = contract("klcd, cdil->ki", l, t)
    out[v, v, o, o] -= contract("abkj, ki->abij", tt, tmp_oo)
    out[v, v, o, o] -= contract("abik, kj->abij", tt, tmp_oo)

    tmp_vv = contract("klcd, bdkl->bc", l, t)
    out[v, v, o, o] -= contract("acij, bc->abij", tt, tmp_vv)
    out[v, v, o, o] -= contract("bdji, ad->abij", tt, tmp_vv)

    tmp_ovvo = contract("klcd, acjk->ldaj", l, t)
    out[v, v, o, o] -= contract("bdil, ldaj->abij", tt, tmp_ovvo)
    out[v, v, o, o] += 2 * contract("acik, kcbj->abij", tt, tmp_ovvo)
    out[v, v, o, o] += contract("ackj, kcbi->abij", t, tmp_ovvo)

    tmp_oooo = contract("klcd, cdij->klij", l, t)
    out[v, v, o, o] += contract("abkl, klij->abij", t, tmp_oooo)

    tmp_ovvo = contract("klcd, bdlj->kcbj", l, t)
    out[v, v, o, o] -= contract("acik, kcbj->abij", tt, tmp_ovvo)

    tmp_ovvo = contract("kldc, bdli->kcbi", l, t)
    out[v, v, o, o] += contract("ackj, kcbi->abij", t, tmp_ovvo)

    out[v, v, o, o] += 2 * tt


def add_rho_jbia(t, l, o, v, out, np):

    no = o.stop
    nv = v.stop - no

    delta = np.eye(o.stop)

    rho = np.zeros((no, nv, no, nv), dtype=t.dtype)

    rho -= contract("kjac,bcki->jbia", l, t)
    rho -= contract("kjca,bcik->jbia", l, t)

    tmp = contract("klac,bckl->ab", l, t)
    rho += 2 * contract("ij,ab->jbia", delta, tmp)
    out[o, v, o, v] += rho
    out[v, o, v, o] += rho.transpose(1, 0, 3, 2)


def add_rho_bjia(t, l, o, v, out, np):

    no = o.stop
    nv = v.stop - no

    delta = np.eye(o.stop)

    rho = np.zeros((nv, no, no, nv), dtype=t.dtype)

    rho -= contract("kjca,bcki->bjia", l, t)
    rho += 2 * contract("kjca,bcik->bjia", l, t)

    tmp = contract("klac,bckl->ab", l, t)
    rho -= contract("ij,ab->bjia", delta, tmp)

    out[v, o, o, v] += rho
    out[o, v, v, o] += rho.transpose(1, 0, 3, 2)


def add_rho_ijab(t, l, o, v, out, np):
    out[o, o, v, v] += l


def add_rho_cdab(t, l, o, v, out, np):
    # out[v, v, v, v] += np.tensordot(t, l, axes=((2, 3), (0, 1)))
    out[v, v, v, v] += contract("ijab,cdij->cdab", l, t)
