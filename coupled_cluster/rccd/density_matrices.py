def compute_one_body_density_matrix(t2, l2, o, v, np, out=None):

    nocc = o.stop
    nvirt = v.stop - nocc

    rho = np.zeros((nocc + nvirt, nocc + nvirt), dtype=t2.dtype)

    rho[o, o] += 2 * np.eye(nocc)
    rho[o, o] -= np.einsum("kjab,baik->ij", l2, t2)

    rho[v, v] += np.einsum("ijac,bcij->ab", l2, t2)

    return rho


def compute_two_body_density_matrix(t, l, o, v, np, out=None):
    """
    The final two body density matrix should satisfy

        np.einsum('pqpq->', rho_qspr) = N(N-1)

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

    I0 = np.zeros((no, no), dtype=t.dtype)

    I0 += np.einsum("ikab,abjk->ij", l, t)

    rho = np.zeros((no, no, no, no), dtype=t.dtype)

    rho -= 2 * np.einsum("ik,jl->ijkl", delta, I0)

    rho += np.einsum("il,jk->ijkl", delta, I0)

    rho += np.einsum("jk,il->ijkl", delta, I0)

    rho -= 2 * np.einsum("jl,ik->ijkl", delta, I0)

    del I0

    rho += np.einsum("jiab,ablk->ijkl", l, t)

    rho -= 2 * np.einsum("il,jk->ijkl", delta, delta)

    rho += 4 * np.einsum("ik,jl->ijkl", delta, delta)

    out[o, o, o, o] += rho


def add_rho_abij(t, l, o, v, out, np):

    no = o.stop
    nv = v.stop - no

    I0 = np.zeros((no, no, no, no), dtype=t.dtype)

    I0 += np.einsum("ijba,bakl->ijkl", l, t)

    rho = np.zeros((nv, nv, no, no), dtype=t.dtype)

    rho += np.einsum("lkij,ablk->abij", I0, t)

    del I0

    I1 = np.zeros((no, no, nv, nv), dtype=t.dtype)

    I1 += np.einsum("ikac,cbjk->ijab", l, t)

    rho += np.einsum("kica,cbjk->abij", I1, t)

    del I1

    I2 = np.zeros((no, no, nv, nv), dtype=t.dtype)

    I2 += np.einsum("ikca,cbjk->ijab", l, t)

    rho += np.einsum("kicb,cajk->abij", I2, t)

    del I2

    I3 = np.zeros((no, no, nv, nv), dtype=t.dtype)

    I3 += np.einsum("kica,bcjk->ijab", l, t)

    I4 = np.zeros((no, no, nv, nv), dtype=t.dtype)

    I4 += np.einsum("kjcb,acik->ijab", I3, t)

    I5 = np.zeros((no, no, nv, nv), dtype=t.dtype)

    I5 += np.einsum("ijab->ijab", I4)

    del I4

    I10 = np.zeros((no, no, nv, nv), dtype=t.dtype)

    I10 += np.einsum("kica,cbjk->ijab", I3, t)

    del I3

    I11 = np.zeros((no, no, nv, nv), dtype=t.dtype)

    I11 += np.einsum("ijab->ijab", I10)

    del I10

    I5 += np.einsum("baji->ijab", t)

    rho -= 2 * np.einsum("ijba->abij", I5)

    rho += 4 * np.einsum("ijab->abij", I5)

    del I5

    I6 = np.zeros((no, no), dtype=t.dtype)

    I6 += np.einsum("kiab,bajk->ij", l, t)

    I7 = np.zeros((no, no, nv, nv), dtype=t.dtype)

    I7 += np.einsum("kj,abik->ijab", I6, t)

    del I6

    I11 += np.einsum("ijab->ijab", I7)

    del I7

    I8 = np.zeros((nv, nv), dtype=t.dtype)

    I8 += np.einsum("jiac,bcji->ab", l, t)

    I9 = np.zeros((no, no, nv, nv), dtype=t.dtype)

    I9 += np.einsum("cb,acij->ijab", I8, t)

    del I8

    I11 += np.einsum("ijab->ijab", I9)

    del I9

    rho -= 2 * np.einsum("ijab->abij", I11)

    rho += np.einsum("ijba->abij", I11)

    rho += np.einsum("jiab->abij", I11)

    rho -= 2 * np.einsum("jiba->abij", I11)

    del I11

    out[v, v, o, o] += rho


def add_rho_jbia(t, l, o, v, out, np):

    no = o.stop
    nv = v.stop - no

    delta = np.eye(o.stop)

    rho = np.zeros((no, nv, no, nv), dtype=t.dtype)

    rho -= np.einsum("kjac,bcki->jbia", l, t)
    rho -= np.einsum("kjca,bcik->jbia", l, t)

    tmp = np.einsum("klac,bckl->ab", l, t)
    rho += 2 * np.einsum("ij,ab->jbia", delta, tmp)
    out[o, v, o, v] += rho
    out[v, o, v, o] += rho.transpose(1, 0, 3, 2)


def add_rho_bjia(t, l, o, v, out, np):

    no = o.stop
    nv = v.stop - no

    delta = np.eye(o.stop)

    rho = np.zeros((nv, no, no, nv), dtype=t.dtype)

    rho -= np.einsum("kjca,bcki->bjia", l, t)
    rho += 2 * np.einsum("kjca,bcik->bjia", l, t)

    tmp = np.einsum("klac,bckl->ab", l, t)
    rho -= np.einsum("ij,ab->bjia", delta, tmp)

    out[v, o, o, v] += rho
    out[o, v, v, o] += rho.transpose(1, 0, 3, 2)


def add_rho_ijab(t, l, o, v, out, np):
    out[o, o, v, v] += l


def add_rho_cdab(t, l, o, v, out, np):
    out[v, v, v, v] += np.tensordot(t, l, axes=((2, 3), (0, 1)))
