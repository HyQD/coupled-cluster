def compute_t_2_amplitudes(f, u, t, o, v, np, out=None):
    """
    if out is None:
        out = np.zeros_like(t_2)
    """
    nocc = o.stop
    nvirt = v.stop - nocc

    I0_t2 = np.zeros((nocc, nocc, nocc, nocc), dtype=t.dtype)

    I0_t2 += np.einsum("abij,klab->ijkl", t, u[o, o, v, v])

    rhs = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t.dtype)

    rhs += np.einsum("ijlk,ablk->abij", I0_t2, t)

    del I0_t2

    I1_t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t.dtype)

    I1_t2 += np.einsum("ki,abjk->ijab", f[o, o], t)

    I3_t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t.dtype)

    I3_t2 += np.einsum("ijab->ijab", I1_t2)

    del I1_t2

    I2_t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t.dtype)

    I2_t2 += np.einsum("ac,bcij->ijab", f[v, v], t)

    I3_t2 -= np.einsum("ijab->ijab", I2_t2)

    del I2_t2

    rhs -= np.einsum("ijba->abij", I3_t2)

    rhs -= np.einsum("jiab->abij", I3_t2)

    del I3_t2

    I4_t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t.dtype)

    I4_t2 -= np.einsum("jiab->ijab", u[o, o, v, v])

    I4_t2 += 2 * np.einsum("jiba->ijab", u[o, o, v, v])

    I5_t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t.dtype)

    I5_t2 += np.einsum("kjcb,acki->ijab", I4_t2, t)

    I6_t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t.dtype)

    I6_t2 += np.einsum("jkbc,caki->ijab", I5_t2, t)

    del I5_t2

    I11_t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t.dtype)

    I11_t2 += np.einsum("ijab->ijab", I6_t2)

    del I6_t2

    I7_t2 = np.zeros((nvirt, nvirt), dtype=t.dtype)

    I7_t2 += np.einsum("ijbc,acij->ab", I4_t2, t)

    I8_t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t.dtype)

    I8_t2 += np.einsum("bc,caij->ijab", I7_t2, t)

    del I7_t2

    I11_t2 += np.einsum("jiab->ijab", I8_t2)

    del I8_t2

    I9_t2 = np.zeros((nocc, nocc), dtype=t.dtype)

    I9_t2 += np.einsum("kjab,abki->ij", I4_t2, t)

    I10_t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t.dtype)

    I10_t2 += np.einsum("jk,abki->ijab", I9_t2, t)

    del I9_t2

    I11_t2 += np.einsum("ijba->ijab", I10_t2)

    del I10_t2

    rhs -= np.einsum("ijab->abij", I11_t2)

    rhs -= np.einsum("jiba->abij", I11_t2)

    del I11_t2

    I12_t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t.dtype)

    I12_t2 += 2 * np.einsum("kjcb,caki->ijab", I4_t2, t)

    del I4_t2

    I12_t2 += 2 * np.einsum("jabi->ijab", u[o, v, v, o])

    I12_t2 -= np.einsum("jaib->ijab", u[o, v, o, v])

    rhs += np.einsum("jkbc,caki->abij", I12_t2, t)

    del I12_t2

    I13_t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t.dtype)

    I13_t2 -= np.einsum("jabi->ijab", u[o, v, v, o])

    I13_t2 += np.einsum("caik,jkbc->ijab", t, u[o, o, v, v])

    rhs += np.einsum("ikac,bckj->abij", I13_t2, t)

    del I13_t2

    I14_t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t.dtype)

    I14_t2 -= np.einsum("jaib->ijab", u[o, v, o, v])

    I14_t2 += np.einsum("caik,kjbc->ijab", t, u[o, o, v, v])

    rhs += np.einsum("ikbc,ackj->abij", I14_t2, t)

    del I14_t2

    rhs += np.einsum("baji->abij", u[v, v, o, o])

    rhs -= np.einsum("bcjk,kaic->abij", t, u[o, v, o, v])

    rhs -= np.einsum("caik,kbcj->abij", t, u[o, v, v, o])

    rhs -= np.einsum("cbik,kajc->abij", t, u[o, v, o, v])

    rhs += 2 * np.einsum("bcjk,kaci->abij", t, u[o, v, v, o])

    rhs += np.einsum("balk,lkji->abij", t, u[o, o, o, o])

    rhs += np.einsum("cdji,abdc->abij", t, u[v, v, v, v])

    return rhs
