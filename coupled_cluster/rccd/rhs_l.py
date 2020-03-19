def compute_l_2_amplitudes(f, u, t2, l2, o, v, np, out=None):
    """
    if out is None:
        out = np.zeros_like(l_1)
    """
    nocc = o.stop
    nvirt = v.stop - nocc

    I0_l2 = np.zeros((nocc, nocc, nocc, nocc), dtype=t2.dtype)

    I0_l2 += np.einsum("ijba,bakl->ijkl", l2, t2)

    rhs = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    rhs += np.einsum("jilk,lkba->ijab", I0_l2, u[o, o, v, v])

    del I0_l2

    I1_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I1_l2 += np.einsum("ik,jkab->ijab", f[o, o], l2)

    I6_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I6_l2 += np.einsum("ijab->ijab", I1_l2)

    del I1_l2

    I2_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I2_l2 += np.einsum("ca,ijbc->ijab", f[v, v], l2)

    I6_l2 -= np.einsum("ijab->ijab", I2_l2)

    del I2_l2

    I3_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I3_l2 += np.einsum("acki,jkcb->ijab", t2, u[o, o, v, v])

    I4_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I4_l2 += np.einsum("ijab->ijab", I3_l2)

    del I3_l2

    I4_l2 -= np.einsum("jaib->ijab", u[o, v, o, v])

    I5_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I5_l2 += np.einsum("kjcb,kiac->ijab", I4_l2, l2)

    I6_l2 -= np.einsum("ijab->ijab", I5_l2)

    del I5_l2

    rhs -= np.einsum("ijba->ijab", I6_l2)

    rhs -= np.einsum("jiab->ijab", I6_l2)

    del I6_l2

    I7_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I7_l2 += np.einsum("kjcb,kica->ijab", I4_l2, l2)

    del I4_l2

    I14_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I14_l2 -= np.einsum("ijab->ijab", I7_l2)

    del I7_l2

    I8_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I8_l2 -= np.einsum("abji->ijab", t2)

    I8_l2 += 2 * np.einsum("baji->ijab", t2)

    I9_l2 = np.zeros((nvirt, nvirt), dtype=t2.dtype)

    I9_l2 += np.einsum("ijbc,ijac->ab", I8_l2, u[o, o, v, v])

    del I8_l2

    I10_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I10_l2 += np.einsum("bc,ijca->ijab", I9_l2, l2)

    del I9_l2

    I14_l2 += np.einsum("jiab->ijab", I10_l2)

    del I10_l2

    I11_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I11_l2 += 2 * np.einsum("abji->ijab", t2)

    I11_l2 -= np.einsum("baji->ijab", t2)

    I12_l2 = np.zeros((nocc, nocc), dtype=t2.dtype)

    I12_l2 += np.einsum("kjba,kiab->ij", I11_l2, u[o, o, v, v])

    I13_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I13_l2 += np.einsum("jk,kiab->ijab", I12_l2, l2)

    del I12_l2

    I14_l2 += np.einsum("ijba->ijab", I13_l2)

    del I13_l2

    rhs -= np.einsum("ijab->ijab", I14_l2)

    rhs -= np.einsum("jiba->ijab", I14_l2)

    del I14_l2

    I16_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I16_l2 += np.einsum("kjbc,kica->ijab", I11_l2, u[o, o, v, v])

    del I11_l2

    I17_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I17_l2 += np.einsum("jiba->ijab", I16_l2)

    del I16_l2

    I15_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I15_l2 += np.einsum("caki,jkcb->ijab", t2, u[o, o, v, v])

    I17_l2 -= np.einsum("ijab->ijab", I15_l2)

    del I15_l2

    I17_l2 += np.einsum("jabi->ijab", u[o, v, v, o])

    I18_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I18_l2 += np.einsum("kjcb,kica->ijab", I17_l2, l2)

    del I17_l2

    rhs += 2 * np.einsum("ijab->ijab", I18_l2)

    rhs -= np.einsum("ijba->ijab", I18_l2)

    rhs -= np.einsum("jiab->ijab", I18_l2)

    rhs += 2 * np.einsum("jiba->ijab", I18_l2)

    del I18_l2

    I19_l2 = np.zeros((nocc, nocc), dtype=t2.dtype)

    I19_l2 += np.einsum("ikba,abkj->ij", l2, t2)

    I20_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I20_l2 += np.einsum("ik,jkab->ijab", I19_l2, u[o, o, v, v])

    del I19_l2

    I23_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I23_l2 += np.einsum("ijab->ijab", I20_l2)

    del I20_l2

    I21_l2 = np.zeros((nvirt, nvirt), dtype=t2.dtype)

    I21_l2 += np.einsum("jica,cbji->ab", l2, t2)

    I22_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    I22_l2 += np.einsum("ac,ijbc->ijab", I21_l2, u[o, o, v, v])

    del I21_l2

    I23_l2 += np.einsum("ijab->ijab", I22_l2)

    del I22_l2

    rhs += np.einsum("ijab->ijab", I23_l2)

    rhs -= 2 * np.einsum("ijba->ijab", I23_l2)

    rhs -= 2 * np.einsum("jiab->ijab", I23_l2)

    rhs += np.einsum("jiba->ijab", I23_l2)

    del I23_l2

    I24_l2 = np.zeros((nocc, nocc, nocc, nocc), dtype=t2.dtype)

    I24_l2 += np.einsum("jilk->ijkl", u[o, o, o, o])

    I24_l2 += np.einsum("ablk,ijba->ijkl", t2, u[o, o, v, v])

    rhs += np.einsum("jilk,klab->ijab", I24_l2, l2)

    del I24_l2

    rhs += np.einsum("jidc,dcba->ijab", l2, u[v, v, v, v])

    rhs -= 2 * np.einsum("jiab->ijab", u[o, o, v, v])

    rhs += 4 * np.einsum("jiba->ijab", u[o, o, v, v])

    return rhs
