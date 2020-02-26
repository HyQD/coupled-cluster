import coupled_cluster.ccs.rhs_l as ccs_l
import coupled_cluster.ccd.rhs_l as ccd_l


def compute_l_1_amplitudes(f, u, t1, t2, l1, l2, o, v, np, out=None):
    """
    if out is None:
        out = np.zeros_like(l_1)
    """
    no = o.stop
    nv = v.stop - no

    I0_l1 = np.zeros((no, no))

    I0_l1 += np.einsum("ikba,abkj->ij", l2, t2)

    I22_l1 = np.zeros((no, no, no, nv))

    I22_l1 -= 2 * np.einsum("ik,aj->ijka", I0_l1, t1)

    I25_l1 = np.zeros((no, no, no, nv))

    I25_l1 += np.einsum("ik,aj->ijka", I0_l1, t1)

    I30_l1 = np.zeros((no, no))

    I30_l1 += np.einsum("ij->ij", I0_l1)

    rhs = np.zeros((no, nv))

    rhs -= np.einsum("ij,ja->ia", I0_l1, f[o, v])

    del I0_l1

    I1_l1 = np.zeros((no, no, no, no))

    I1_l1 += np.einsum("ijba,ablk->ijkl", l2, t2)

    I21_l1 = np.zeros((no, no, no, no))

    I21_l1 += np.einsum("jilk->ijkl", I1_l1)

    rhs += np.einsum("ijlk,klja->ia", I1_l1, u[o, o, o, v])

    del I1_l1

    I2_l1 = np.zeros((no, no, no, nv))

    I2_l1 += np.einsum("bk,jiba->ijka", t1, l2)

    I3_l1 = np.zeros((no, no, no, no))

    I3_l1 += np.einsum("ak,ijla->ijkl", t1, I2_l1)

    I21_l1 += np.einsum("ijkl->ijkl", I3_l1)

    I22_l1 += np.einsum("al,lijk->ijka", t1, I21_l1)

    del I21_l1

    rhs += np.einsum("iljk,kjla->ia", I3_l1, u[o, o, o, v])

    del I3_l1

    I25_l1 += np.einsum("likb,abjl->ijka", I2_l1, t2)

    I25_l1 += np.einsum("ilkb,bajl->ijka", I2_l1, t2)

    rhs += np.einsum("ijkb,jkab->ia", I25_l1, u[o, o, v, v])

    del I25_l1

    I31_l1 = np.zeros((no, nv))

    I31_l1 += np.einsum("kjib,bakj->ia", I2_l1, t2)

    I4_l1 = np.zeros((no, no, no, nv))

    I4_l1 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v])

    I5_l1 = np.zeros((no, no, no, nv))

    I5_l1 -= np.einsum("ijka->ijka", I4_l1)

    I5_l1 += 2 * np.einsum("ikja->ijka", I4_l1)

    I6_l1 = np.zeros((no, no, no, nv))

    I6_l1 += np.einsum("ijka->ijka", I4_l1)

    I12_l1 = np.zeros((no, no, no, nv))

    I12_l1 += np.einsum("kjia->ijka", I4_l1)

    del I4_l1

    I5_l1 += 2 * np.einsum("jkia->ijka", u[o, o, o, v])

    I5_l1 -= np.einsum("kjia->ijka", u[o, o, o, v])

    I14_l1 = np.zeros((no, no, no, nv))

    I14_l1 -= np.einsum("kilb,balj->ijka", I5_l1, t2)

    del I5_l1

    I6_l1 += np.einsum("kjia->ijka", u[o, o, o, v])

    I14_l1 += np.einsum("klib,ablj->ijka", I6_l1, t2)

    I24_l1 = np.zeros((no, no, no, nv))

    I24_l1 += np.einsum("ijlb,ablk->ijka", I6_l1, t2)

    del I6_l1

    I7_l1 = np.zeros((no, no, nv, nv))

    I7_l1 += 2 * np.einsum("jiab->ijab", u[o, o, v, v])

    I7_l1 -= np.einsum("jiba->ijab", u[o, o, v, v])

    I8_l1 = np.zeros((no, nv))

    I8_l1 += np.einsum("bj,jiab->ia", t1, I7_l1)

    I9_l1 = np.zeros((no, nv))

    I9_l1 += np.einsum("ia->ia", I8_l1)

    del I8_l1

    I9_l1 += np.einsum("ia->ia", f[o, v])

    I14_l1 -= np.einsum("ib,bakj->ijka", I9_l1, t2)

    I35_l1 = np.zeros((no, no))

    I35_l1 += np.einsum("ia,aj->ij", I9_l1, t1)

    I10_l1 = np.zeros((no, no, nv, nv))

    I10_l1 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v])

    I11_l1 = np.zeros((no, no, nv, nv))

    I11_l1 += np.einsum("jiab->ijab", I10_l1)

    del I10_l1

    I11_l1 += np.einsum("iabj->ijab", u[o, v, v, o])

    I14_l1 -= np.einsum("bk,ijab->ijka", t1, I11_l1)

    rhs -= np.einsum("jkba,kijb->ia", I11_l1, I2_l1)

    del I11_l1

    I12_l1 += np.einsum("ijka->ijka", u[o, o, o, v])

    I13_l1 = np.zeros((no, no, no, no))

    I13_l1 += np.einsum("al,ijka->ijkl", t1, I12_l1)

    del I12_l1

    I13_l1 += np.einsum("jilk->ijkl", u[o, o, o, o])

    I13_l1 += np.einsum("ablk,ijba->ijkl", t2, u[o, o, v, v])

    I14_l1 += np.einsum("al,lijk->ijka", t1, I13_l1)

    del I13_l1

    I14_l1 -= np.einsum("iakj->ijka", u[o, v, o, o])

    I14_l1 -= np.einsum("bcjk,iacb->ijka", t2, u[o, v, v, v])

    rhs += np.einsum("ijkb,jkba->ia", I14_l1, l2)

    del I14_l1

    I15_l1 = np.zeros((no, no, nv, nv))

    I15_l1 += np.einsum("ikac,bcjk->ijab", l2, t2)

    I15_l1 += np.einsum("ikca,cbjk->ijab", l2, t2)

    rhs -= np.einsum("ijbc,jbac->ia", I15_l1, u[o, v, v, v])

    del I15_l1

    I16_l1 = np.zeros((no, no, nv, nv))

    I16_l1 += 2 * np.einsum("abji->ijab", t2)

    I16_l1 -= np.einsum("baji->ijab", t2)

    I17_l1 = np.zeros((no, no, nv, nv))

    I17_l1 += np.einsum("kjbc,kica->ijab", I16_l1, l2)

    rhs += np.einsum("ijbc,jbca->ia", I17_l1, u[o, v, v, v])

    del I17_l1

    I31_l1 -= np.einsum("jb,jiab->ia", l1, I16_l1)

    del I16_l1

    I18_l1 = np.zeros((no, nv, nv, nv))

    I18_l1 += np.einsum("abic->iabc", u[v, v, o, v])

    I18_l1 += np.einsum("di,bacd->iabc", t1, u[v, v, v, v])

    rhs += np.einsum("jbca,jibc->ia", I18_l1, l2)

    del I18_l1

    I19_l1 = np.zeros((no, no, no, nv))

    I19_l1 += np.einsum("ib,bakj->ijka", l1, t2)

    I22_l1 -= 2 * np.einsum("ijka->ijka", I19_l1)

    I22_l1 += np.einsum("ikja->ijka", I19_l1)

    del I19_l1

    I20_l1 = np.zeros((no, no, nv, nv))

    I20_l1 -= np.einsum("abji->ijab", t2)

    I20_l1 += 2 * np.einsum("baji->ijab", t2)

    I22_l1 -= np.einsum("ljba,likb->ijka", I20_l1, I2_l1)

    rhs += np.einsum("ijkb,jkba->ia", I22_l1, u[o, o, v, v])

    del I22_l1

    I35_l1 += np.einsum("kjab,kiab->ij", I20_l1, u[o, o, v, v])

    del I20_l1

    I23_l1 = np.zeros((no, no, no, no))

    I23_l1 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v])

    I24_l1 += np.einsum("al,ijlk->ijka", t1, I23_l1)

    del I23_l1

    I24_l1 -= np.einsum("bi,jakb->ijka", t1, u[o, v, o, v])

    rhs += np.einsum("jikb,kjab->ia", I24_l1, l2)

    del I24_l1

    I26_l1 = np.zeros((no, no, nv, nv))

    I26_l1 += np.einsum("iajb->ijab", u[o, v, o, v])

    I26_l1 += np.einsum("cj,iacb->ijab", t1, u[o, v, v, v])

    rhs -= np.einsum("jkba,ikjb->ia", I26_l1, I2_l1)

    del I2_l1

    del I26_l1

    I27_l1 = np.zeros((nv, nv))

    I27_l1 += np.einsum("ia,bi->ab", l1, t1)

    I27_l1 += np.einsum("jica,cbji->ab", l2, t2)

    I28_l1 = np.zeros((no, nv, nv, nv))

    I28_l1 -= np.einsum("iabc->iabc", u[o, v, v, v])

    I28_l1 += 2 * np.einsum("iacb->iabc", u[o, v, v, v])

    I34_l1 = np.zeros((nv, nv))

    I34_l1 += np.einsum("ci,iabc->ab", t1, I28_l1)

    rhs += np.einsum("bc,ibca->ia", I27_l1, I28_l1)

    del I27_l1

    del I28_l1

    I29_l1 = np.zeros((no, no))

    I29_l1 += np.einsum("ia,aj->ij", l1, t1)

    I30_l1 += np.einsum("ij->ij", I29_l1)

    I31_l1 += np.einsum("ji,aj->ia", I30_l1, t1)

    rhs -= np.einsum("ij,ja->ia", I29_l1, I9_l1)

    del I9_l1

    del I29_l1

    I31_l1 -= 2 * np.einsum("ai->ia", t1)

    rhs -= np.einsum("jb,jiab->ia", I31_l1, I7_l1)

    del I31_l1

    del I7_l1

    I32_l1 = np.zeros((no, no, nv, nv))

    I32_l1 += 2 * np.einsum("iabj->ijab", u[o, v, v, o])

    I32_l1 -= np.einsum("iajb->ijab", u[o, v, o, v])

    rhs += np.einsum("jb,ijba->ia", l1, I32_l1)

    del I32_l1

    I33_l1 = np.zeros((no, no, no, nv))

    I33_l1 += 2 * np.einsum("ijka->ijka", u[o, o, o, v])

    I33_l1 -= np.einsum("jika->ijka", u[o, o, o, v])

    I35_l1 += np.einsum("ak,ikja->ij", t1, I33_l1)

    rhs -= np.einsum("jk,kija->ia", I30_l1, I33_l1)

    del I33_l1

    del I30_l1

    I34_l1 += np.einsum("ab->ab", f[v, v])

    rhs += np.einsum("ba,ib->ia", I34_l1, l1)

    del I34_l1

    I35_l1 += np.einsum("ij->ij", f[o, o])

    rhs -= np.einsum("ij,ja->ia", I35_l1, l1)

    del I35_l1

    rhs += 2 * np.einsum("ia->ia", f[o, v])

    return rhs


def compute_l_2_amplitudes(f, u, t1, t2, l1, l2, o, v, np, out=None):
    """
    if out is None:
        out = np.zeros_like(l_2)
    """
    no = o.stop
    nv = v.stop - no

    I0_l2 = np.zeros((no, no, nv, nv))

    I0_l2 += np.einsum("ca,jicb->ijab", f[v, v], l2)

    I15_l2 = np.zeros((no, no, nv, nv))

    I15_l2 -= np.einsum("ijab->ijab", I0_l2)

    del I0_l2

    I1_l2 = np.zeros((no, no, no, nv))

    I1_l2 += np.einsum("bk,jiba->ijka", t1, l2)

    I2_l2 = np.zeros((no, no, nv, nv))

    I2_l2 += np.einsum("ka,ijkb->ijab", f[o, v], I1_l2)

    I15_l2 += np.einsum("ijab->ijab", I2_l2)

    del I2_l2

    I36_l2 = np.zeros((no, no, nv, nv))

    I36_l2 += np.einsum("jikc,kcab->ijab", I1_l2, u[o, v, v, v])

    I52_l2 = np.zeros((no, no, nv, nv))

    I52_l2 += np.einsum("ijab->ijab", I36_l2)

    del I36_l2

    I54_l2 = np.zeros((no, no, no, no))

    I54_l2 += np.einsum("al,jika->ijkl", t1, I1_l2)

    I3_l2 = np.zeros((no, no, no, no))

    I3_l2 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v])

    I4_l2 = np.zeros((no, no, nv, nv))

    I4_l2 += np.einsum("kijl,lkba->ijab", I3_l2, l2)

    del I3_l2

    I15_l2 -= np.einsum("ijab->ijab", I4_l2)

    del I4_l2

    I5_l2 = np.zeros((no, no, nv, nv))

    I5_l2 += np.einsum("ci,jacb->ijab", t1, u[o, v, v, v])

    I7_l2 = np.zeros((no, no, nv, nv))

    I7_l2 += np.einsum("ijab->ijab", I5_l2)

    del I5_l2

    I6_l2 = np.zeros((no, no, nv, nv))

    I6_l2 += np.einsum("acki,jkcb->ijab", t2, u[o, o, v, v])

    I7_l2 -= np.einsum("ijab->ijab", I6_l2)

    del I6_l2

    I7_l2 += np.einsum("jaib->ijab", u[o, v, o, v])

    I8_l2 = np.zeros((no, no, nv, nv))

    I8_l2 += np.einsum("kica,kjbc->ijab", I7_l2, l2)

    I15_l2 += np.einsum("jiba->ijab", I8_l2)

    del I8_l2

    I37_l2 = np.zeros((no, no, nv, nv))

    I37_l2 += np.einsum("kica,kjcb->ijab", I7_l2, l2)

    del I7_l2

    I52_l2 += np.einsum("jiba->ijab", I37_l2)

    del I37_l2

    I9_l2 = np.zeros((no, no, no, nv))

    I9_l2 += np.einsum("bi,kjba->ijka", t1, u[o, o, v, v])

    I10_l2 = np.zeros((no, no, no, nv))

    I10_l2 += np.einsum("ijka->ijka", I9_l2)

    I17_l2 = np.zeros((no, no, nv, nv))

    I17_l2 += np.einsum("ka,kijb->ijab", l1, I9_l2)

    I24_l2 = np.zeros((no, no, nv, nv))

    I24_l2 += np.einsum("ijab->ijab", I17_l2)

    del I17_l2

    I53_l2 = np.zeros((no, no, no, no))

    I53_l2 += np.einsum("al,kjia->ijkl", t1, I9_l2)

    del I9_l2

    I10_l2 += np.einsum("kjia->ijka", u[o, o, o, v])

    I11_l2 = np.zeros((no, no, nv, nv))

    I11_l2 += np.einsum("klia,kjlb->ijab", I10_l2, I1_l2)

    I15_l2 -= np.einsum("jiba->ijab", I11_l2)

    del I11_l2

    I32_l2 = np.zeros((no, no, nv, nv))

    I32_l2 += np.einsum("kila,jklb->ijab", I10_l2, I1_l2)

    I35_l2 = np.zeros((no, no, nv, nv))

    I35_l2 += np.einsum("jiba->ijab", I32_l2)

    del I32_l2

    I38_l2 = np.zeros((no, no, nv, nv))

    I38_l2 += np.einsum("klia,jklb->ijab", I10_l2, I1_l2)

    del I10_l2

    I52_l2 -= np.einsum("jiba->ijab", I38_l2)

    del I38_l2

    I12_l2 = np.zeros((no, no))

    I12_l2 += np.einsum("ia,aj->ij", f[o, v], t1)

    I13_l2 = np.zeros((no, no))

    I13_l2 += np.einsum("ij->ij", I12_l2)

    del I12_l2

    I13_l2 += np.einsum("ij->ij", f[o, o])

    I14_l2 = np.zeros((no, no, nv, nv))

    I14_l2 += np.einsum("ik,kjab->ijab", I13_l2, l2)

    del I13_l2

    I15_l2 += np.einsum("ijba->ijab", I14_l2)

    del I14_l2

    rhs = np.zeros((no, no, nv, nv))

    rhs -= np.einsum("ijba->ijab", I15_l2)

    rhs -= np.einsum("jiab->ijab", I15_l2)

    del I15_l2

    I16_l2 = np.zeros((no, no, nv, nv))

    I16_l2 += np.einsum("ic,jcab->ijab", l1, u[o, v, v, v])

    I24_l2 -= np.einsum("ijab->ijab", I16_l2)

    del I16_l2

    I18_l2 = np.zeros((nv, nv))

    I18_l2 += np.einsum("jica,cbji->ab", l2, t2)

    I19_l2 = np.zeros((no, no, nv, nv))

    I19_l2 += np.einsum("ac,jicb->ijab", I18_l2, u[o, o, v, v])

    del I18_l2

    I24_l2 += np.einsum("ijab->ijab", I19_l2)

    del I19_l2

    I20_l2 = np.zeros((no, no))

    I20_l2 += np.einsum("ia,aj->ij", l1, t1)

    I22_l2 = np.zeros((no, no))

    I22_l2 += np.einsum("ij->ij", I20_l2)

    del I20_l2

    I21_l2 = np.zeros((no, no))

    I21_l2 += np.einsum("ikba,abkj->ij", l2, t2)

    I22_l2 += np.einsum("ij->ij", I21_l2)

    del I21_l2

    I23_l2 = np.zeros((no, no, nv, nv))

    I23_l2 += np.einsum("ik,kjab->ijab", I22_l2, u[o, o, v, v])

    del I22_l2

    I24_l2 += np.einsum("ijba->ijab", I23_l2)

    del I23_l2

    rhs += np.einsum("ijab->ijab", I24_l2)

    rhs -= 2 * np.einsum("ijba->ijab", I24_l2)

    rhs -= 2 * np.einsum("jiab->ijab", I24_l2)

    rhs += np.einsum("jiba->ijab", I24_l2)

    del I24_l2

    I25_l2 = np.zeros((no, no, nv, nv))

    I25_l2 += np.einsum("ka,ijkb->ijab", l1, u[o, o, o, v])

    I35_l2 += np.einsum("ijab->ijab", I25_l2)

    del I25_l2

    I26_l2 = np.zeros((no, no, nv, nv))

    I26_l2 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v])

    I30_l2 = np.zeros((no, no, nv, nv))

    I30_l2 += np.einsum("ijab->ijab", I26_l2)

    del I26_l2

    I27_l2 = np.zeros((no, no, nv, nv))

    I27_l2 += np.einsum("caki,jkcb->ijab", t2, u[o, o, v, v])

    I30_l2 -= np.einsum("ijab->ijab", I27_l2)

    del I27_l2

    I28_l2 = np.zeros((no, no, nv, nv))

    I28_l2 += 2 * np.einsum("abji->ijab", t2)

    I28_l2 -= np.einsum("baji->ijab", t2)

    I29_l2 = np.zeros((no, no, nv, nv))

    I29_l2 += np.einsum("kiac,kjcb->ijab", I28_l2, u[o, o, v, v])

    I30_l2 += np.einsum("ijab->ijab", I29_l2)

    del I29_l2

    I39_l2 = np.zeros((nv, nv))

    I39_l2 += np.einsum("ijac,ijcb->ab", I28_l2, u[o, o, v, v])

    del I28_l2

    I42_l2 = np.zeros((nv, nv))

    I42_l2 += np.einsum("ab->ab", I39_l2)

    del I39_l2

    I30_l2 += np.einsum("jabi->ijab", u[o, v, v, o])

    I31_l2 = np.zeros((no, no, nv, nv))

    I31_l2 += np.einsum("kica,kjcb->ijab", I30_l2, l2)

    del I30_l2

    I35_l2 -= np.einsum("jiba->ijab", I31_l2)

    del I31_l2

    I33_l2 = np.zeros((no, no, nv, nv))

    I33_l2 += 2 * np.einsum("jiab->ijab", u[o, o, v, v])

    I33_l2 -= np.einsum("jiba->ijab", u[o, o, v, v])

    I34_l2 = np.zeros((no, nv))

    I34_l2 += np.einsum("bj,jiab->ia", t1, I33_l2)

    del I33_l2

    I35_l2 -= np.einsum("jb,ia->ijab", I34_l2, l1)

    I48_l2 = np.zeros((no, no))

    I48_l2 += np.einsum("ia,aj->ij", I34_l2, t1)

    I49_l2 = np.zeros((no, no))

    I49_l2 += np.einsum("ij->ij", I48_l2)

    del I48_l2

    I51_l2 = np.zeros((no, no, nv, nv))

    I51_l2 += np.einsum("ka,ijkb->ijab", I34_l2, I1_l2)

    del I1_l2

    del I34_l2

    I52_l2 += np.einsum("ijba->ijab", I51_l2)

    del I51_l2

    I35_l2 -= np.einsum("ia,jb->ijab", f[o, v], l1)

    rhs -= 2 * np.einsum("ijab->ijab", I35_l2)

    rhs += np.einsum("ijba->ijab", I35_l2)

    rhs += np.einsum("jiab->ijab", I35_l2)

    rhs -= 2 * np.einsum("jiba->ijab", I35_l2)

    del I35_l2

    I40_l2 = np.zeros((no, nv, nv, nv))

    I40_l2 -= np.einsum("iabc->iabc", u[o, v, v, v])

    I40_l2 += 2 * np.einsum("iacb->iabc", u[o, v, v, v])

    I41_l2 = np.zeros((nv, nv))

    I41_l2 += np.einsum("ci,iabc->ab", t1, I40_l2)

    del I40_l2

    I42_l2 -= np.einsum("ab->ab", I41_l2)

    del I41_l2

    I43_l2 = np.zeros((no, no, nv, nv))

    I43_l2 += np.einsum("ca,ijcb->ijab", I42_l2, l2)

    del I42_l2

    I52_l2 += np.einsum("jiba->ijab", I43_l2)

    del I43_l2

    I44_l2 = np.zeros((no, no, nv, nv))

    I44_l2 -= np.einsum("abji->ijab", t2)

    I44_l2 += 2 * np.einsum("baji->ijab", t2)

    I45_l2 = np.zeros((no, no))

    I45_l2 += np.einsum("kiab,kjab->ij", I44_l2, u[o, o, v, v])

    del I44_l2

    I49_l2 += np.einsum("ji->ij", I45_l2)

    del I45_l2

    I46_l2 = np.zeros((no, no, no, nv))

    I46_l2 += 2 * np.einsum("ijka->ijka", u[o, o, o, v])

    I46_l2 -= np.einsum("jika->ijka", u[o, o, o, v])

    I47_l2 = np.zeros((no, no))

    I47_l2 += np.einsum("ak,ikja->ij", t1, I46_l2)

    del I46_l2

    I49_l2 += np.einsum("ij->ij", I47_l2)

    del I47_l2

    I50_l2 = np.zeros((no, no, nv, nv))

    I50_l2 += np.einsum("ik,kjab->ijab", I49_l2, l2)

    del I49_l2

    I52_l2 += np.einsum("jiba->ijab", I50_l2)

    del I50_l2

    rhs -= np.einsum("ijab->ijab", I52_l2)

    rhs -= np.einsum("jiba->ijab", I52_l2)

    del I52_l2

    I53_l2 += np.einsum("jilk->ijkl", u[o, o, o, o])

    I53_l2 += np.einsum("ablk,jiab->ijkl", t2, u[o, o, v, v])

    rhs += np.einsum("jikl,klba->ijab", I53_l2, l2)

    del I53_l2

    I54_l2 += np.einsum("jiab,ablk->ijkl", l2, t2)

    rhs += np.einsum("jikl,klba->ijab", I54_l2, u[o, o, v, v])

    del I54_l2

    rhs += np.einsum("jicd,dcab->ijab", l2, u[v, v, v, v])

    rhs -= 2 * np.einsum("jiab->ijab", u[o, o, v, v])

    rhs += 4 * np.einsum("jiba->ijab", u[o, o, v, v])

    return rhs
