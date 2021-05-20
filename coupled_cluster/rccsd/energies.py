from coupled_cluster.cc_helper import compute_reference_energy
from coupled_cluster.ccsd.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)
from coupled_cluster.ccd.energies import (
    compute_lagrangian_functional as ccd_functional,
)
from opt_einsum import contract


def compute_rccsd_ground_state_energy(f, u, t_1, t_2, o, v, np):

    # energy = compute_reference_energy(f, u, o, v, np=np)
    """
    Reference energy missing
    """
    energy = 0

    energy += compute_ground_state_energy_correction(
        f, u, t_1, t_2, o, v, np=np
    )

    return energy


def compute_ground_state_energy_correction(f, u, t_1, t_2, o, v, np):
    """

    f^{i}_{a} t^{a}_{i}
    + (0.25) u^{ij}_{ab} t^{ab}_{ij}
    + (0.5) t^{ij}_{ab} t^{a}_{i} t^{b}_{j}

    """

    e_corr = 2 * contract("ia,ai->", f[o, v], t_1)

    e_corr += 2 * contract("abij,ijab->", t_2, u[o, o, v, v])
    e_corr -= contract("abij,ijba->", t_2, u[o, o, v, v])

    e_corr += 2 * contract(
        "ai,bj,ijab->", t_1, t_1, u[o, o, v, v], optimize=True
    )
    e_corr -= contract("ai,bj,ijba->", t_1, t_1, u[o, o, v, v], optimize=True)

    return e_corr


def compute_time_dependent_energy(f, u, t_1, t_2, l_1, l_2, o, v, np):
    energy = (
        2 * np.trace(f[o, o])
        - 2 * np.trace(np.trace(u[o, o, o, o], axis1=1, axis2=3))
        + np.trace(np.trace(u[o, o, o, o], axis1=1, axis2=2))
    )
    energy += lagrangian_functional(f, u, t_1, t_2, l_1, l_2, o, v, np=np)
    return energy


def lagrangian_functional(f, u, t1, t2, l1, l2, o, v, np, test=False):

    no = o.stop
    nv = v.stop - no

    I0_l1 = np.zeros((no, no), dtype=t1.dtype)

    I0_l1 += contract("ia,aj->ij", f[o, v], t1)

    I1_l1 = np.zeros((no, no), dtype=t1.dtype)

    I1_l1 += contract("ia,aj->ij", l1, t1)

    I7_l1 = np.zeros((no, no), dtype=t1.dtype)

    I7_l1 += contract("ij->ij", I1_l1)

    lagrangian = 0 + 0j

    lagrangian -= contract("ij,ji->", I0_l1, I1_l1)

    lagrangian -= contract("ji,ij->", I1_l1, f[o, o])

    del I1_l1

    I2_l1 = np.zeros((no, no), dtype=t1.dtype)

    I2_l1 += contract("kiba,bakj->ij", l2, t2)

    I7_l1 += contract("ij->ij", I2_l1)

    I8_l1 = np.zeros((no, nv), dtype=t1.dtype)

    I8_l1 += contract("ji,aj->ia", I7_l1, t1)

    lagrangian -= contract("ji,ij->", I2_l1, f[o, o])

    lagrangian -= contract("ij,ji->", I0_l1, I2_l1)

    del I0_l1

    del I2_l1

    I3_l1 = np.zeros((nv, nv), dtype=t1.dtype)

    I3_l1 += contract("jica,cbji->ab", l2, t2)

    I33_l1 = np.zeros((nv, nv), dtype=t1.dtype)

    I33_l1 += contract("ab->ab", I3_l1)

    I36_l1 = np.zeros((no, nv), dtype=t1.dtype)

    I36_l1 += 4 * contract("bc,ibac->ia", I3_l1, u[o, v, v, v])

    lagrangian += contract("ba,ba->", I3_l1, f[v, v])

    I4_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I4_l1 += contract("kica,bcjk->ijab", l2, t2)

    I23_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I23_l1 += contract("ijab->ijab", I4_l1)

    I5_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I5_l1 -= contract("abji->ijab", t2)

    I5_l1 += contract("baji->ijab", t2)

    I11_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I11_l1 -= contract("kjcb,kica->ijab", I4_l1, I5_l1)

    del I4_l1

    del I5_l1

    I6_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)

    I6_l1 += contract("bk,jiba->ijka", t1, l2)

    I8_l1 += contract("jkib,abkj->ia", I6_l1, t2)

    I32_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)

    I32_l1 += 2 * contract("iljb,bakl->ijka", I6_l1, t2)

    I36_l1 -= 2 * contract("kijb,jbak->ia", I6_l1, u[o, v, v, o])

    I8_l1 -= contract("ai->ia", t1)

    I11_l1 += contract("jb,ai->ijab", I8_l1, t1)

    del I8_l1

    I9_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I9_l1 -= contract("abji->ijab", t2)

    I9_l1 += 2 * contract("baji->ijab", t2)

    I10_l1 = np.zeros((no, nv), dtype=t1.dtype)

    I10_l1 += contract("jb,jiba->ia", l1, I9_l1)

    I11_l1 -= contract("ia,bj->ijab", I10_l1, t1)

    del I10_l1

    I14_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I14_l1 += 2 * contract("kica,kbcj->ijab", I9_l1, u[o, v, v, o])

    I22_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I22_l1 += contract("kjcb,kica->ijab", I9_l1, l2)

    I36_l1 += 2 * contract("ijbc,jbca->ia", I22_l1, u[o, v, v, v])

    del I22_l1

    I39_l1 = np.zeros((no, nv), dtype=t1.dtype)

    I39_l1 += contract("jibc,jabc->ia", I9_l1, u[o, v, v, v])

    I39_l1 -= contract("jkab,jkib->ia", I9_l1, u[o, o, o, v])

    del I9_l1

    I11_l1 -= contract("baji->ijab", t2)

    I12_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I12_l1 -= contract("jiab->ijab", u[o, o, v, v])

    I12_l1 += 2 * contract("jiba->ijab", u[o, o, v, v])

    lagrangian -= contract("ijab,ijab->", I11_l1, I12_l1)

    del I11_l1

    del I12_l1

    I13_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I13_l1 += contract("acki,kjcb->ijab", t2, u[o, o, v, v])

    I14_l1 += contract("ikac,bckj->ijab", I13_l1, t2)

    del I13_l1

    I14_l1 += contract("baji->ijab", u[v, v, o, o])

    I14_l1 -= 2 * contract("caki,kbjc->ijab", t2, u[o, v, o, v])

    I14_l1 += contract("cdji,bacd->ijab", t2, u[v, v, v, v], optimize=True)

    lagrangian += contract("ijab,ijab->", I14_l1, l2) / 2

    del I14_l1

    I15_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I15_l1 += 2 * contract("jaib->ijab", u[o, v, o, v])

    I15_l1 -= contract("acki,kjbc->ijab", t2, u[o, o, v, v])

    I16_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I16_l1 += contract("kjcb,kiac->ijab", I15_l1, l2)

    del I15_l1

    I16_l1 += 2 * contract("ia,jb->ijab", f[o, v], l1)

    lagrangian -= contract("ijab,abji->", I16_l1, t2) / 2

    del I16_l1

    I17_l1 = np.zeros((no, no, no, no), dtype=t1.dtype)

    I17_l1 += contract("abij,lkba->ijkl", t2, u[o, o, v, v])

    I18_l1 = np.zeros((no, no, no, no), dtype=t1.dtype)

    I18_l1 += contract("lkji->ijkl", I17_l1)

    I31_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)

    I31_l1 -= contract("al,jkli->ijka", t1, I17_l1)

    del I17_l1

    I18_l1 += contract("jilk->ijkl", u[o, o, o, o])

    I19_l1 = np.zeros((no, no, no, no), dtype=t1.dtype)

    I19_l1 += contract("ijba,ablk->ijkl", l2, t2)

    I32_l1 += contract("al,ilkj->ijka", t1, I19_l1)

    I36_l1 += 2 * contract("jikl,klja->ia", I19_l1, u[o, o, o, v])

    lagrangian += contract("ijkl,lkji->", I18_l1, I19_l1) / 2

    del I18_l1

    del I19_l1

    I20_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)

    I20_l1 += contract("ib,abjk->ijka", l1, t2)

    I32_l1 -= 4 * contract("ijka->ijka", I20_l1)

    I36_l1 += contract("ikjb,jkab->ia", I32_l1, u[o, o, v, v])

    del I32_l1

    I36_l1 += 2 * contract("ijkb,kjba->ia", I20_l1, u[o, o, v, v])

    del I20_l1

    I21_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I21_l1 += contract("ci,jacb->ijab", t1, u[o, v, v, v])

    I36_l1 -= 2 * contract("kjba,ikjb->ia", I21_l1, I6_l1)

    del I21_l1

    del I6_l1

    I23_l1 += contract("kiac,cbjk->ijab", l2, t2)

    I36_l1 -= 2 * contract("ijbc,jbac->ia", I23_l1, u[o, v, v, v])

    del I23_l1

    I24_l1 = np.zeros((no, nv, nv, nv), dtype=t1.dtype)

    I24_l1 += 2 * contract("abic->iabc", u[v, v, o, v])

    I24_l1 += contract("di,abdc->iabc", t1, u[v, v, v, v])

    I36_l1 += contract("jbca,jibc->ia", I24_l1, l2)

    del I24_l1

    I25_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)

    I25_l1 += contract("bi,jkab->ijka", t1, u[o, o, v, v])

    I26_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)

    I26_l1 += 2 * contract("ijka->ijka", I25_l1)

    I26_l1 -= contract("ikja->ijka", I25_l1)

    I27_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)

    I27_l1 += contract("ijka->ijka", I25_l1)

    I28_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)

    I28_l1 += contract("kjia->ijka", I25_l1)

    del I25_l1

    I26_l1 -= contract("jkia->ijka", u[o, o, o, v])

    I26_l1 += 2 * contract("kjia->ijka", u[o, o, o, v])

    I30_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)

    I30_l1 -= 2 * contract("kljb,bali->ijka", I26_l1, t2)

    del I26_l1

    I27_l1 += contract("kjia->ijka", u[o, o, o, v])

    I30_l1 += 2 * contract("kljb,abli->ijka", I27_l1, t2)

    del I27_l1

    I28_l1 += 2 * contract("ijka->ijka", u[o, o, o, v])

    I29_l1 = np.zeros((no, no, no, no), dtype=t1.dtype)

    I29_l1 += contract("al,ijka->ijkl", t1, I28_l1)

    del I28_l1

    I29_l1 += contract("jilk->ijkl", u[o, o, o, o])

    I30_l1 += contract("al,ljik->ijka", t1, I29_l1)

    del I29_l1

    I36_l1 += contract("jikb,jkba->ia", I30_l1, l2)

    del I30_l1

    I31_l1 += 2 * contract("iakj->ijka", u[o, v, o, o])

    I31_l1 += 2 * contract("ib,bakj->ijka", f[o, v], t2)

    I31_l1 += 2 * contract("bj,iakb->ijka", t1, u[o, v, o, v])

    I31_l1 -= 2 * contract("ablk,lijb->ijka", t2, u[o, o, o, v])

    I31_l1 += 2 * contract("bckj,iabc->ijka", t2, u[o, v, v, v])

    I36_l1 -= contract("ijkb,kjab->ia", I31_l1, l2)

    del I31_l1

    I33_l1 += contract("ia,bi->ab", l1, t1)

    I36_l1 -= 2 * contract("bc,ibca->ia", I33_l1, u[o, v, v, v])

    del I33_l1

    I34_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I34_l1 += 2 * contract("iabj->ijab", u[o, v, v, o])

    I34_l1 -= contract("iajb->ijab", u[o, v, o, v])

    I36_l1 += 2 * contract("jb,ijba->ia", l1, I34_l1)

    del I34_l1

    I35_l1 = np.zeros((nv, nv), dtype=t1.dtype)

    I35_l1 += contract("ab->ab", f[v, v])

    I35_l1 += 2 * contract("ci,iacb->ab", t1, u[o, v, v, v])

    I36_l1 += 2 * contract("ba,ib->ia", I35_l1, l1)

    del I35_l1

    I36_l1 += 4 * contract("ia->ia", f[o, v])

    lagrangian += contract("ia,ai->", I36_l1, t1) / 2

    del I36_l1

    I37_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I37_l1 += 2 * contract("jiab->ijab", u[o, o, v, v])

    I37_l1 -= contract("jiba->ijab", u[o, o, v, v])

    I38_l1 = np.zeros((nv, nv), dtype=t1.dtype)

    I38_l1 += contract("ijcb,acij->ab", I37_l1, t2)

    lagrangian -= contract("ab,ab->", I38_l1, I3_l1)

    del I3_l1

    del I38_l1

    I41_l1 = np.zeros((no, no), dtype=t1.dtype)

    I41_l1 += contract("kiab,bakj->ij", I37_l1, t2)

    del I37_l1

    I39_l1 += contract("ai->ia", f[v, o])

    I39_l1 += 2 * contract("jb,abij->ia", f[o, v], t2)

    lagrangian += contract("ia,ia->", I39_l1, l1)

    del I39_l1

    I40_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)

    I40_l1 += 2 * contract("ijka->ijka", u[o, o, o, v])

    I40_l1 -= contract("jika->ijka", u[o, o, o, v])

    I41_l1 += contract("ak,ikja->ij", t1, I40_l1)

    del I40_l1

    lagrangian -= contract("ji,ij->", I41_l1, I7_l1)

    del I41_l1

    del I7_l1

    return lagrangian
