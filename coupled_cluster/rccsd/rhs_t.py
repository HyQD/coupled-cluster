# Labelling of the different terms comes from the book "Many-Body Methods in
# Chemistry and Physics" by I. Shavitt and R. J. Bartlett.


# Diagrams for CCSD T_1 amplitude equations

import coupled_cluster.ccs.rhs_t as ccs_t
import coupled_cluster.ccd.rhs_t as ccd_t


def compute_t_1_amplitudes(
    F, W, t1, t2, o, v, np, intermediates=None, out=None
):

    """
    if out is None:
        out = np.zeros_like(t_1)
    """

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    tau0 = np.zeros((nocc, nvirt, nvirt, nvirt), dtype=t1.dtype)
    tau0 += 2 * np.einsum("iabc->iabc", W[o, v, v, v], optimize=True)
    tau0 -= np.einsum("iacb->iabc", W[o, v, v, v], optimize=True)
    tau8 = np.zeros((nvirt, nvirt), dtype=t1.dtype)
    tau8 += np.einsum("ci,iacb->ab", t1, tau0, optimize=True)
    Omega = np.zeros((nvirt, nocc), dtype=t1.dtype)
    Omega += np.einsum("bcji,jabc->ai", t2, tau0, optimize=True)
    del tau0
    tau1 = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)
    tau1 += np.einsum("bi,kjba->ijka", t1, W[o, o, v, v], optimize=True)
    tau2 = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)
    tau2 += 2 * np.einsum("ijka->ijka", tau1, optimize=True)
    tau2 -= np.einsum("ikja->ijka", tau1, optimize=True)
    del tau1
    tau2 -= np.einsum("jkia->ijka", W[o, o, o, v], optimize=True)
    tau2 += 2 * np.einsum("kjia->ijka", W[o, o, o, v], optimize=True)
    Omega -= np.einsum("bajk,ijkb->ai", t2, tau2, optimize=True)
    del tau2
    tau3 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau3 += 2 * np.einsum("jiab->ijab", W[o, o, v, v], optimize=True)
    tau3 -= np.einsum("jiba->ijab", W[o, o, v, v], optimize=True)
    tau4 = np.zeros((nocc, nvirt), dtype=t1.dtype)
    tau4 += np.einsum("bj,jiab->ia", t1, tau3, optimize=True)
    tau5 = np.zeros((nocc, nvirt), dtype=t1.dtype)
    tau5 += np.einsum("ia->ia", tau4, optimize=True)
    del tau4
    tau10 = np.zeros((nocc, nocc), dtype=t1.dtype)
    tau10 += np.einsum("abkj,kiba->ij", t2, tau3, optimize=True)
    del tau3
    tau5 += np.einsum("ia->ia", F[o, v], optimize=True)
    tau10 += np.einsum("aj,ia->ij", t1, tau5, optimize=True)
    tau6 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau6 += 2 * np.einsum("abji->ijab", t2, optimize=True)
    tau6 -= np.einsum("baji->ijab", t2, optimize=True)
    Omega += np.einsum("jb,jiab->ai", tau5, tau6, optimize=True)
    del tau5
    del tau6
    tau7 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau7 += 2 * np.einsum("iabj->ijab", W[o, v, v, o], optimize=True)
    tau7 -= np.einsum("iajb->ijab", W[o, v, o, v], optimize=True)
    Omega += np.einsum("bj,jiab->ai", t1, tau7, optimize=True)
    del tau7
    tau8 += np.einsum("ab->ab", F[v, v], optimize=True)
    Omega += np.einsum("bi,ab->ai", t1, tau8, optimize=True)
    del tau8
    tau9 = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)
    tau9 += 2 * np.einsum("ijka->ijka", W[o, o, o, v], optimize=True)
    tau9 -= np.einsum("jika->ijka", W[o, o, o, v], optimize=True)
    tau10 += np.einsum("ak,ikja->ij", t1, tau9, optimize=True)
    del tau9
    tau10 += np.einsum("ij->ij", F[o, o], optimize=True)
    Omega -= np.einsum("aj,ji->ai", t1, tau10, optimize=True)
    del tau10
    Omega += np.einsum("ai->ai", F[v, o], optimize=True)
    return Omega


def compute_t_2_amplitudes(
    F, W, t1, t2, o, v, np, intermediates=None, out=None
):
    """
    if out is None:
        out = np.zeros_like(t_2)
    """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    tau0 = np.zeros((nocc, nvirt, nvirt, nvirt), dtype=t1.dtype)
    tau0 += np.einsum("di,badc->iabc", t1, W[v, v, v, v], optimize=True)
    Omegb = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t1.dtype)
    Omegb += np.einsum("cj,ibac->abij", t1, tau0, optimize=True)
    del tau0
    tau1 = np.zeros((nocc, nocc, nocc, nocc), dtype=t1.dtype)
    tau1 += np.einsum("baij,lkab->ijkl", t2, W[o, o, v, v], optimize=True)
    Omegb += np.einsum("balk,ijkl->abij", t2, tau1, optimize=True)
    del tau1
    tau2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau2 += np.einsum("ki,abjk->ijab", F[o, o], t2, optimize=True)
    tau16 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau16 -= np.einsum("ijab->ijab", tau2, optimize=True)
    del tau2
    tau3 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau3 += np.einsum("ac,bcij->ijab", F[v, v], t2, optimize=True)
    tau16 += np.einsum("ijab->ijab", tau3, optimize=True)
    del tau3
    tau4 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau4 += np.einsum("ci,abjc->ijab", t1, W[v, v, o, v], optimize=True)
    tau16 += np.einsum("ijab->ijab", tau4, optimize=True)
    del tau4
    tau5 = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)
    tau5 += np.einsum("bi,jakb->ijka", t1, W[o, v, o, v], optimize=True)
    tau6 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau6 += np.einsum("ak,ikjb->ijab", t1, tau5, optimize=True)
    del tau5
    tau16 -= np.einsum("ijab->ijab", tau6, optimize=True)
    del tau6
    tau7 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau7 += np.einsum("ci,jacb->ijab", t1, W[o, v, v, v], optimize=True)
    tau11 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau11 -= np.einsum("ijab->ijab", tau7, optimize=True)
    del tau7
    tau8 = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)
    tau8 += np.einsum("bi,kjba->ijka", t1, W[o, o, v, v], optimize=True)
    tau9 = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)
    tau9 += np.einsum("ijka->ijka", tau8, optimize=True)
    tau49 = np.zeros((nocc, nocc, nocc, nocc), dtype=t1.dtype)
    tau49 += np.einsum("ak,lija->ijkl", t1, tau8, optimize=True)
    del tau8
    tau9 += np.einsum("kjia->ijka", W[o, o, o, v], optimize=True)
    tau10 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau10 += np.einsum("bk,ikja->ijab", t1, tau9, optimize=True)
    tau11 += np.einsum("ijba->ijab", tau10, optimize=True)
    del tau10
    tau12 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau12 += np.einsum("bckj,ikac->ijab", t2, tau11, optimize=True)
    del tau11
    tau16 += np.einsum("jiba->ijab", tau12, optimize=True)
    del tau12
    tau26 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau26 += np.einsum("bk,ijka->ijab", t1, tau9, optimize=True)
    del tau9
    tau27 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau27 += np.einsum("ijba->ijab", tau26, optimize=True)
    del tau26
    tau13 = np.zeros((nocc, nocc, nocc, nocc), dtype=t1.dtype)
    tau13 += np.einsum("ai,jkla->ijkl", t1, W[o, o, o, v], optimize=True)
    tau14 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau14 += np.einsum("baji->ijab", t2, optimize=True)
    tau14 += np.einsum("ai,bj->ijab", t1, t1, optimize=True)
    tau15 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau15 += np.einsum("iklj,lkab->ijab", tau13, tau14, optimize=True)
    del tau13
    tau16 += np.einsum("ijba->ijab", tau15, optimize=True)
    del tau15
    Omegb += np.einsum("ijba->abij", tau16, optimize=True)
    Omegb += np.einsum("jiab->abij", tau16, optimize=True)
    del tau16
    tau17 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau17 -= np.einsum("jiab->ijab", W[o, o, v, v], optimize=True)
    tau17 += 2 * np.einsum("jiba->ijab", W[o, o, v, v], optimize=True)
    tau18 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau18 += np.einsum("kjbc,kica->ijab", tau14, tau17, optimize=True)
    tau23 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau23 += np.einsum("jiba->ijab", tau18, optimize=True)
    del tau18
    tau30 = np.zeros((nvirt, nvirt), dtype=t1.dtype)
    tau30 += np.einsum("ijcb,ijca->ab", tau14, tau17, optimize=True)
    tau32 = np.zeros((nvirt, nvirt), dtype=t1.dtype)
    tau32 += np.einsum("ba->ab", tau30, optimize=True)
    del tau30
    tau48 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau48 += 2 * np.einsum("caki,kjcb->ijab", t2, tau17, optimize=True)
    del tau17
    tau19 = np.zeros((nocc, nvirt, nvirt, nvirt), dtype=t1.dtype)
    tau19 += 2 * np.einsum("iabc->iabc", W[o, v, v, v], optimize=True)
    tau19 -= np.einsum("iacb->iabc", W[o, v, v, v], optimize=True)
    tau20 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau20 += np.einsum("cj,iabc->ijab", t1, tau19, optimize=True)
    tau23 -= np.einsum("jiab->ijab", tau20, optimize=True)
    del tau20
    tau31 = np.zeros((nvirt, nvirt), dtype=t1.dtype)
    tau31 += np.einsum("ci,iacb->ab", t1, tau19, optimize=True)
    del tau19
    tau32 -= np.einsum("ab->ab", tau31, optimize=True)
    del tau31
    tau21 = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)
    tau21 += 2 * np.einsum("ijka->ijka", W[o, o, o, v], optimize=True)
    tau21 -= np.einsum("jika->ijka", W[o, o, o, v], optimize=True)
    tau22 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau22 += np.einsum("bk,kija->ijab", t1, tau21, optimize=True)
    tau23 += np.einsum("jiba->ijab", tau22, optimize=True)
    del tau22
    tau24 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau24 += np.einsum("cbkj,ikac->ijab", t2, tau23, optimize=True)
    del tau23
    tau47 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau47 += np.einsum("jiba->ijab", tau24, optimize=True)
    del tau24
    tau36 = np.zeros((nocc, nocc), dtype=t1.dtype)
    tau36 += np.einsum("ak,ikja->ij", t1, tau21, optimize=True)
    del tau21
    tau40 = np.zeros((nocc, nocc), dtype=t1.dtype)
    tau40 += np.einsum("ij->ij", tau36, optimize=True)
    del tau36
    tau25 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau25 += np.einsum("ci,jabc->ijab", t1, W[o, v, v, v], optimize=True)
    tau27 -= np.einsum("ijab->ijab", tau25, optimize=True)
    tau28 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau28 += np.einsum("bckj,ikac->ijab", t2, tau27, optimize=True)
    del tau27
    tau47 -= np.einsum("jiba->ijab", tau28, optimize=True)
    del tau28
    tau43 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau43 += np.einsum("jiab->ijab", tau25, optimize=True)
    del tau25
    tau29 = np.zeros((nvirt, nvirt), dtype=t1.dtype)
    tau29 += np.einsum("ia,bi->ab", F[o, v], t1, optimize=True)
    tau32 += np.einsum("ba->ab", tau29, optimize=True)
    del tau29
    tau33 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau33 += np.einsum("ac,cbij->ijab", tau32, t2, optimize=True)
    del tau32
    tau47 += np.einsum("jiba->ijab", tau33, optimize=True)
    del tau33
    tau34 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau34 += 2 * np.einsum("jiab->ijab", W[o, o, v, v], optimize=True)
    tau34 -= np.einsum("jiba->ijab", W[o, o, v, v], optimize=True)
    tau35 = np.zeros((nocc, nocc), dtype=t1.dtype)
    tau35 += np.einsum("bakj,kiab->ij", t2, tau34, optimize=True)
    tau40 += np.einsum("ij->ij", tau35, optimize=True)
    del tau35
    tau37 = np.zeros((nocc, nvirt), dtype=t1.dtype)
    tau37 += np.einsum("bj,jiab->ia", t1, tau34, optimize=True)
    del tau34
    tau38 = np.zeros((nocc, nvirt), dtype=t1.dtype)
    tau38 += np.einsum("ia->ia", tau37, optimize=True)
    del tau37
    tau38 += np.einsum("ia->ia", F[o, v], optimize=True)
    tau39 = np.zeros((nocc, nocc), dtype=t1.dtype)
    tau39 += np.einsum("aj,ia->ij", t1, tau38, optimize=True)
    del tau38
    tau40 += np.einsum("ij->ij", tau39, optimize=True)
    del tau39
    tau41 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau41 += np.einsum("ki,abkj->ijab", tau40, t2, optimize=True)
    del tau40
    tau47 += np.einsum("jiba->ijab", tau41, optimize=True)
    del tau41
    tau42 = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)
    tau42 += np.einsum("cbij,kacb->ijka", t2, W[o, v, v, v], optimize=True)
    tau45 = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)
    tau45 += np.einsum("ikja->ijka", tau42, optimize=True)
    del tau42
    tau43 += np.einsum("iabj->ijab", W[o, v, v, o], optimize=True)
    tau44 = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)
    tau44 += np.einsum("bk,ijab->ijka", t1, tau43, optimize=True)
    del tau43
    tau45 += np.einsum("jkia->ijka", tau44, optimize=True)
    del tau44
    tau45 += np.einsum("jaik->ijka", W[o, v, o, o], optimize=True)
    tau46 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau46 += np.einsum("bk,ikja->ijab", t1, tau45, optimize=True)
    del tau45
    tau47 += np.einsum("ijba->ijab", tau46, optimize=True)
    del tau46
    Omegb -= np.einsum("ijab->abij", tau47, optimize=True)
    Omegb -= np.einsum("jiba->abij", tau47, optimize=True)
    del tau47
    tau48 += 2 * np.einsum("jabi->ijab", W[o, v, v, o], optimize=True)
    tau48 -= np.einsum("jaib->ijab", W[o, v, o, v], optimize=True)
    Omegb += np.einsum("caki,jkbc->abij", t2, tau48, optimize=True)
    del tau48
    tau49 += np.einsum("jilk->ijkl", W[o, o, o, o], optimize=True)
    Omegb += np.einsum("klab,klij->abij", tau14, tau49, optimize=True)
    del tau49
    del tau14
    tau50 = np.zeros((nocc, nvirt, nvirt, nvirt), dtype=t1.dtype)
    tau50 += np.einsum("aj,ijbc->iabc", t1, W[o, o, v, v], optimize=True)
    tau51 = np.zeros((nvirt, nvirt, nvirt, nvirt), dtype=t1.dtype)
    tau51 += np.einsum("ai,ibcd->abcd", t1, tau50, optimize=True)
    del tau50
    tau51 += np.einsum("badc->abcd", W[v, v, v, v], optimize=True)
    Omegb += np.einsum("cdji,bacd->abij", t2, tau51, optimize=True)
    del tau51
    tau52 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau52 -= np.einsum("jabi->ijab", W[o, v, v, o], optimize=True)
    tau52 += np.einsum("caik,kjcb->ijab", t2, W[o, o, v, v], optimize=True)
    Omegb += np.einsum("bckj,ikac->abij", t2, tau52, optimize=True)
    del tau52
    tau53 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau53 -= np.einsum("jaib->ijab", W[o, v, o, v], optimize=True)
    tau53 += np.einsum("caik,kjbc->ijab", t2, W[o, o, v, v], optimize=True)
    Omegb += np.einsum("ackj,ikbc->abij", t2, tau53, optimize=True)
    del tau53
    Omegb -= np.einsum("bcjk,kaic->abij", t2, W[o, v, o, v], optimize=True)
    Omegb -= np.einsum("caik,kbcj->abij", t2, W[o, v, v, o], optimize=True)
    Omegb -= np.einsum("cbik,kajc->abij", t2, W[o, v, o, v], optimize=True)
    Omegb += 2 * np.einsum("bcjk,kaci->abij", t2, W[o, v, v, o], optimize=True)
    Omegb += np.einsum("baji->abij", W[v, v, o, o], optimize=True)
    return Omegb
