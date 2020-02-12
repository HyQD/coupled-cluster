import coupled_cluster.ccs.rhs_l as ccs_l
import coupled_cluster.ccd.rhs_l as ccd_l


def compute_l_1_amplitudes(F, W, t1, t2, l1, l2, o, v, np, out=None):
    """
    if out is None:
        out = np.zeros_like(l_1)
    """
    nocc = o.stop
    nvirt = v.stop - nocc
    tau0 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau0 += np.einsum("caik,cbjk->ijab", l2, t2, optimize=True)
    tOmega = np.zeros((nvirt, nocc))
    tOmega -= 2 * np.einsum("ijbc,jbac->ai", tau0, W[o, v, v, v], optimize=True)
    del tau0
    tau1 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau1 += np.einsum("acik,cbjk->ijab", l2, t2, optimize=True)
    tOmega -= 2 * np.einsum("ijbc,jbca->ai", tau1, W[o, v, v, v], optimize=True)
    del tau1
    tau2 = np.zeros((nocc, nocc, nocc, nocc))
    tau2 += np.einsum("baij,ablk->ijkl", l2, t2, optimize=True)
    tau23 = np.zeros((nocc, nocc, nocc, nvirt))
    tau23 -= 2 * np.einsum("al,ilkj->ijka", t1, tau2, optimize=True)
    tOmega += 2 * np.einsum("ijlk,klja->ai", tau2, W[o, o, o, v], optimize=True)
    del tau2
    tau3 = np.zeros((nocc, nocc, nocc, nvirt))
    tau3 += np.einsum("bk,baji->ijka", t1, l2, optimize=True)
    tau4 = np.zeros((nocc, nocc, nocc, nocc))
    tau4 += np.einsum("ak,ijla->ijkl", t1, tau3, optimize=True)
    tOmega += 2 * np.einsum("iljk,kjla->ai", tau4, W[o, o, o, v], optimize=True)
    del tau4
    tau27 = np.zeros((nocc, nocc, nocc, nvirt))
    tau27 += 2 * np.einsum("bajl,ilkb->ijka", t2, tau3, optimize=True)
    tau27 += 2 * np.einsum("abjl,likb->ijka", t2, tau3, optimize=True)
    tau31 = np.zeros((nocc, nvirt))
    tau31 += 2 * np.einsum("abkj,jkib->ia", t2, tau3, optimize=True)
    tau5 = np.zeros((nocc, nocc, nocc, nvirt))
    tau5 += np.einsum("bi,kjba->ijka", t1, W[o, o, v, v], optimize=True)
    tau6 = np.zeros((nocc, nocc, nocc, nvirt))
    tau6 += 2 * np.einsum("ijka->ijka", tau5, optimize=True)
    tau6 -= np.einsum("ikja->ijka", tau5, optimize=True)
    tau7 = np.zeros((nocc, nocc, nocc, nvirt))
    tau7 += np.einsum("ijka->ijka", tau5, optimize=True)
    tau13 = np.zeros((nocc, nocc, nocc, nvirt))
    tau13 += np.einsum("kjia->ijka", tau5, optimize=True)
    del tau5
    tau6 -= np.einsum("jkia->ijka", W[o, o, o, v], optimize=True)
    tau6 += 2 * np.einsum("kjia->ijka", W[o, o, o, v], optimize=True)
    tau15 = np.zeros((nocc, nocc, nocc, nvirt))
    tau15 -= np.einsum("balj,klib->ijka", t2, tau6, optimize=True)
    del tau6
    tau7 += np.einsum("kjia->ijka", W[o, o, o, v], optimize=True)
    tau15 += np.einsum("ablj,klib->ijka", t2, tau7, optimize=True)
    tau26 = np.zeros((nocc, nocc, nocc, nvirt))
    tau26 += np.einsum("ablk,ijlb->ijka", t2, tau7, optimize=True)
    del tau7
    tau8 = np.zeros((nocc, nvirt, nvirt, nvirt))
    tau8 += np.einsum("iabc->iabc", W[o, v, v, v], optimize=True)
    tau8 -= np.einsum("aj,ijbc->iabc", t1, W[o, o, v, v], optimize=True)
    tau15 -= np.einsum("bckj,iabc->ijka", t2, tau8, optimize=True)
    del tau8
    tau9 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau9 += 2 * np.einsum("jiab->ijab", W[o, o, v, v], optimize=True)
    tau9 -= np.einsum("jiba->ijab", W[o, o, v, v], optimize=True)
    tau10 = np.zeros((nocc, nvirt))
    tau10 += np.einsum("bj,jiab->ia", t1, tau9, optimize=True)
    tau10 += np.einsum("ia->ia", F[o, v], optimize=True)
    tau15 -= np.einsum("ib,bakj->ijka", tau10, t2, optimize=True)
    del tau10
    tau11 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau11 += np.einsum("ci,jabc->ijab", t1, W[o, v, v, v], optimize=True)
    tau12 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau12 += np.einsum("jiab->ijab", tau11, optimize=True)
    del tau11
    tau12 += np.einsum("iabj->ijab", W[o, v, v, o], optimize=True)
    tau15 -= np.einsum("bk,ijab->ijka", t1, tau12, optimize=True)
    tOmega -= 2 * np.einsum("jkba,kijb->ai", tau12, tau3, optimize=True)
    del tau12
    tau13 += np.einsum("ijka->ijka", W[o, o, o, v], optimize=True)
    tau14 = np.zeros((nocc, nocc, nocc, nocc))
    tau14 += np.einsum("al,ijka->ijkl", t1, tau13, optimize=True)
    del tau13
    tau14 += np.einsum("jilk->ijkl", W[o, o, o, o], optimize=True)
    tau15 += np.einsum("al,lijk->ijka", t1, tau14, optimize=True)
    del tau14
    tau15 -= np.einsum("iakj->ijka", W[o, v, o, o], optimize=True)
    tOmega += 2 * np.einsum("bajk,ijkb->ai", l2, tau15, optimize=True)
    del tau15
    tau16 = np.zeros((nocc, nvirt, nvirt, nvirt))
    tau16 += 2 * np.einsum("iabc->iabc", W[o, v, v, v], optimize=True)
    tau16 -= np.einsum("iacb->iabc", W[o, v, v, v], optimize=True)
    tau17 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau17 += np.einsum("ai,bj->ijab", l1, t1, optimize=True)
    tau17 += 2 * np.einsum("acik,bcjk->ijab", l2, t2, optimize=True)
    tOmega += np.einsum("jbca,ijbc->ai", tau16, tau17, optimize=True)
    del tau17
    tau18 = np.zeros((nocc, nocc, nocc, nvirt))
    tau18 += np.einsum("bi,bakj->ijka", l1, t2, optimize=True)
    tau23 += 2 * np.einsum("ijka->ijka", tau18, optimize=True)
    tau23 -= np.einsum("ikja->ijka", tau18, optimize=True)
    del tau18
    tau19 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau19 += np.einsum("ai,bj->ijab", t1, t1, optimize=True)
    tau19 -= 2 * np.einsum("abji->ijab", t2, optimize=True)
    tau19 += np.einsum("baji->ijab", t2, optimize=True)
    tau23 -= 2 * np.einsum("ljab,likb->ijka", tau19, tau3, optimize=True)
    tau36 = np.zeros((nocc, nocc))
    tau36 -= np.einsum("kjab,kiba->ij", tau19, W[o, o, v, v], optimize=True)
    del tau19
    tau20 = np.zeros((nocc, nocc))
    tau20 += np.einsum("ai,aj->ij", l1, t1, optimize=True)
    tau22 = np.zeros((nocc, nocc))
    tau22 += np.einsum("ij->ij", tau20, optimize=True)
    del tau20
    tau21 = np.zeros((nocc, nocc))
    tau21 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)
    tau22 += 2 * np.einsum("ij->ij", tau21, optimize=True)
    del tau21
    tau23 += 2 * np.einsum("aj,ik->ijka", t1, tau22, optimize=True)
    tOmega -= np.einsum("ijkb,jkba->ai", tau23, W[o, o, v, v], optimize=True)
    del tau23
    tau27 += np.einsum("aj,ik->ijka", t1, tau22, optimize=True)
    tOmega += np.einsum("ijkb,jkab->ai", tau27, W[o, o, v, v], optimize=True)
    del tau27
    tau31 += np.einsum("aj,ji->ia", t1, tau22, optimize=True)
    tOmega -= np.einsum("ja,ij->ai", F[o, v], tau22, optimize=True)
    tau24 = np.zeros((nocc, nvirt, nvirt, nvirt))
    tau24 += np.einsum("abic->iabc", W[v, v, o, v], optimize=True)
    tau24 += np.einsum("di,bacd->iabc", t1, W[v, v, v, v], optimize=True)
    tOmega += 2 * np.einsum("bcji,jbca->ai", l2, tau24, optimize=True)
    del tau24
    tau25 = np.zeros((nocc, nocc, nocc, nocc))
    tau25 += np.einsum("ai,jkla->ijkl", t1, W[o, o, o, v], optimize=True)
    tau26 += np.einsum("al,ijlk->ijka", t1, tau25, optimize=True)
    del tau25
    tau26 -= np.einsum("bi,jakb->ijka", t1, W[o, v, o, v], optimize=True)
    tOmega += 2 * np.einsum("abkj,jikb->ai", l2, tau26, optimize=True)
    del tau26
    tau28 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau28 += np.einsum("iajb->ijab", W[o, v, o, v], optimize=True)
    tau28 += np.einsum("cj,iacb->ijab", t1, W[o, v, v, v], optimize=True)
    tOmega -= 2 * np.einsum("jkba,ikjb->ai", tau28, tau3, optimize=True)
    del tau3
    del tau28
    tau29 = np.zeros((nvirt, nvirt))
    tau29 += np.einsum("ai,bi->ab", l1, t1, optimize=True)
    tau29 += 2 * np.einsum("acji,bcji->ab", l2, t2, optimize=True)
    tOmega += np.einsum("bc,ibac->ai", tau29, tau16, optimize=True)
    del tau16
    del tau29
    tau30 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau30 += 2 * np.einsum("abji->ijab", t2, optimize=True)
    tau30 -= np.einsum("baji->ijab", t2, optimize=True)
    tau31 -= np.einsum("bj,jiab->ia", l1, tau30, optimize=True)
    del tau30
    tau31 -= 2 * np.einsum("ai->ia", t1, optimize=True)
    tOmega -= np.einsum("jb,jiab->ai", tau31, tau9, optimize=True)
    del tau9
    del tau31
    tau32 = np.zeros((nocc, nocc, nocc, nvirt))
    tau32 -= np.einsum("ijka->ijka", W[o, o, o, v], optimize=True)
    tau32 += 2 * np.einsum("jika->ijka", W[o, o, o, v], optimize=True)
    tOmega -= np.einsum("jk,ikja->ai", tau22, tau32, optimize=True)
    del tau32
    del tau22
    tau33 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau33 += 2 * np.einsum("iabj->ijab", W[o, v, v, o], optimize=True)
    tau33 -= np.einsum("iajb->ijab", W[o, v, o, v], optimize=True)
    tOmega += np.einsum("bj,ijba->ai", l1, tau33, optimize=True)
    del tau33
    tau34 = np.zeros((nocc, nocc, nocc, nvirt))
    tau34 += 2 * np.einsum("ijka->ijka", W[o, o, o, v], optimize=True)
    tau34 -= np.einsum("jika->ijka", W[o, o, o, v], optimize=True)
    tau36 += np.einsum("ak,ikja->ij", t1, tau34, optimize=True)
    del tau34
    tau35 = np.zeros((nocc, nvirt))
    tau35 += np.einsum("ia->ia", F[o, v], optimize=True)
    tau35 += 2 * np.einsum("bj,ijab->ia", t1, W[o, o, v, v], optimize=True)
    tau36 += np.einsum("aj,ia->ij", t1, tau35, optimize=True)
    del tau35
    tau36 += np.einsum("ij->ij", F[o, o], optimize=True)
    tOmega -= np.einsum("aj,ij->ai", l1, tau36, optimize=True)
    del tau36
    tOmega += np.einsum("ba,bi->ai", F[v, v], l1, optimize=True)
    tOmega += 2 * np.einsum("ia->ai", F[o, v], optimize=True)
    return tOmega


def compute_l_2_amplitudes(F, W, t1, t2, l1, l2, o, v, np, out=None):
    if out is None:
        out = np.zeros_like(l_2)

    nocc = o.stop
    nvirt = v.stop - nocc

    tau0 = np.zeros((nocc, nocc, nocc, nocc))
    tau0 += np.einsum("baij,kmba->ijkm", t2, W[o, o, v, v], optimize=True)
    tOmegb = np.zeros((nvirt, nvirt, nocc, nocc))
    tOmegb += np.einsum("abmk,mkij->abij", l2, tau0, optimize=True)
    del tau0
    tau1 = np.zeros((nocc, nocc, nocc, nocc))
    tau1 += np.einsum("baij,bakm->ijkm", l2, t2, optimize=True)
    tOmegb += np.einsum("ijmk,mkab->abij", tau1, W[o, o, v, v], optimize=True)
    del tau1
    tau2 = np.zeros((nocc, nocc, nocc, nvirt))
    tau2 += np.einsum("bk,baji->ijka", t1, l2, optimize=True)
    tau3 = np.zeros((nocc, nocc, nocc, nocc))
    tau3 += np.einsum("ak,ijma->ijkm", t1, tau2, optimize=True)
    tOmegb += np.einsum("ijkm,mkba->abij", tau3, W[o, o, v, v], optimize=True)
    del tau3
    tau39 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau39 += np.einsum("jikc,kcab->ijab", tau2, W[o, v, v, v], optimize=True)
    del tau2
    tau55 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau55 += np.einsum("ijab->ijab", tau39, optimize=True)
    del tau39
    tau4 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau4 += np.einsum("ak,ijkb->ijab", l1, W[o, o, o, v], optimize=True)
    tau14 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau14 += np.einsum("ijab->ijab", tau4, optimize=True)
    del tau4
    tau5 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau5 += np.einsum("ak,kijb->ijab", t1, W[o, o, o, v], optimize=True)
    tau11 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau11 += np.einsum("jiab->ijab", tau5, optimize=True)
    del tau5
    tau6 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau6 += np.einsum("ci,jabc->ijab", t1, W[o, v, v, v], optimize=True)
    tau11 -= np.einsum("ijab->ijab", tau6, optimize=True)
    del tau6
    tau7 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau7 += 2 * np.einsum("jiab->ijab", W[o, o, v, v], optimize=True)
    tau7 -= np.einsum("jiba->ijab", W[o, o, v, v], optimize=True)
    tau8 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau8 += np.einsum("cbkj,kiac->ijab", t2, tau7, optimize=True)
    tau11 -= np.einsum("jiba->ijab", tau8, optimize=True)
    del tau8
    tau13 = np.zeros((nocc, nvirt))
    tau13 += np.einsum("bj,jiab->ia", t1, tau7, optimize=True)
    del tau7
    tau14 -= np.einsum("ai,jb->ijab", l1, tau13, optimize=True)
    del tau13
    tau9 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau9 += np.einsum("baji->ijab", t2, optimize=True)
    tau9 += np.einsum("ai,bj->ijab", t1, t1, optimize=True)
    tau10 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau10 += np.einsum("kjbc,kica->ijab", tau9, W[o, o, v, v], optimize=True)
    tau11 += np.einsum("jiba->ijab", tau10, optimize=True)
    del tau10
    tau19 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau19 += np.einsum("kjbc,kiac->ijab", tau9, W[o, o, v, v], optimize=True)
    del tau9
    tau20 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau20 += np.einsum("jiba->ijab", tau19, optimize=True)
    del tau19
    tau11 -= np.einsum("jabi->ijab", W[o, v, v, o], optimize=True)
    tau12 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau12 += np.einsum("cbkj,kica->ijab", l2, tau11, optimize=True)
    del tau11
    tau14 += 2 * np.einsum("jiba->ijab", tau12, optimize=True)
    del tau12
    tau14 -= np.einsum("ia,bj->ijab", F[o, v], l1, optimize=True)
    tOmegb -= np.einsum("ijab->abij", tau14, optimize=True)
    tOmegb += np.einsum("ijba->abij", tau14, optimize=True) / 2
    tOmegb += np.einsum("jiab->abij", tau14, optimize=True) / 2
    tOmegb -= np.einsum("jiba->abij", tau14, optimize=True)
    del tau14
    tau15 = np.zeros((nocc, nocc, nocc, nocc))
    tau15 += np.einsum("ai,jkma->ijkm", t1, W[o, o, o, v], optimize=True)
    tau16 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau16 += np.einsum("bamk,kijm->ijab", l2, tau15, optimize=True)
    del tau15
    tau28 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau28 -= np.einsum("ijab->ijab", tau16, optimize=True)
    del tau16
    tau17 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau17 += np.einsum("ak,ikjb->ijab", t1, W[o, o, o, v], optimize=True)
    tau20 += np.einsum("jiab->ijab", tau17, optimize=True)
    del tau17
    tau18 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau18 += np.einsum("ci,jacb->ijab", t1, W[o, v, v, v], optimize=True)
    tau20 -= np.einsum("ijab->ijab", tau18, optimize=True)
    del tau18
    tau20 -= np.einsum("jaib->ijab", W[o, v, o, v], optimize=True)
    tau21 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau21 += np.einsum("bckj,kica->ijab", l2, tau20, optimize=True)
    tau28 -= np.einsum("jiba->ijab", tau21, optimize=True)
    del tau21
    tau40 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau40 += np.einsum("cbkj,kica->ijab", l2, tau20, optimize=True)
    del tau20
    tau55 -= np.einsum("jiba->ijab", tau40, optimize=True)
    del tau40
    tau22 = np.zeros((nvirt, nvirt))
    tau22 += np.einsum("ia,bi->ab", F[o, v], t1, optimize=True)
    tau23 = np.zeros((nvirt, nvirt))
    tau23 -= np.einsum("ba->ab", tau22, optimize=True)
    del tau22
    tau23 += np.einsum("ab->ab", F[v, v], optimize=True)
    tau24 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau24 += np.einsum("ca,cbij->ijab", tau23, l2, optimize=True)
    del tau23
    tau28 -= np.einsum("jiab->ijab", tau24, optimize=True)
    del tau24
    tau25 = np.zeros((nocc, nocc))
    tau25 += np.einsum("ia,aj->ij", F[o, v], t1, optimize=True)
    tau26 = np.zeros((nocc, nocc))
    tau26 += np.einsum("ij->ij", tau25, optimize=True)
    del tau25
    tau26 += np.einsum("ij->ij", F[o, o], optimize=True)
    tau27 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau27 += np.einsum("ik,abkj->ijab", tau26, l2, optimize=True)
    del tau26
    tau28 += np.einsum("ijba->ijab", tau27, optimize=True)
    del tau27
    tOmegb -= np.einsum("ijba->abij", tau28, optimize=True)
    tOmegb -= np.einsum("jiab->abij", tau28, optimize=True)
    del tau28
    tau29 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau29 += np.einsum("ci,jcab->ijab", l1, W[o, v, v, v], optimize=True)
    tau38 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau38 -= np.einsum("ijab->ijab", tau29, optimize=True)
    del tau29
    tau30 = np.zeros((nvirt, nvirt))
    tau30 += np.einsum("ai,bi->ab", l1, t1, optimize=True)
    tau32 = np.zeros((nvirt, nvirt))
    tau32 += np.einsum("ab->ab", tau30, optimize=True)
    del tau30
    tau31 = np.zeros((nvirt, nvirt))
    tau31 += np.einsum("acji,bcji->ab", l2, t2, optimize=True)
    tau32 += 2 * np.einsum("ab->ab", tau31, optimize=True)
    del tau31
    tau33 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau33 += np.einsum("bc,ijca->ijab", tau32, W[o, o, v, v], optimize=True)
    del tau32
    tau38 += np.einsum("jiba->ijab", tau33, optimize=True)
    del tau33
    tau34 = np.zeros((nocc, nocc))
    tau34 += np.einsum("ai,aj->ij", l1, t1, optimize=True)
    tau36 = np.zeros((nocc, nocc))
    tau36 += np.einsum("ij->ij", tau34, optimize=True)
    del tau34
    tau35 = np.zeros((nocc, nocc))
    tau35 += np.einsum("baik,abkj->ij", l2, t2, optimize=True)
    tau36 += 2 * np.einsum("ij->ij", tau35, optimize=True)
    del tau35
    tau37 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau37 += np.einsum("jk,kiab->ijab", tau36, W[o, o, v, v], optimize=True)
    del tau36
    tau38 += np.einsum("jiba->ijab", tau37, optimize=True)
    del tau37
    tOmegb += np.einsum("ijab->abij", tau38, optimize=True) / 2
    tOmegb -= np.einsum("ijba->abij", tau38, optimize=True)
    tOmegb -= np.einsum("jiab->abij", tau38, optimize=True)
    tOmegb += np.einsum("jiba->abij", tau38, optimize=True) / 2
    del tau38
    tau41 = np.zeros((nocc, nvirt))
    tau41 += np.einsum("bj,ijab->ia", t1, W[o, o, v, v], optimize=True)
    tau42 = np.zeros((nvirt, nvirt))
    tau42 += np.einsum("ai,ib->ab", t1, tau41, optimize=True)
    tau47 = np.zeros((nvirt, nvirt))
    tau47 += 2 * np.einsum("ab->ab", tau42, optimize=True)
    del tau42
    tau49 = np.zeros((nocc, nocc))
    tau49 += np.einsum("ai,ja->ij", t1, tau41, optimize=True)
    del tau41
    tau53 = np.zeros((nocc, nocc))
    tau53 += 2 * np.einsum("ji->ij", tau49, optimize=True)
    del tau49
    tau43 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau43 += np.einsum("ai,bj->ijab", t1, t1, optimize=True)
    tau43 -= 2 * np.einsum("abji->ijab", t2, optimize=True)
    tau43 += np.einsum("baji->ijab", t2, optimize=True)
    tau44 = np.zeros((nvirt, nvirt))
    tau44 += np.einsum("ijcb,ijac->ab", tau43, W[o, o, v, v], optimize=True)
    tau47 -= np.einsum("ba->ab", tau44, optimize=True)
    del tau44
    tau50 = np.zeros((nocc, nocc))
    tau50 += np.einsum("kjba,kiab->ij", tau43, W[o, o, v, v], optimize=True)
    del tau43
    tau53 -= np.einsum("ij->ij", tau50, optimize=True)
    del tau50
    tau45 = np.zeros((nocc, nvirt, nvirt, nvirt))
    tau45 += 2 * np.einsum("iabc->iabc", W[o, v, v, v], optimize=True)
    tau45 -= np.einsum("iacb->iabc", W[o, v, v, v], optimize=True)
    tau46 = np.zeros((nvirt, nvirt))
    tau46 += np.einsum("ci,iacb->ab", t1, tau45, optimize=True)
    del tau45
    tau47 -= np.einsum("ab->ab", tau46, optimize=True)
    del tau46
    tau48 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau48 += np.einsum("ca,cbij->ijab", tau47, l2, optimize=True)
    del tau47
    tau55 += np.einsum("jiba->ijab", tau48, optimize=True)
    del tau48
    tau51 = np.zeros((nocc, nocc, nocc, nvirt))
    tau51 -= np.einsum("ijka->ijka", W[o, o, o, v], optimize=True)
    tau51 += 2 * np.einsum("jika->ijka", W[o, o, o, v], optimize=True)
    tau52 = np.zeros((nocc, nocc))
    tau52 += np.einsum("ak,kija->ij", t1, tau51, optimize=True)
    del tau51
    tau53 += np.einsum("ij->ij", tau52, optimize=True)
    del tau52
    tau54 = np.zeros((nocc, nocc, nvirt, nvirt))
    tau54 += np.einsum("ik,abkj->ijab", tau53, l2, optimize=True)
    del tau53
    tau55 += np.einsum("jiba->ijab", tau54, optimize=True)
    del tau54
    tOmegb -= np.einsum("ijab->abij", tau55, optimize=True)
    tOmegb -= np.einsum("jiba->abij", tau55, optimize=True)
    del tau55
    tau56 = np.zeros((nocc, nocc, nocc, nvirt))
    tau56 += np.einsum("bi,kjba->ijka", t1, W[o, o, v, v], optimize=True)
    tau57 = np.zeros((nocc, nocc, nocc, nocc))
    tau57 += np.einsum("am,kjia->ijkm", t1, tau56, optimize=True)
    del tau56
    tau57 += np.einsum("jimk->ijkm", W[o, o, o, o], optimize=True)
    tOmegb += np.einsum("bakm,jikm->abij", l2, tau57, optimize=True)
    del tau57
    tOmegb += np.einsum("dcji,dcba->abij", l2, W[v, v, v, v], optimize=True)
    tOmegb -= np.einsum("jiab->abij", W[o, o, v, v], optimize=True)
    tOmegb += 2 * np.einsum("jiba->abij", W[o, o, v, v], optimize=True)
    return tOmegb
