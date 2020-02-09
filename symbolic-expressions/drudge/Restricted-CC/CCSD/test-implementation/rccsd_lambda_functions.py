import numpy as np


def l1_rhs(f, u, l1, l2, t1, t2):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    o, v = slice(0, nocc), slice(nocc, nvirt + nocc)

    rhs = np.zeros((nocc, nvirt))

    rhs += 2 * np.einsum("ia->ia", f[o, v])

    rhs -= 2 * np.einsum("bj,ijba->ia", t1, u[o, o, v, v])

    rhs += 4 * np.einsum("bj,ijab->ia", t1, u[o, o, v, v])

    rhs += np.einsum("ba,ib->ia", f[v, v], l1)

    rhs += 2 * np.einsum(
        "dj,ijbc,bcad->ia", t1, l2, u[v, v, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ijbc,bdkj,kcad->ia", l2, t2, u[o, v, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ijbc,cdjk,kbad->ia", l2, t2, u[o, v, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ijbc,cdkj,bkad->ia", l2, t2, u[v, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "jkab,cdjk,ibcd->ia", l2, t2, u[o, v, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "jkbc,bdjk,icda->ia", l2, t2, u[o, v, v, v], optimize=True
    )

    rhs += 4 * np.einsum(
        "ijbc,cdjk,bkad->ia", l2, t2, u[v, o, v, v], optimize=True
    )

    rhs += 4 * np.einsum(
        "jkbc,bdjk,icad->ia", l2, t2, u[o, v, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "bk,dj,ijbc,kcad->ia", t1, t1, l2, u[o, v, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "cj,dk,jkab,ibcd->ia", t1, t1, l2, u[o, v, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ck,dj,ijbc,bkad->ia", t1, t1, l2, u[v, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "bm,ijbc,cdjk,mkad->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "cm,jkbc,bdjk,imad->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "dm,ijbc,bckj,kmad->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "dm,jkab,bckj,imcd->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "dj,jkab,bckm,imdc->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "dk,jkbc,bcjm,imad->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "bm,ijbc,cdjk,kmad->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "bm,ijbc,cdkj,mkad->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "bm,jkab,cdjk,imcd->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,ijbc,bdkj,kmad->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,jkbc,bdjk,imda->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "dm,ijbc,bckj,mkad->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "dm,jkab,bckj,imdc->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "dj,ijbc,bckm,kmad->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "dj,jkab,bcmk,imdc->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "dj,jkab,bckm,imcd->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "dk,jkab,bcmj,imcd->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "dk,jkbc,bcjm,imda->ia", t1, l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "bm,cj,dk,jkab,imcd->ia", t1, t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "bk,cm,dj,ijbc,kmad->ia", t1, t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum("ib,bckj,jkac->ia", l1, t2, u[o, o, v, v], optimize=True)

    rhs += np.einsum("ja,bckj,ikbc->ia", l1, t2, u[o, o, v, v], optimize=True)

    rhs += np.einsum("jb,bckj,ikca->ia", l1, t2, u[o, o, v, v], optimize=True)

    rhs -= 2 * np.einsum("ib,jkac,bcjk->ia", f[o, v], l2, t2, optimize=True)

    rhs -= 2 * np.einsum("ja,ikbc,bcjk->ia", f[o, v], l2, t2, optimize=True)

    rhs -= 2 * np.einsum(
        "ib,bcjk,jkac->ia", l1, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ja,bcjk,ikbc->ia", l1, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "jb,bcjk,ikca->ia", l1, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "jb,bckj,ikac->ia", l1, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "bk,ijbc,kcaj->ia", t1, l2, u[o, v, v, o], optimize=True
    )

    rhs -= 2 * np.einsum(
        "cj,jkab,ibck->ia", t1, l2, u[o, v, v, o], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ck,ijbc,bkaj->ia", t1, l2, u[v, o, v, o], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ck,jkab,ibjc->ia", t1, l2, u[o, v, o, v], optimize=True
    )

    rhs += 4 * np.einsum(
        "jb,bcjk,ikac->ia", l1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "ib,bk,cj,jkac->ia", l1, t1, t1, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "ja,bk,cj,ikbc->ia", l1, t1, t1, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "jb,bk,cj,ikca->ia", l1, t1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ib,bj,ck,jkac->ia", l1, t1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ja,bj,ck,ikbc->ia", l1, t1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "jb,bk,cj,ikac->ia", l1, t1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "jkab,bckm,imjc->ia", l2, t2, u[o, o, o, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "jkbc,bcjm,imak->ia", l2, t2, u[o, o, v, o], optimize=True
    )

    rhs += 2 * np.einsum(
        "ijbc,bckm,kmaj->ia", l2, t2, u[o, o, v, o], optimize=True
    )

    rhs += 2 * np.einsum(
        "jkab,bcmj,imck->ia", l2, t2, u[o, o, v, o], optimize=True
    )

    rhs += 2 * np.einsum(
        "jkab,bcmk,imjc->ia", l2, t2, u[o, o, o, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "jkab,bckm,imcj->ia", l2, t2, u[o, o, v, o], optimize=True
    )

    rhs += 2 * np.einsum(
        "jkbc,bcjm,imka->ia", l2, t2, u[o, o, o, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "bm,cj,jkab,imck->ia", t1, t1, l2, u[o, o, v, o], optimize=True
    )

    rhs += 2 * np.einsum(
        "bm,ck,jkab,imjc->ia", t1, t1, l2, u[o, o, o, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "bk,cm,ijbc,kmaj->ia", t1, t1, l2, u[o, o, v, o], optimize=True
    )

    rhs -= np.einsum("ij,ja->ia", f[o, o], l1)

    rhs += 2 * np.einsum("ijbc,bcaj->ia", l2, u[v, v, v, o], optimize=True)

    rhs -= np.einsum("ib,cj,jbac->ia", l1, t1, u[o, v, v, v], optimize=True)

    rhs -= np.einsum("jb,cj,ibca->ia", l1, t1, u[o, v, v, v], optimize=True)

    rhs += 2 * np.einsum("ib,cj,bjac->ia", l1, t1, u[v, o, v, v], optimize=True)

    rhs += 2 * np.einsum("jb,cj,ibac->ia", l1, t1, u[o, v, v, v], optimize=True)

    rhs -= np.einsum("jb,ibja->ia", l1, u[o, v, o, v])

    rhs += 2 * np.einsum("jb,ibaj->ia", l1, u[o, v, v, o])

    rhs -= np.einsum("ib,ja,bj->ia", f[o, v], l1, t1, optimize=True)

    rhs -= np.einsum("ja,ib,bj->ia", f[o, v], l1, t1, optimize=True)

    rhs -= 2 * np.einsum("jkab,ibjk->ia", l2, u[o, v, o, o])

    rhs += np.einsum("ja,bk,ikbj->ia", l1, t1, u[o, o, v, o], optimize=True)

    rhs += np.einsum("jb,bk,ikja->ia", l1, t1, u[o, o, o, v], optimize=True)

    rhs -= 2 * np.einsum("ja,bk,ikjb->ia", l1, t1, u[o, o, o, v], optimize=True)

    rhs -= 2 * np.einsum("jb,bk,ikaj->ia", l1, t1, u[o, o, v, o], optimize=True)

    rhs += 2 * np.einsum(
        "bm,jkab,imjk->ia", t1, l2, u[o, o, o, o], optimize=True
    )

    return rhs


def l2_rhs(f, u, l1, l2, t1, t2):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    o, v = slice(0, nocc), slice(nocc, nvirt + nocc)

    rhs = np.zeros((nocc, nocc, nvirt, nvirt))

    rhs -= 2 * np.einsum("ijba->ijab", u[o, o, v, v])

    rhs += 4 * np.einsum("ijab->ijab", u[o, o, v, v])

    rhs -= np.einsum("ic,cjba->ijab", l1, u[v, o, v, v])

    rhs -= np.einsum("jc,icba->ijab", l1, u[o, v, v, v])

    rhs += 2 * np.einsum("ca,ijcb->ijab", f[v, v], l2)

    rhs += 2 * np.einsum("cb,ijac->ijab", f[v, v], l2)

    rhs += 2 * np.einsum("ic,cjab->ijab", l1, u[v, o, v, v])

    rhs += 2 * np.einsum("jc,icab->ijab", l1, u[o, v, v, v])

    rhs -= np.einsum("ib,ja->ijab", f[o, v], l1)

    rhs -= np.einsum("ja,ib->ijab", f[o, v], l1)

    rhs += 2 * np.einsum("ia,jb->ijab", f[o, v], l1)

    rhs += 2 * np.einsum("jb,ia->ijab", f[o, v], l1)

    rhs -= 2 * np.einsum(
        "ck,ijcd,kdab->ijab", t1, l2, u[o, v, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "dk,ijac,kcbd->ijab", t1, l2, u[o, v, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "dk,ijcb,kcad->ijab", t1, l2, u[o, v, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "dk,ijcd,ckab->ijab", t1, l2, u[v, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "dk,ikac,cjbd->ijab", t1, l2, u[v, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "dk,ikbc,cjda->ijab", t1, l2, u[v, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "dk,ikcb,cjad->ijab", t1, l2, u[v, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "dk,kjac,icdb->ijab", t1, l2, u[o, v, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "dk,kjca,icbd->ijab", t1, l2, u[o, v, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "dk,kjcb,icda->ijab", t1, l2, u[o, v, v, v], optimize=True
    )

    rhs += 4 * np.einsum(
        "dk,ijac,ckbd->ijab", t1, l2, u[v, o, v, v], optimize=True
    )

    rhs += 4 * np.einsum(
        "dk,ijcb,ckad->ijab", t1, l2, u[v, o, v, v], optimize=True
    )

    rhs += 4 * np.einsum(
        "dk,ikac,cjdb->ijab", t1, l2, u[v, o, v, v], optimize=True
    )

    rhs += 4 * np.einsum(
        "dk,kjcb,icad->ijab", t1, l2, u[o, v, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "cm,ikab,mjck->ijab", t1, l2, u[o, o, v, o], optimize=True
    )

    rhs -= 4 * np.einsum(
        "cm,ikac,mjkb->ijab", t1, l2, u[o, o, o, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "cm,kjab,imkc->ijab", t1, l2, u[o, o, o, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "cm,kjcb,imak->ijab", t1, l2, u[o, o, v, o], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,ikab,mjkc->ijab", t1, l2, u[o, o, o, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,ikac,mjbk->ijab", t1, l2, u[o, o, v, o], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,ikbc,mjka->ijab", t1, l2, u[o, o, o, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,ikcb,mjak->ijab", t1, l2, u[o, o, v, o], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,kmab,ijkc->ijab", t1, l2, u[o, o, o, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,kjab,imck->ijab", t1, l2, u[o, o, v, o], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,kjac,imkb->ijab", t1, l2, u[o, o, o, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,kjca,imbk->ijab", t1, l2, u[o, o, v, o], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,kjcb,imka->ijab", t1, l2, u[o, o, o, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ck,kmab,ijcm->ijab", t1, l2, u[o, o, v, o], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ijac,cdkm,kmbd->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ijcb,cdkm,kmad->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ikab,cdmk,mjcd->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ikac,cdmk,mjdb->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ikac,cdkm,mjbd->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ikbc,cdkm,mjda->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ikcd,cdmk,mjab->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kmac,cdmk,ijdb->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kmbc,cdmk,ijad->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kjab,cdkm,imcd->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kjca,cdkm,imbd->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kjcb,cdmk,imad->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kjcb,cdkm,imda->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kjcd,cdkm,imab->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ijac,cdmk,kmbd->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ijcb,cdmk,kmad->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ijcd,cdkm,kmab->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ikab,cdkm,mjcd->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ikac,cdmk,mjbd->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ikbc,cdmk,mjda->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ikbc,cdkm,mjad->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ikcb,cdmk,mjad->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ikcd,cdmk,mjba->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kmab,cdkm,ijcd->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kmac,cdmk,ijbd->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kmbc,cdmk,ijda->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kjab,cdmk,imcd->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kjac,cdmk,imdb->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kjca,cdmk,imbd->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kjca,cdkm,imdb->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kjcb,cdmk,imda->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kjcd,cdkm,imba->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 8 * np.einsum(
        "ikac,cdkm,mjdb->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 8 * np.einsum(
        "kjcb,cdkm,imad->ijab", l2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "cm,dk,ikab,mjcd->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "cm,dk,ikac,mjdb->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "cm,dk,kjcb,imad->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ck,dm,ijac,kmbd->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ck,dm,ijcb,kmad->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ck,dm,kjab,imcd->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,dk,ijac,kmbd->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,dk,ijcb,kmad->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,dk,ikac,mjbd->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,dk,ikbc,mjda->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,dk,ikcb,mjad->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,dk,kjab,imcd->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,dk,kjac,imdb->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,dk,kjca,imbd->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "cm,dk,kjcb,imda->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ck,dm,ijcd,kmab->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ck,dm,ikab,mjcd->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ck,dm,kmab,ijcd->ijab", t1, t1, l2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum("ikac,cjbk->ijab", l2, u[v, o, v, o], optimize=True)

    rhs -= 2 * np.einsum("ikbc,cjka->ijab", l2, u[v, o, o, v], optimize=True)

    rhs -= 2 * np.einsum("ikcb,cjak->ijab", l2, u[v, o, v, o], optimize=True)

    rhs -= 2 * np.einsum("kjac,ickb->ijab", l2, u[o, v, o, v], optimize=True)

    rhs -= 2 * np.einsum("kjca,icbk->ijab", l2, u[o, v, v, o], optimize=True)

    rhs -= 2 * np.einsum("kjcb,icka->ijab", l2, u[o, v, o, v], optimize=True)

    rhs += 4 * np.einsum("ikac,cjkb->ijab", l2, u[v, o, o, v], optimize=True)

    rhs += 4 * np.einsum("kjcb,icak->ijab", l2, u[o, v, v, o], optimize=True)

    rhs += np.einsum("ib,ck,kjac->ijab", l1, t1, u[o, o, v, v], optimize=True)

    rhs += np.einsum("ic,ck,kjba->ijab", l1, t1, u[o, o, v, v], optimize=True)

    rhs += np.einsum("ja,ck,ikcb->ijab", l1, t1, u[o, o, v, v], optimize=True)

    rhs += np.einsum("jc,ck,ikba->ijab", l1, t1, u[o, o, v, v], optimize=True)

    rhs += np.einsum("ka,ck,ijbc->ijab", l1, t1, u[o, o, v, v], optimize=True)

    rhs += np.einsum("kb,ck,ijca->ijab", l1, t1, u[o, o, v, v], optimize=True)

    rhs -= 2 * np.einsum("ic,ck,kjab->ijab", f[o, v], t1, l2, optimize=True)

    rhs -= 2 * np.einsum("jc,ck,ikab->ijab", f[o, v], t1, l2, optimize=True)

    rhs -= 2 * np.einsum("ka,ck,ijcb->ijab", f[o, v], t1, l2, optimize=True)

    rhs -= 2 * np.einsum("kb,ck,ijac->ijab", f[o, v], t1, l2, optimize=True)

    rhs -= 2 * np.einsum(
        "ia,ck,kjbc->ijab", l1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ib,ck,kjca->ijab", l1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ic,ck,kjab->ijab", l1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ja,ck,ikbc->ijab", l1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "jb,ck,ikca->ijab", l1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "jc,ck,ikab->ijab", l1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ka,ck,ijcb->ijab", l1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "kb,ck,ijac->ijab", l1, t1, u[o, o, v, v], optimize=True
    )

    rhs += 4 * np.einsum(
        "ia,ck,kjcb->ijab", l1, t1, u[o, o, v, v], optimize=True
    )

    rhs += 4 * np.einsum(
        "jb,ck,ikac->ijab", l1, t1, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum("ijcd,cdab->ijab", l2, u[v, v, v, v], optimize=True)

    rhs += np.einsum("ka,ijbk->ijab", l1, u[o, o, v, o])

    rhs += np.einsum("kb,ijka->ijab", l1, u[o, o, o, v])

    rhs -= 2 * np.einsum("ik,kjab->ijab", f[o, o], l2)

    rhs -= 2 * np.einsum("jk,ikab->ijab", f[o, o], l2)

    rhs -= 2 * np.einsum("ka,ijkb->ijab", l1, u[o, o, o, v])

    rhs -= 2 * np.einsum("kb,ijak->ijab", l1, u[o, o, v, o])

    rhs += 2 * np.einsum("kmab,ijkm->ijab", l2, u[o, o, o, o])

    return rhs
