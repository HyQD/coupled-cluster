import numpy as np


def corr_energy(f, u, t1, t2):
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    o, v = slice(0, nocc), slice(nocc, nvirt + nocc)

    e_corr = 2 * np.einsum("ia,ai->", f[o, v], t1)

    e_corr += 2 * np.einsum("abij,ijab->", t2, u[o, o, v, v])
    e_corr -= np.einsum("abij,ijba->", t2, u[o, o, v, v])

    e_corr += 2 * np.einsum(
        "ai,bj,ijab->", t1, t1, u[o, o, v, v], optimize=True
    )
    e_corr -= np.einsum("ai,bj,ijba->", t1, t1, u[o, o, v, v], optimize=True)

    return e_corr


def t1_rhs(f, u, t1, t2):
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    o, v = slice(0, nocc), slice(nocc, nvirt + nocc)

    rhs = np.zeros((nvirt, nocc))

    rhs += np.einsum("ab,bi->ai", f[v, v], t1)

    rhs += np.einsum("ai->ai", f[v, o])

    rhs += np.einsum("ak,bcij,jkbc->ai", t1, t2, u[o, o, v, v], optimize=True)

    rhs += np.einsum("ci,abjk,jkbc->ai", t1, t2, u[o, o, v, v], optimize=True)

    rhs += np.einsum("ck,abji,jkcb->ai", t1, t2, u[o, o, v, v], optimize=True)

    rhs -= 2 * np.einsum(
        "ak,bcij,jkcb->ai", t1, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ci,abjk,jkcb->ai", t1, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ck,abij,jkcb->ai", t1, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ck,abji,jkbc->ai", t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += 4 * np.einsum(
        "ck,abij,jkbc->ai", t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "aj,bi,ck,jkcb->ai", t1, t1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "aj,bi,ck,jkbc->ai", t1, t1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= np.einsum("jb,abji->ai", f[o, v], t2)

    rhs -= np.einsum("bj,ajbi->ai", t1, u[v, o, v, o])

    rhs += 2 * np.einsum("jb,abij->ai", f[o, v], t2)

    rhs += 2 * np.einsum("bj,ajib->ai", t1, u[v, o, o, v])

    rhs -= np.einsum("jb,aj,bi->ai", f[o, v], t1, t1, optimize=True)

    rhs -= np.einsum("bcij,ajcb->ai", t2, u[v, o, v, v])

    rhs += 2 * np.einsum("bcij,ajbc->ai", t2, u[v, o, v, v])

    rhs -= np.einsum("bi,cj,ajcb->ai", t1, t1, u[v, o, v, v], optimize=True)

    rhs += 2 * np.einsum("bi,cj,ajbc->ai", t1, t1, u[v, o, v, v], optimize=True)

    rhs -= np.einsum("ji,aj->ai", f[o, o], t1)

    rhs += np.einsum("abjk,kjib->ai", t2, u[o, o, o, v])

    rhs -= 2 * np.einsum("abjk,jkib->ai", t2, u[o, o, o, v])

    rhs += np.einsum("aj,bk,kjib->ai", t1, t1, u[o, o, o, v], optimize=True)

    rhs -= 2 * np.einsum("aj,bk,jkib->ai", t1, t1, u[o, o, o, v], optimize=True)

    return rhs


def t2_rhs(f, u, t1, t2):
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    o, v = slice(0, nocc), slice(nocc, nvirt + nocc)

    rhs = np.zeros((nvirt, nvirt, nocc, nocc))

    rhs += np.einsum("ac,bcji->abij", f[v, v], t2)

    rhs += np.einsum("bc,acij->abij", f[v, v], t2)

    rhs += np.einsum("ci,abcj->abij", t1, u[v, v, v, o])

    rhs += np.einsum("cj,abic->abij", t1, u[v, v, o, v])

    rhs += np.einsum("abij->abij", u[v, v, o, o])

    rhs += np.einsum("abkl,klij->abij", t2, u[o, o, o, o])

    rhs += np.einsum("ak,bl,klij->abij", t1, t1, u[o, o, o, o], optimize=True)

    rhs -= np.einsum("ak,cdij,bkdc->abij", t1, t2, u[v, o, v, v], optimize=True)

    rhs -= np.einsum("bk,cdij,akcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    rhs -= np.einsum("di,ackj,bkcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    rhs -= np.einsum("di,bcjk,akcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    rhs -= np.einsum("di,bckj,akdc->abij", t1, t2, u[v, o, v, v], optimize=True)

    rhs -= np.einsum("dj,acik,bkcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    rhs -= np.einsum("dj,acki,bkdc->abij", t1, t2, u[v, o, v, v], optimize=True)

    rhs -= np.einsum("dj,bcki,akcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    rhs -= np.einsum("dk,acij,bkdc->abij", t1, t2, u[v, o, v, v], optimize=True)

    rhs -= np.einsum("dk,bcji,akdc->abij", t1, t2, u[v, o, v, v], optimize=True)

    rhs += 2 * np.einsum(
        "di,bcjk,akdc->abij", t1, t2, u[v, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "dj,acik,bkdc->abij", t1, t2, u[v, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "dk,acij,bkcd->abij", t1, t2, u[v, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "dk,bcji,akcd->abij", t1, t2, u[v, o, v, v], optimize=True
    )

    rhs -= np.einsum(
        "ak,ci,dj,bkdc->abij", t1, t1, t1, u[v, o, v, v], optimize=True
    )

    rhs -= np.einsum(
        "bk,ci,dj,akcd->abij", t1, t1, t1, u[v, o, v, v], optimize=True
    )

    rhs -= np.einsum("acik,bkcj->abij", t2, u[v, o, v, o], optimize=True)

    rhs -= np.einsum("acki,bkjc->abij", t2, u[v, o, o, v], optimize=True)

    rhs -= np.einsum("ackj,bkci->abij", t2, u[v, o, v, o], optimize=True)

    rhs -= np.einsum("bcjk,akci->abij", t2, u[v, o, v, o], optimize=True)

    rhs -= np.einsum("bcki,akcj->abij", t2, u[v, o, v, o], optimize=True)

    rhs -= np.einsum("bckj,akic->abij", t2, u[v, o, o, v], optimize=True)

    rhs += 2 * np.einsum("acik,bkjc->abij", t2, u[v, o, o, v], optimize=True)

    rhs += 2 * np.einsum("bcjk,akic->abij", t2, u[v, o, o, v], optimize=True)

    rhs -= np.einsum("kc,ak,bcji->abij", f[o, v], t1, t2)

    rhs -= np.einsum("kc,bk,acij->abij", f[o, v], t1, t2)

    rhs -= np.einsum("kc,ci,abkj->abij", f[o, v], t1, t2)

    rhs -= np.einsum("kc,cj,abik->abij", f[o, v], t1, t2)

    rhs -= np.einsum("ak,ci,bkjc->abij", t1, t1, u[v, o, o, v], optimize=True)

    rhs -= np.einsum("ak,cj,bkci->abij", t1, t1, u[v, o, v, o], optimize=True)

    rhs -= np.einsum("bk,ci,akcj->abij", t1, t1, u[v, o, v, o], optimize=True)

    rhs -= np.einsum("bk,cj,akic->abij", t1, t1, u[v, o, o, v], optimize=True)

    rhs += np.einsum("cdij,abcd->abij", t2, u[v, v, v, v], optimize=True)

    rhs += np.einsum("ci,dj,abcd->abij", t1, t1, u[v, v, v, v], optimize=True)

    rhs += np.einsum("al,bcjk,klic->abij", t1, t2, u[o, o, o, v], optimize=True)

    rhs += np.einsum("al,bcki,lkcj->abij", t1, t2, u[o, o, v, o], optimize=True)

    rhs += np.einsum("al,bckj,lkic->abij", t1, t2, u[o, o, o, v], optimize=True)

    rhs += np.einsum("bl,acik,lkcj->abij", t1, t2, u[o, o, v, o], optimize=True)

    rhs += np.einsum("bl,acki,klcj->abij", t1, t2, u[o, o, v, o], optimize=True)

    rhs += np.einsum("bl,ackj,klic->abij", t1, t2, u[o, o, o, v], optimize=True)

    rhs += np.einsum("ci,abkl,klcj->abij", t1, t2, u[o, o, v, o], optimize=True)

    rhs += np.einsum("cj,abkl,klic->abij", t1, t2, u[o, o, o, v], optimize=True)

    rhs += np.einsum("cl,abik,klcj->abij", t1, t2, u[o, o, v, o], optimize=True)

    rhs += np.einsum("cl,abkj,lkic->abij", t1, t2, u[o, o, o, v], optimize=True)

    rhs -= 2 * np.einsum(
        "al,bcjk,lkic->abij", t1, t2, u[o, o, o, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "bl,acik,klcj->abij", t1, t2, u[o, o, v, o], optimize=True
    )

    rhs -= 2 * np.einsum(
        "cl,abik,lkcj->abij", t1, t2, u[o, o, v, o], optimize=True
    )

    rhs -= 2 * np.einsum(
        "cl,abkj,klic->abij", t1, t2, u[o, o, o, v], optimize=True
    )

    rhs += np.einsum(
        "ak,bl,ci,klcj->abij", t1, t1, t1, u[o, o, v, o], optimize=True
    )

    rhs += np.einsum(
        "ak,bl,cj,klic->abij", t1, t1, t1, u[o, o, o, v], optimize=True
    )

    rhs += np.einsum(
        "abik,cdlj,klcd->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "abkj,cdil,kldc->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "abkl,cdij,klcd->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "acij,bdkl,kldc->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "acik,bdlj,kldc->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "acki,bdjl,kldc->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "acki,bdlj,klcd->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "ackj,bdli,kldc->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "ackl,bdji,klcd->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "abik,cdlj,kldc->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "abkj,cdil,klcd->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "acij,bdkl,klcd->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "acik,bdjl,kldc->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "acik,bdlj,klcd->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "acki,bdjl,klcd->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ackl,bdji,kldc->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs += 4 * np.einsum(
        "acik,bdjl,klcd->abij", t2, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "ak,bl,cdij,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "ak,dl,bcji,kldc->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "al,di,bcjk,kldc->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "al,di,bckj,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "al,dj,bcki,kldc->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "bk,dl,acij,kldc->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "bl,di,ackj,kldc->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "bl,dj,acik,kldc->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "bl,dj,acki,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "ci,dj,abkl,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "ci,dl,abkj,kldc->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "cl,dj,abik,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ak,dl,bcji,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "al,di,bcjk,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "bk,dl,acij,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "bl,dj,acik,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ci,dl,abkj,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "cl,dj,abik,kldc->abij", t1, t1, t2, u[o, o, v, v], optimize=True
    )

    rhs += np.einsum(
        "ak,bl,ci,dj,klcd->abij", t1, t1, t1, t1, u[o, o, v, v], optimize=True
    )

    rhs -= np.einsum("ki,abkj->abij", f[o, o], t2)

    rhs -= np.einsum("kj,abik->abij", f[o, o], t2)

    rhs -= np.einsum("ak,bkji->abij", t1, u[v, o, o, o], optimize=True)

    rhs -= np.einsum("bk,akij->abij", t1, u[v, o, o, o], optimize=True)

    return rhs
