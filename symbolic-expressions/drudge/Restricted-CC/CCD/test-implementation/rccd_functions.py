import numpy as np


def corr_energy(t2, u):
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    o, v = slice(0, nocc), slice(nocc, nvirt + nocc)
    e_corr = 2 * np.einsum("abij,ijab->", t2, u[o, o, v, v])
    e_corr -= np.einsum("abij,ijba->", t2, u[o, o, v, v])
    return e_corr


def t2_rhs(f, u, t):

    nocc = t.shape[2]
    nvirt = t.shape[0]
    o, v = slice(0, nocc), slice(nocc, nvirt + nocc)

    rhs = np.zeros((nvirt, nvirt, nocc, nocc))

    rhs += np.einsum("abij->abij", u[v, v, o, o])

    rhs += np.einsum("ac,bcji->abij", f[v, v], t)

    rhs += np.einsum("bc,acij->abij", f[v, v], t)

    rhs -= np.einsum("acik,bkcj->abij", t, u[v, o, v, o])

    rhs -= np.einsum("acki,bkjc->abij", t, u[v, o, o, v])

    rhs -= np.einsum("ackj,bkci->abij", t, u[v, o, v, o])

    rhs -= np.einsum("bcjk,akci->abij", t, u[v, o, v, o])

    rhs -= np.einsum("bcki,akcj->abij", t, u[v, o, v, o])

    rhs -= np.einsum("bckj,akic->abij", t, u[v, o, o, v])

    rhs += 2 * np.einsum("acik,bkjc->abij", t, u[v, o, o, v])

    rhs += 2 * np.einsum("bcjk,akic->abij", t, u[v, o, o, v])

    rhs += np.einsum("abkl,klij->abij", t, u[o, o, o, o])

    rhs += np.einsum("cdij,abcd->abij", t, u[v, v, v, v])

    rhs += np.einsum("abik,cdlj,klcd->abij", t, t, u[o, o, v, v], optimize=True)

    rhs += np.einsum("abkj,cdil,kldc->abij", t, t, u[o, o, v, v], optimize=True)

    rhs += np.einsum("abkl,cdij,klcd->abij", t, t, u[o, o, v, v], optimize=True)

    rhs += np.einsum("acij,bdkl,kldc->abij", t, t, u[o, o, v, v], optimize=True)

    rhs += np.einsum("acik,bdlj,kldc->abij", t, t, u[o, o, v, v], optimize=True)

    rhs += np.einsum("acki,bdjl,kldc->abij", t, t, u[o, o, v, v], optimize=True)

    rhs += np.einsum("acki,bdlj,klcd->abij", t, t, u[o, o, v, v], optimize=True)

    rhs += np.einsum("ackj,bdli,kldc->abij", t, t, u[o, o, v, v], optimize=True)

    rhs += np.einsum("ackl,bdji,klcd->abij", t, t, u[o, o, v, v], optimize=True)

    rhs -= 2 * np.einsum(
        "abik,cdlj,kldc->abij", t, t, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "abkj,cdil,klcd->abij", t, t, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "acij,bdkl,klcd->abij", t, t, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "acik,bdjl,kldc->abij", t, t, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "acik,bdlj,klcd->abij", t, t, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "acki,bdjl,klcd->abij", t, t, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "ackl,bdji,kldc->abij", t, t, u[o, o, v, v], optimize=True
    )

    rhs += 4 * np.einsum(
        "acik,bdjl,klcd->abij", t, t, u[o, o, v, v], optimize=True
    )

    rhs -= np.einsum("ki,abkj->abij", f[o, o], t)

    rhs -= np.einsum("kj,abik->abij", f[o, o], t)

    return rhs
