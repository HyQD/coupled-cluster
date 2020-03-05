import numpy as np


def l2_rhs(f, u, l, t):

    nocc = t.shape[2]
    nvirt = t.shape[0]
    o, v = slice(0, nocc), slice(nocc, nvirt + nocc)

    rhs = np.zeros((nocc, nocc, nvirt, nvirt))

    rhs += 2 * np.einsum("ca,ijcb->ijab", f[v, v], l)

    rhs += 2 * np.einsum("cb,ijac->ijab", f[v, v], l)

    rhs -= 2 * np.einsum("ijba->ijab", u[o, o, v, v])

    rhs += 4 * np.einsum("ijab->ijab", u[o, o, v, v])

    rhs -= 4 * np.einsum(
        "ijac,cdkm,kmbd->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ijcb,cdkm,kmad->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ikab,cdmk,mjcd->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ikac,cdmk,mjdb->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ikac,cdkm,mjbd->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ikbc,cdkm,mjda->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "ikcd,cdmk,mjab->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kmac,cdmk,ijdb->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kmbc,cdmk,ijad->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kjab,cdkm,imcd->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kjca,cdkm,imbd->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kjcb,cdmk,imad->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kjcb,cdkm,imda->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 4 * np.einsum(
        "kjcd,cdkm,imab->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ijac,cdmk,kmbd->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ijcb,cdmk,kmad->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ijcd,cdkm,kmab->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ikab,cdkm,mjcd->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ikac,cdmk,mjbd->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ikbc,cdmk,mjda->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ikbc,cdkm,mjad->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ikcb,cdmk,mjad->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "ikcd,cdmk,mjba->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kmab,cdkm,ijcd->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kmac,cdmk,ijbd->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kmbc,cdmk,ijda->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kjab,cdmk,imcd->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kjac,cdmk,imdb->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kjca,cdmk,imbd->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kjca,cdkm,imdb->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kjcb,cdmk,imda->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "kjcd,cdkm,imba->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 8 * np.einsum(
        "ikac,cdkm,mjdb->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs += 8 * np.einsum(
        "kjcb,cdkm,imad->ijab", l, t, u[o, o, v, v], optimize=True
    )

    rhs -= 2 * np.einsum("ikac,cjbk->ijab", l, u[v, o, v, o])

    rhs -= 2 * np.einsum("ikbc,cjka->ijab", l, u[v, o, o, v])

    rhs -= 2 * np.einsum("ikcb,cjak->ijab", l, u[v, o, v, o])

    rhs -= 2 * np.einsum("kjac,ickb->ijab", l, u[o, v, o, v])

    rhs -= 2 * np.einsum("kjca,icbk->ijab", l, u[o, v, v, o])

    rhs -= 2 * np.einsum("kjcb,icka->ijab", l, u[o, v, o, v])

    rhs += 4 * np.einsum("ikac,cjkb->ijab", l, u[v, o, o, v])

    rhs += 4 * np.einsum("kjcb,icak->ijab", l, u[o, v, v, o])

    rhs += 2 * np.einsum("ijcd,cdab->ijab", l, u[v, v, v, v])

    rhs -= 2 * np.einsum("ik,kjab->ijab", f[o, o], l)

    rhs -= 2 * np.einsum("jk,ikab->ijab", f[o, o], l)

    rhs += 2 * np.einsum("kmab,ijkm->ijab", l, u[o, o, o, o])

    return rhs
