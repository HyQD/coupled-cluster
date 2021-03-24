import numpy as np
from coupled_cluster.rccsd.utils import ndot


def build_Doo(t1, t2, l1, l2):
    Doo = ndot("cj,ic->ij", t1, l1, prefactor=-1)
    Doo -= ndot("cdkj,kicd->ij", t2, l2)
    return Doo


def build_Dvv(t1, t2, l1, l2):
    Dvv = ndot("ak,kb->ab", t1, l1)
    Dvv += ndot("cakl,klcb->ab", t2, l2)
    return Dvv


def build_Dov(t1, t2, l1, l2):
    Dov = 2 * t1
    Dov += ndot("acik,kc->ai", t2, l1, prefactor=2)
    Dov -= ndot("caik,kc->ai", t2, l1)
    tmp = ndot("ak,kc->ca", t1, l1)
    Dov -= ndot("ci,ca->ai", t1, tmp)
    Dov -= np.einsum("al,cdki,klcd->ai", t1, t2, l2, optimize=True)
    Dov -= np.einsum("di,cakl,klcd->ai", t1, t2, l2, optimize=True)
    return Dov


def compute_one_body_density_matrix(t1, t2, l1, l2, o, v, np, out=None):
    nmo = t1.shape[1] + t1.shape[0]
    nocc = t1.shape[1]

    Doo = build_Doo(t1, t2, l1, l2)
    Dvv = build_Dvv(t1, t2, l1, l2)
    Dov = build_Dov(t1, t2, l1, l2)

    D = np.zeros((nmo, nmo), dtype=t1.dtype)
    D[:nocc, :nocc] += 2 * np.eye(nocc)  # HF MO density
    D[:nocc, :nocc] += Doo
    D[:nocc, nocc:] += l1
    D[nocc:, nocc:] += Dvv
    D[nocc:, :nocc] += Dov

    return D
