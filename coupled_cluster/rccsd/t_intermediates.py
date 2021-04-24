__authors__ = "Ashutosh Kumar"
__credits__ = [
    "T. D. Crawford",
    "Daniel G. A. Smith",
    "Lori A. Burns",
    "Ashutosh Kumar",
]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-17"

import numpy as np

# Bulid Eqn 9:
def build_tilde_tau(t1, t2, o, v):
    ttau = t2.copy()
    tmp = 0.5 * np.einsum("ai,bj->abij", t1, t1)
    ttau += tmp
    return ttau


# Build Eqn 10:
def build_tau(t1, t2, o, v):
    ttau = t2.copy()
    tmp = np.einsum("ai,bj->abij", t1, t1)
    ttau += tmp
    return ttau


# Build Eqn 3:
def build_Fae(f, u, t1, t2, o, v):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Fae = np.zeros((nvirt, nvirt), dtype=t1.dtype)

    Fae += f[v, v]
    Fae -= 0.5 * np.einsum("me,am->ae", f[o, v], t1)
    Fae += 2 * np.einsum("fm,mafe->ae", t1, u[o, v, v, v])
    Fae -= np.einsum("fm,maef->ae", t1, u[o, v, v, v])
    Fae -= 2 * np.einsum(
        "afmn,mnef->ae", build_tilde_tau(t1, t2, o, v), u[o, o, v, v]
    )
    Fae += np.einsum(
        "afmn,mnfe->ae", build_tilde_tau(t1, t2, o, v), u[o, o, v, v]
    )
    return Fae


# Build Eqn 4:
def build_Fmi(f, u, t1, t2, o, v):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Fmi = np.zeros((nocc, nocc), dtype=t1.dtype)

    Fmi += f[o, o]
    Fmi += 0.5 * np.einsum("ei,me->mi", t1, f[o, v])
    Fmi += 2 * np.einsum("en,mnie->mi", t1, u[o, o, o, v])
    Fmi -= np.einsum("en,mnei->mi", t1, u[o, o, v, o])
    Fmi += 2 * np.einsum(
        "efin,mnef->mi", build_tilde_tau(t1, t2, o, v), u[o, o, v, v]
    )
    Fmi -= np.einsum(
        "efin,mnfe->mi", build_tilde_tau(t1, t2, o, v), u[o, o, v, v]
    )
    return Fmi


# Build Eqn 5:
def build_Fme(f, u, t1, o, v):
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Fme = np.zeros((nocc, nvirt), dtype=t1.dtype)
    Fme += f[o, v]
    Fme += 2 * np.einsum("fn,mnef->me", t1, u[o, o, v, v])
    Fme -= np.einsum("fn,mnfe->me", t1, u[o, o, v, v])
    return Fme


# Build Eqn 6:
def build_Wmnij(u, t1, t2, o, v):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Wmnij = np.zeros((nocc, nocc, nocc, nocc), dtype=t1.dtype)

    Wmnij += u[o, o, o, o]
    Wmnij += np.einsum("ej,mnie->mnij", t1, u[o, o, o, v])
    Wmnij += np.einsum("ei,mnej->mnij", t1, u[o, o, v, o])
    # prefactor of 1 instead of 0.5 below to fold the last term of
    # 0.5 * tau_ijef Wabef in Wmnij contraction: 0.5 * tau_mnab Wmnij_mnij
    Wmnij += np.einsum(
        "efij,mnef->mnij", build_tau(t1, t2, o, v), u[o, o, v, v]
    )
    return Wmnij


# Build Eqn 8:
def build_Wmbej(u, t1, t2, o, v):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Wmbej = np.zeros((nocc, nvirt, nvirt, nocc), dtype=t1.dtype)

    Wmbej += u[o, v, v, o]
    Wmbej += np.einsum("fj,mbef->mbej", t1, u[o, v, v, v])
    Wmbej -= np.einsum("bn,mnej->mbej", t1, u[o, o, v, o])
    tmp = 0.5 * t2
    tmp += np.einsum("fj,bn->fbjn", t1, t1)
    Wmbej -= np.einsum("fbjn,mnef->mbej", tmp, u[o, o, v, v])
    Wmbej += np.einsum("fbnj,mnef->mbej", t2, u[o, o, v, v])
    Wmbej -= 0.5 * np.einsum("fbnj,mnfe->mbej", t2, u[o, o, v, v])
    return Wmbej


# This intermediate appaears in the spin factorization of Wmbej terms.
def build_Wmbje(u, t1, t2, o, v):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Wmbje = np.zeros((nocc, nvirt, nocc, nvirt), dtype=t1.dtype)

    Wmbje += -1.0 * u[o, v, o, v]
    Wmbje -= np.einsum("fj,mbfe->mbje", t1, u[o, v, v, v])
    Wmbje += np.einsum("bn,mnje->mbje", t1, u[o, o, o, v])
    tmp = 0.5 * t2
    tmp += np.einsum("fj,bn->fbjn", t1, t1)
    Wmbje += np.einsum("fbjn,mnfe->mbje", tmp, u[o, o, v, v])
    return Wmbje


# This intermediate is required to build second term of 0.5 * tau_ijef * Wabef,
# as explicit construction of Wabef is avoided here.
def build_Zmbij(u, t1, t2, o, v):
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Zmbij = np.zeros((nocc, nvirt, nocc, nocc), dtype=t1.dtype)

    Zmbij += np.einsum(
        "mbef,efij->mbij", u[o, v, v, v], build_tau(t1, t2, o, v)
    )
    return Zmbij
