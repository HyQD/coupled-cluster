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


def build_Loovv(u, o, v):
    tmp = u[o, o, v, v].copy()
    Loovv = 2.0 * tmp - tmp.swapaxes(2, 3)
    return Loovv


def build_Looov(u, o, v):
    tmp = u[o, o, o, v].copy()
    Looov = 2.0 * tmp - tmp.swapaxes(0, 1)
    return Looov


def build_Lvovv(u, o, v):
    tmp = u[v, o, v, v].copy()
    Lvovv = 2.0 * tmp - tmp.swapaxes(2, 3)
    return Lvovv


def build_tau(t1, t2, o, v):
    ttau = t2.copy()
    tmp = np.einsum("ai,bj->abij", t1, t1)
    ttau += tmp
    return ttau


# F and W are the one and two body intermediates which appear in the CCSD
# T1 and T2 equations. Please refer to helper_ccenergy file for more details.


def build_Hov(f, Loovv, t1, o, v):
    """<m|Hbar|e> = F_me = f_me + t_nf <mn||ef>"""

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hov = np.zeros((nocc, nvirt), dtype=t1.dtype)

    Hov += f[o, v]
    Hov += np.einsum("fn,mnef->me", t1, Loovv)
    return Hov


def build_Hoo(f, Looov, Loovv, t1, t2, o, v):
    """
    <m|Hbar|i> = F_mi + 0.5 * t_ie F_me = f_mi + t_ie f_me
                 + t_ne <mn||ie> + tau_inef <mn||ef>
    """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hoo = np.zeros((nocc, nocc), dtype=t1.dtype)

    Hoo += f[o, o]
    Hoo += np.einsum("ei,me->mi", t1, f[o, v])
    Hoo += np.einsum("en,mnie->mi", t1, Looov)
    Hoo += np.einsum("efin,mnef->mi", build_tau(t1, t2, o, v), Loovv)
    return Hoo


def build_Hvv(f, Lvovv, Loovv, t1, t2, o, v):
    """
    <a|Hbar|e> = F_ae - 0.5 * t_ma F_me = f_ae - t_ma f_me
                 + t_mf <am||ef> - tau_mnfa <mn||fe>
    """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hvv = np.zeros((nvirt, nvirt), dtype=t1.dtype)

    Hvv += f[v, v]
    Hvv -= np.einsum("am,me->ae", t1, f[o, v])
    Hvv += np.einsum("fm,amef->ae", t1, Lvovv)
    Hvv -= np.einsum("famn,mnfe->ae", build_tau(t1, t2, o, v), Loovv)
    return Hvv


def build_Hoooo(u, t1, t2, o, v):
    """
    <mn|Hbar|ij> = W_mnij + 0.25 * tau_ijef <mn||ef> = <mn||ij>
                   + P(ij) t_je <mn||ie> + 0.5 * tau_ijef <mn||ef>
    """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hoooo = np.zeros((nocc, nocc, nocc, nocc), dtype=t1.dtype)

    Hoooo += u[o, o, o, o]
    Hoooo += np.einsum("ej,mnie->mnij", t1, u[o, o, o, v])
    Hoooo += np.einsum("ei,mnej->mnij", t1, u[o, o, v, o])
    Hoooo += np.einsum(
        "efij,mnef->mnij", build_tau(t1, t2, o, v), u[o, o, v, v]
    )
    return Hoooo


def build_Hvvvv(u, t1, t2, o, v):
    """
    <ab|Hbar|ef> = W_abef + 0.25 * tau_mnab <mn||ef> = <ab||ef>
                   - P(ab) t_mb <am||ef> + 0.5 * tau_mnab <mn||ef>
    """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hvvvv = np.zeros((nvirt, nvirt, nvirt, nvirt), dtype=t1.dtype)

    Hvvvv += u[v, v, v, v]
    Hvvvv -= np.einsum("bm,amef->abef", t1, u[v, o, v, v])
    Hvvvv -= np.einsum("am,bmfe->abef", t1, u[v, o, v, v])
    Hvvvv += np.einsum(
        "abmn,mnef->abef", build_tau(t1, t2, o, v), u[o, o, v, v]
    )
    return Hvvvv


def build_Hvovv(u, t1, o, v):
    """<am|Hbar|ef> = <am||ef> - t_na <nm||ef>"""
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hvovv = np.zeros((nvirt, nocc, nvirt, nvirt), dtype=t1.dtype)

    Hvovv += u[v, o, v, v]
    Hvovv -= np.einsum("an,nmef->amef", t1, u[o, o, v, v])
    return Hvovv


def build_Hooov(u, t1, o, v):
    """<mn|Hbar|ie> = <mn||ie> + t_if <mn||fe>"""
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hooov = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)

    Hooov += u[o, o, o, v]
    Hooov += np.einsum("fi,mnfe->mnie", t1, u[o, o, v, v])
    return Hooov


def build_Hovvo(u, Loovv, t1, t2, o, v):
    """
    <mb|Hbar|ej> = W_mbej - 0.5 * t_jnfb <mn||ef> = <mb||ej> + t_jf <mb||ef>
                   - t_nb <mn||ej> - (t_jnfb + t_jf t_nb) <nm||fe>
    """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hovvo = np.zeros((nocc, nvirt, nvirt, nocc), dtype=t1.dtype)

    Hovvo += u[o, v, v, o]
    Hovvo += np.einsum("fj,mbef->mbej", t1, u[o, v, v, v])
    Hovvo -= np.einsum("bn,mnej->mbej", t1, u[o, o, v, o])
    Hovvo -= np.einsum(
        "fbjn,nmfe->mbej", build_tau(t1, t2, o, v), u[o, o, v, v]
    )
    Hovvo += np.einsum("bfjn,nmfe->mbej", t2, Loovv)
    return Hovvo


def build_Hovov(u, t1, t2, o, v):
    """
    <mb|Hbar|je> = - <mb|Hbar|ej> = <mb||je> + t_jf <bm||ef> - t_nb <mn||je>
                   - (t_jnfb + t_jf t_nb) <nm||ef>
    """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hovov = np.zeros((nocc, nvirt, nocc, nvirt), dtype=t1.dtype)

    Hovov += u[o, v, o, v]
    Hovov += np.einsum("fj,bmef->mbje", t1, u[v, o, v, v])
    Hovov -= np.einsum("bn,mnje->mbje", t1, u[o, o, o, v])
    Hovov -= np.einsum(
        "fbjn,nmef->mbje", build_tau(t1, t2, o, v), u[o, o, v, v]
    )
    return Hovov


def build_Hvvvo(f, u, Loovv, Lvovv, t1, t2, o, v):
    """
    <ab|Hbar|ei> = <ab||ei> - F_me t_miab + t_if Wabef + 0.5 * tau_mnab <mn||ei>
                   - P(ab) t_miaf <mb||ef> - P(ab) t_ma {<mb||ei> - t_nibf <mn||ef>}
    """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hvvvo = np.zeros((nvirt, nvirt, nvirt, nocc), dtype=t1.dtype)

    # <ab||ei>

    Hvvvo += u[v, v, v, o]

    # - Fme t_miab

    Hvvvo -= np.einsum("me,abmi->abei", f[o, v], t2)
    tmp = np.einsum("mnfe,fm->ne", Loovv, t1)
    Hvvvo -= np.einsum("abni,ne->abei", t2, tmp)

    # t_if Wabef

    Hvvvo += np.einsum("fi,abef->abei", t1, u[v, v, v, v])
    tmp = np.einsum("fi,am->imfa", t1, t1)
    Hvvvo -= np.einsum("imfa,mbef->abei", tmp, u[o, v, v, v])
    Hvvvo -= np.einsum("imfb,amef->abei", tmp, u[v, o, v, v])
    tmp = np.einsum("mnef,fi->mnei", u[o, o, v, v], t1)
    Hvvvo += np.einsum("abmn,mnei->abei", t2, tmp)
    tmp = np.einsum("fi,am->imfa", t1, t1)
    tmp1 = np.einsum("mnef,bn->mbef", u[o, o, v, v], t1)
    Hvvvo += np.einsum("imfa,mbef->abei", tmp, tmp1)

    # 0.5 * tau_mnab <mn||ei>

    Hvvvo += np.einsum(
        "abmn,mnei->abei", build_tau(t1, t2, o, v), u[o, o, v, o]
    )

    # - P(ab) t_miaf <mb||ef>

    Hvvvo -= np.einsum("faim,mbef->abei", t2, u[o, v, v, v])
    Hvvvo -= np.einsum("fbim,amef->abei", t2, u[v, o, v, v])
    Hvvvo += np.einsum("fbmi,amef->abei", t2, Lvovv)

    # - P(ab) t_ma <mb||ei>
    Hvvvo -= np.einsum("bm,amei->abei", t1, u[v, o, v, o])
    Hvvvo -= np.einsum("am,bmie->abei", t1, u[v, o, o, v])

    # P(ab) t_ma * t_nibf <mn||ef>

    tmp = np.einsum("mnef,am->anef", u[o, o, v, v], t1)
    Hvvvo += np.einsum("fbin,anef->abei", t2, tmp)
    tmp = np.einsum("mnef,am->nafe", Loovv, t1)
    Hvvvo -= np.einsum("fbni,nafe->abei", t2, tmp)
    tmp = np.einsum("nmef,bm->nefb", u[o, o, v, v], t1)
    Hvvvo += np.einsum("afni,nefb->abei", t2, tmp)
    return Hvvvo


def build_Hovoo(f, u, Loovv, Looov, t1, t2, o, v):
    """
    <mb|Hbar|ij> = <mb||ij> - Fme t_ijbe - t_nb Wmnij + 0.5 * tau_ijef <mb||ef>
                   + P(ij) t_jnbe <mn||ie> + P(ij) t_ie {<mb||ej> - t_njbf <mn||ef>}
    """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hovoo = np.zeros((nocc, nvirt, nocc, nocc), dtype=t1.dtype)

    # <mb||ij>

    Hovoo += u[o, v, o, o]

    # - Fme t_ijbe

    Hovoo += np.einsum("me,ebij->mbij", f[o, v], t2)
    tmp = np.einsum("mnef,fn->me", Loovv, t1)
    Hovoo += np.einsum("me,ebij->mbij", tmp, t2)

    # - t_nb Wmnij

    Hovoo -= np.einsum("bn,mnij->mbij", t1, u[o, o, o, o])
    tmp = np.einsum("ei,bn->ineb", t1, t1)
    Hovoo -= np.einsum("ineb,mnej->mbij", tmp, u[o, o, v, o])
    Hovoo -= np.einsum("jneb,mnie->mbij", tmp, u[o, o, o, v])
    tmp = np.einsum("bn,mnef->mefb", t1, u[o, o, v, v])
    Hovoo -= np.einsum("efij,mefb->mbij", t2, tmp)
    tmp = np.einsum("ei,fj->ijef", t1, t1)
    tmp1 = np.einsum("bn,mnef->mbef", t1, u[o, o, v, v])
    Hovoo -= np.einsum("mbef,ijef->mbij", tmp1, tmp)

    # 0.5 * tau_ijef <mb||ef>

    Hovoo += np.einsum(
        "efij,mbef->mbij", build_tau(t1, t2, o, v), u[o, v, v, v]
    )

    # P(ij) t_jnbe <mn||ie>

    Hovoo -= np.einsum("ebin,mnej->mbij", t2, u[o, o, v, o])
    Hovoo -= np.einsum("ebjn,mnie->mbij", t2, u[o, o, o, v])
    Hovoo += np.einsum("bejn,mnie->mbij", t2, Looov)

    # P(ij) t_ie <mb||ej>

    Hovoo += np.einsum("ej,mbie->mbij", t1, u[o, v, o, v])
    Hovoo += np.einsum("ei,mbej->mbij", t1, u[o, v, v, o])

    # - P(ij) t_ie * t_njbf <mn||ef>

    tmp = np.einsum("ei,mnef->mnif", t1, u[o, o, v, v])
    Hovoo -= np.einsum("fbjn,mnif->mbij", t2, tmp)
    tmp = np.einsum("mnef,fbnj->mejb", Loovv, t2)
    Hovoo += np.einsum("mejb,ei->mbij", tmp, t1)
    tmp = np.einsum("ej,mnfe->mnfj", t1, u[o, o, v, v])
    Hovoo -= np.einsum("fbin,mnfj->mbij", t2, tmp)
    return Hovoo
