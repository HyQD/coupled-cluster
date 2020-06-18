from coupled_cluster.rccsd.utils import ndot
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
    """ <m|Hbar|e> = F_me = f_me + t_nf <mn||ef> """

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hov = np.zeros((nocc, nvirt), dtype=t1.dtype)

    Hov += f[o, v]
    Hov += ndot("fn,mnef->me", t1, Loovv)
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
    Hoo += ndot("ei,me->mi", t1, f[o, v])
    Hoo += ndot("en,mnie->mi", t1, Looov)
    Hoo += ndot("efin,mnef->mi", build_tau(t1, t2, o, v), Loovv)
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
    Hvv -= ndot("am,me->ae", t1, f[o, v])
    Hvv += ndot("fm,amef->ae", t1, Lvovv)
    Hvv -= ndot("famn,mnfe->ae", build_tau(t1, t2, o, v), Loovv)
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
    Hoooo += ndot("ej,mnie->mnij", t1, u[o, o, o, v])
    Hoooo += ndot("ei,mnej->mnij", t1, u[o, o, v, o])
    Hoooo += ndot("efij,mnef->mnij", build_tau(t1, t2, o, v), u[o, o, v, v])
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
    Hvvvv -= ndot("bm,amef->abef", t1, u[v, o, v, v])
    Hvvvv -= ndot("am,bmfe->abef", t1, u[v, o, v, v])
    Hvvvv += ndot("abmn,mnef->abef", build_tau(t1, t2, o, v), u[o, o, v, v])
    return Hvvvv


def build_Hvovv(u, t1, o, v):
    """ <am|Hbar|ef> = <am||ef> - t_na <nm||ef> """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hvovv = np.zeros((nvirt, nocc, nvirt, nvirt), dtype=t1.dtype)

    Hvovv += u[v, o, v, v]
    Hvovv -= ndot("an,nmef->amef", t1, u[o, o, v, v])
    return Hvovv


def build_Hooov(u, t1, o, v):
    """ <mn|Hbar|ie> = <mn||ie> + t_if <mn||fe> """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Hooov = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)

    Hooov += u[o, o, o, v]
    Hooov += ndot("fi,mnfe->mnie", t1, u[o, o, v, v])
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
    Hovvo += ndot("fj,mbef->mbej", t1, u[o, v, v, v])
    Hovvo -= ndot("bn,mnej->mbej", t1, u[o, o, v, o])
    Hovvo -= ndot("fbjn,nmfe->mbej", build_tau(t1, t2, o, v), u[o, o, v, v])
    Hovvo += ndot("bfjn,nmfe->mbej", t2, Loovv)
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
    Hovov += ndot("fj,bmef->mbje", t1, u[v, o, v, v])
    Hovov -= ndot("bn,mnje->mbje", t1, u[o, o, o, v])
    Hovov -= ndot("fbjn,nmef->mbje", build_tau(t1, t2, o, v), u[o, o, v, v])
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

    Hvvvo -= ndot("me,abmi->abei", f[o, v], t2)
    tmp = ndot("mnfe,fm->ne", Loovv, t1)
    Hvvvo -= ndot("abni,ne->abei", t2, tmp)

    # t_if Wabef

    Hvvvo += ndot("fi,abef->abei", t1, u[v, v, v, v])
    tmp = ndot("fi,am->imfa", t1, t1)
    Hvvvo -= ndot("imfa,mbef->abei", tmp, u[o, v, v, v])
    Hvvvo -= ndot("imfb,amef->abei", tmp, u[v, o, v, v])
    tmp = ndot("mnef,fi->mnei", u[o, o, v, v], t1)
    Hvvvo += ndot("abmn,mnei->abei", t2, tmp)
    tmp = ndot("fi,am->imfa", t1, t1)
    tmp1 = ndot("mnef,bn->mbef", u[o, o, v, v], t1)
    Hvvvo += ndot("imfa,mbef->abei", tmp, tmp1)

    # 0.5 * tau_mnab <mn||ei>

    Hvvvo += ndot("abmn,mnei->abei", build_tau(t1, t2, o, v), u[o, o, v, o])

    # - P(ab) t_miaf <mb||ef>

    Hvvvo -= ndot("faim,mbef->abei", t2, u[o, v, v, v])
    Hvvvo -= ndot("fbim,amef->abei", t2, u[v, o, v, v])
    Hvvvo += ndot("fbmi,amef->abei", t2, Lvovv)

    # - P(ab) t_ma <mb||ei>
    Hvvvo -= ndot("bm,amei->abei", t1, u[v, o, v, o])
    Hvvvo -= ndot("am,bmie->abei", t1, u[v, o, o, v])

    # P(ab) t_ma * t_nibf <mn||ef>

    tmp = ndot("mnef,am->anef", u[o, o, v, v], t1)
    Hvvvo += ndot("fbin,anef->abei", t2, tmp)
    tmp = ndot("mnef,am->nafe", Loovv, t1)
    Hvvvo -= ndot("fbni,nafe->abei", t2, tmp)
    tmp = ndot("nmef,bm->nefb", u[o, o, v, v], t1)
    Hvvvo += ndot("afni,nefb->abei", t2, tmp)
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

    Hovoo += ndot("me,ebij->mbij", f[o, v], t2)
    tmp = ndot("mnef,fn->me", Loovv, t1)
    Hovoo += ndot("me,ebij->mbij", tmp, t2)

    # - t_nb Wmnij

    Hovoo -= ndot("bn,mnij->mbij", t1, u[o, o, o, o])
    tmp = ndot("ei,bn->ineb", t1, t1)
    Hovoo -= ndot("ineb,mnej->mbij", tmp, u[o, o, v, o])
    Hovoo -= ndot("jneb,mnie->mbij", tmp, u[o, o, o, v])
    tmp = ndot("bn,mnef->mefb", t1, u[o, o, v, v])
    Hovoo -= ndot("efij,mefb->mbij", t2, tmp)
    tmp = ndot("ei,fj->ijef", t1, t1)
    tmp1 = ndot("bn,mnef->mbef", t1, u[o, o, v, v])
    Hovoo -= ndot("mbef,ijef->mbij", tmp1, tmp)

    # 0.5 * tau_ijef <mb||ef>

    Hovoo += ndot("efij,mbef->mbij", build_tau(t1, t2, o, v), u[o, v, v, v])

    # P(ij) t_jnbe <mn||ie>

    Hovoo -= ndot("ebin,mnej->mbij", t2, u[o, o, v, o])
    Hovoo -= ndot("ebjn,mnie->mbij", t2, u[o, o, o, v])
    Hovoo += ndot("bejn,mnie->mbij", t2, Looov)

    # P(ij) t_ie <mb||ej>

    Hovoo += ndot("ej,mbie->mbij", t1, u[o, v, o, v])
    Hovoo += ndot("ei,mbej->mbij", t1, u[o, v, v, o])

    # - P(ij) t_ie * t_njbf <mn||ef>

    tmp = ndot("ei,mnef->mnif", t1, u[o, o, v, v])
    Hovoo -= ndot("fbjn,mnif->mbij", t2, tmp)
    tmp = ndot("mnef,fbnj->mejb", Loovv, t2)
    Hovoo += ndot("mejb,ei->mbij", tmp, t1)
    tmp = ndot("ej,mnfe->mnfj", t1, u[o, o, v, v])
    Hovoo -= ndot("fbin,mnfj->mbij", t2, tmp)
    return Hovoo
