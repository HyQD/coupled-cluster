def compute_l_2_amplitudes(f, u, t2, l2, o, v, np, out=None):

    ################################################
    # These intermediates are common with those used in
    # compute_l1_amplitudes
    Loovv = build_Loovv(u, o, v)

    Hoo = build_Hoo(f, Loovv, t2, o, v)
    Hvv = build_Hvv(f, Loovv, t2, o, v)

    Hovvo = build_Hovvo(u, Loovv, t2, o, v)
    Hovov = build_Hovov(u, t2, o, v)
    ################################################
    Hoooo = build_Hoooo(u, t2, o, v)
    Hvvvv = build_Hvvvv(u, t2, o, v)

    # l2 equations
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    r_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    r_l2 += Loovv
    r_l2 += np.einsum("ijeb,ea->ijab", l2, Hvv)
    r_l2 -= np.einsum("im,mjab->ijab", Hoo, l2)

    r_l2 += 0.5 * np.einsum("ijmn,mnab->ijab", Hoooo, l2)

    r_l2 += 0.5 * np.einsum("ijef,efab->ijab", l2, Hvvvv)

    r_l2 += 2 * np.einsum("ieam,mjeb->ijab", Hovvo, l2)
    r_l2 -= np.einsum("iema,mjeb->ijab", Hovov, l2)
    r_l2 -= np.einsum("mibe,jema->ijab", l2, Hovov)
    r_l2 -= np.einsum("mieb,jeam->ijab", l2, Hovvo)

    r_l2 += np.einsum("ijeb,ae->ijab", Loovv, build_Gvv(t2, l2, np))
    r_l2 -= np.einsum("mi,mjab->ijab", build_Goo(t2, l2, np), Loovv)

    # Final r_l2_ijab = r_l2_ijab + r_l2_jiba
    r_l2 += r_l2.swapaxes(0, 1).swapaxes(2, 3)
    return r_l2


import numpy as np


def build_Loovv(u, o, v):
    tmp = u[o, o, v, v].copy()
    Loovv = 2.0 * tmp - tmp.swapaxes(2, 3)
    return Loovv


def build_Hoo(f, Loovv, t2, o, v):
    """
    <m|Hbar|i> = F_mi + 0.5 * t_ie F_me = f_mi + t_ie f_me
                 + t_ne <mn||ie> + tau_inef <mn||ef>
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hoo = np.zeros((nocc, nocc), dtype=t2.dtype)

    Hoo += f[o, o]
    Hoo += np.einsum("efin,mnef->mi", t2, Loovv)
    return Hoo


def build_Hvv(f, Loovv, t2, o, v):
    """
    <a|Hbar|e> = F_ae - 0.5 * t_ma F_me = f_ae - t_ma f_me
                 + t_mf <am||ef> - tau_mnfa <mn||fe>
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hvv = np.zeros((nvirt, nvirt), dtype=t2.dtype)

    Hvv += f[v, v]
    Hvv -= np.einsum("famn,mnfe->ae", t2, Loovv)
    return Hvv


def build_Hoooo(u, t2, o, v):
    """
    <mn|Hbar|ij> = W_mnij + 0.25 * tau_ijef <mn||ef> = <mn||ij>
                   + P(ij) t_je <mn||ie> + 0.5 * tau_ijef <mn||ef>
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hoooo = np.zeros((nocc, nocc, nocc, nocc), dtype=t2.dtype)

    Hoooo += u[o, o, o, o]
    Hoooo += np.einsum("efij,mnef->mnij", t2, u[o, o, v, v])
    return Hoooo


def build_Hvvvv(u, t2, o, v):
    """
    <ab|Hbar|ef> = W_abef + 0.25 * tau_mnab <mn||ef> = <ab||ef>
                   - P(ab) t_mb <am||ef> + 0.5 * tau_mnab <mn||ef>
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hvvvv = np.zeros((nvirt, nvirt, nvirt, nvirt), dtype=t2.dtype)

    Hvvvv += u[v, v, v, v]
    Hvvvv += np.einsum("abmn,mnef->abef", t2, u[o, o, v, v])
    return Hvvvv


def build_Hovvo(u, Loovv, t2, o, v):
    """
    <mb|Hbar|ej> = W_mbej - 0.5 * t_jnfb <mn||ef> = <mb||ej> + t_jf <mb||ef>
                   - t_nb <mn||ej> - (t_jnfb + t_jf t_nb) <nm||fe>
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hovvo = np.zeros((nocc, nvirt, nvirt, nocc), dtype=t2.dtype)

    Hovvo += u[o, v, v, o]
    Hovvo -= np.einsum("fbjn,nmfe->mbej", t2, u[o, o, v, v])
    Hovvo += np.einsum("bfjn,nmfe->mbej", t2, Loovv)
    return Hovvo


def build_Hovov(u, t2, o, v):
    """
    <mb|Hbar|je> = - <mb|Hbar|ej> = <mb||je> + t_jf <bm||ef> - t_nb <mn||je>
                   - (t_jnfb + t_jf t_nb) <nm||ef>
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hovov = np.zeros((nocc, nvirt, nocc, nvirt), dtype=t2.dtype)

    Hovov += u[o, v, o, v]
    Hovov -= np.einsum("fbjn,nmef->mbje", t2, u[o, o, v, v])
    return Hovov


def build_Goo(t2, l2, np):
    Goo = 0
    Goo += np.einsum("abmj,ijab->mi", t2, l2)
    return Goo


def build_Gvv(t2, l2, np):
    Gvv = 0
    Gvv -= np.einsum("ijab,ebij->ae", l2, t2)
    return Gvv
