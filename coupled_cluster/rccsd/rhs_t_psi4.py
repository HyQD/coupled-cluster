from coupled_cluster.rccsd.t_intermediates_psi4 import *


def compute_t_1_amplitudes(f, u, t1, t2, o, v, np, out=None):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]

    ### Build OEI intermediates
    Fae = build_Fae(f, u, t1, t2, o, v)
    Fmi = build_Fmi(f, u, t1, t2, o, v)
    Fme = build_Fme(f, u, t1, o, v)

    #### Build residual of T1 equations by spin adaption of  Eqn 1:
    r_T1 = np.zeros((nvirt, nocc), dtype=t1.dtype)
    r_T1 += f[v, o]
    r_T1 += ndot("ei,ae->ai", t1, Fae)
    r_T1 -= ndot("am,mi->ai", t1, Fmi)
    r_T1 += ndot("aeim,me->ai", t2, Fme, prefactor=2.0)
    r_T1 += ndot("eaim,me->ai", t2, Fme, prefactor=-1.0)
    r_T1 += ndot("fn,nafi->ai", t1, u[o, v, v, o], prefactor=2.0)
    r_T1 += ndot("fn,naif->ai", t1, u[o, v, o, v], prefactor=-1.0)
    r_T1 += ndot("efmi,maef->ai", t2, u[o, v, v, v], prefactor=2.0)
    r_T1 += ndot("femi,maef->ai", t2, u[o, v, v, v], prefactor=-1.0)
    r_T1 -= ndot("aemn,nmei->ai", t2, u[o, o, v, o], prefactor=2.0)
    r_T1 -= ndot("aemn,nmie->ai", t2, u[o, o, o, v], prefactor=-1.0)
    return r_T1


def compute_t_2_amplitudes(f, u, t1, t2, o, v, np, out=None):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]

    ### Build OEI intermediates
    # TODO: This should be handled more smoothly in the sense that
    # they are compute in compute_t1_amplitudes as well
    Fae = build_Fae(f, u, t1, t2, o, v)
    Fmi = build_Fmi(f, u, t1, t2, o, v)
    Fme = build_Fme(f, u, t1, o, v)

    r_T2 = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t1.dtype)
    r_T2 += u[v, v, o, o]

    tmp = ndot("aeij,be->abij", t2, Fae)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P(ab) {-0.5 * t_ijae t_mb Fme_me} -> P^(ab)_(ij) {-0.5 * t_ijae t_mb Fme_me}
    tmp = ndot("bm,me->be", t1, Fme)
    first = ndot("aeij,be->abij", t2, tmp, prefactor=0.5)
    r_T2 -= first
    r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

    # P(ij) {-t_imab Fmi_mj}  ->  P^(ab)_(ij) {-t_imab Fmi_mj}
    tmp = ndot("abim,mj->abij", t2, Fmi, prefactor=1.0)
    r_T2 -= tmp
    r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P(ij) {-0.5 * t_imab t_je Fme_me}  -> P^(ab)_(ij) {-0.5 * t_imab t_je Fme_me}
    tmp = ndot("ej,me->jm", t1, Fme)
    first = ndot("abim,jm->abij", t2, tmp, prefactor=0.5)
    r_T2 -= first
    r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

    # Build TEI Intermediates
    tmp_tau = build_tau(t1, t2, o, v)
    Wmnij = build_Wmnij(u, t1, t2, o, v)
    Wmbej = build_Wmbej(u, t1, t2, o, v)
    Wmbje = build_Wmbje(u, t1, t2, o, v)
    Zmbij = build_Zmbij(u, t1, t2, o, v)

    # 0.5 * tau_mnab Wmnij_mnij  -> tau_mnab Wmnij_mnij
    # This also includes the last term in 0.5 * tau_ijef Wabef
    # as Wmnij is modified to include this contribution.
    r_T2 += ndot("abmn,mnij->abij", tmp_tau, Wmnij, prefactor=1.0)

    # Wabef used in eqn 2 of reference 1 is very expensive to build and store, so we have
    # broken down the term , 0.5 * tau_ijef * Wabef (eqn. 7) into different components
    # The last term in the contraction 0.5 * tau_ijef * Wabef is already accounted
    # for in the contraction just above.

    # First term: 0.5 * tau_ijef <ab||ef> -> tau_ijef <ab|ef>
    r_T2 += ndot("efij,abef->abij", tmp_tau, u[v, v, v, v], prefactor=1.0)

    # Second term: 0.5 * tau_ijef (-P(ab) t_mb <am||ef>)  -> -P^(ab)_(ij) {t_ma * Zmbij_mbij}
    # where Zmbij_mbij = <mb|ef> * tau_ijef
    tmp = ndot("am,mbij->abij", t1, Zmbij)
    r_T2 -= tmp
    r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P(ij)P(ab) t_imae Wmbej -> Broken down into three terms below
    # First term: P^(ab)_(ij) {(t_imae - t_imea)* Wmbej_mbej}
    tmp = ndot("aeim,mbej->abij", t2, Wmbej, prefactor=1.0)
    tmp += ndot("eaim,mbej->abij", t2, Wmbej, prefactor=-1.0)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # Second term: P^(ab)_(ij) t_imae * (Wmbej_mbej + Wmbje_mbje)
    tmp = ndot("aeim,mbej->abij", t2, Wmbej, prefactor=1.0)
    tmp += ndot("aeim,mbje->abij", t2, Wmbje, prefactor=1.0)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # Third term: P^(ab)_(ij) t_mjae * Wmbje_mbie
    tmp = ndot("aemj,mbie->abij", t2, Wmbje, prefactor=1.0)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # -P(ij)P(ab) {-t_ie * t_ma * <mb||ej>} -> P^(ab)_(ij) {-t_ie * t_ma * <mb|ej>
    #                                                      + t_ie * t_mb * <ma|je>}
    tmp = ndot("ei,am->imea", t1, t1)
    tmp1 = ndot("imea,mbej->abij", tmp, u[o, v, v, o])
    r_T2 -= tmp1
    r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)
    tmp = ndot("ei,bm->imeb", t1, t1)
    tmp1 = ndot("imeb,maje->abij", tmp, u[o, v, o, v])
    r_T2 -= tmp1
    r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)

    # P(ij) {t_ie <ab||ej>} -> P^(ab)_(ij) {t_ie <ab|ej>}
    tmp = ndot("ei,abej->abij", t1, u[v, v, v, o], prefactor=1.0)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P(ab) {-t_ma <mb||ij>} -> P^(ab)_(ij) {-t_ma <mb|ij>}
    tmp = ndot("am,mbij->abij", t1, u[o, v, o, o], prefactor=1.0)
    r_T2 -= tmp
    r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

    return r_T2
