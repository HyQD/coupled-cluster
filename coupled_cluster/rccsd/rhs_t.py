"""
Copyright (c) 2014-2018, The Psi4NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the Psi4NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Modified from the original source code:
    https://github.com/psi4/psi4numpy/blob/cbef6ddcb32ccfbf773befea6dc4aaae2b428776/Coupled-Cluster/RHF/helper_ccenergy.py
"""

from opt_einsum import contract


def compute_t_1_amplitudes(f, u, t1, t2, o, v, np, out=None):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]

    ### Build OEI intermediates

    Fae = build_Fae(f, u, t1, t2, o, v, np)
    Fmi = build_Fmi(f, u, t1, t2, o, v, np)
    Fme = build_Fme(f, u, t1, o, v, np)

    #### Build residual of T1 equations by spin adaption of  Eqn 1:
    r_T1 = np.zeros((nvirt, nocc), dtype=t1.dtype)
    r_T1 += f[v, o]
    r_T1 += contract("ei,ae->ai", t1, Fae)
    r_T1 -= contract("am,mi->ai", t1, Fmi)
    r_T1 += 2 * contract("aeim,me->ai", t2, Fme)
    r_T1 -= contract("eaim,me->ai", t2, Fme)
    r_T1 += 2 * contract("fn,nafi->ai", t1, u[o, v, v, o])
    r_T1 -= contract("fn,naif->ai", t1, u[o, v, o, v])
    r_T1 += 2 * contract("efmi,maef->ai", t2, u[o, v, v, v])
    r_T1 -= contract("femi,maef->ai", t2, u[o, v, v, v])
    r_T1 -= 2 * contract("aemn,nmei->ai", t2, u[o, o, v, o])
    r_T1 += contract("aemn,nmie->ai", t2, u[o, o, o, v])
    return r_T1


def compute_t_2_amplitudes(f, u, t1, t2, o, v, np, out=None):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]

    ### Build OEI intermediates
    # TODO: This should be handled more smoothly in the sense that
    # they are compute in compute_t1_amplitudes as well

    Fae = build_Fae(f, u, t1, t2, o, v, np)
    Fmi = build_Fmi(f, u, t1, t2, o, v, np)
    Fme = build_Fme(f, u, t1, o, v, np)

    r_T2 = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t1.dtype)
    r_T2 += u[v, v, o, o]

    tmp = contract("aeij,be->abij", t2, Fae)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P(ab) {-0.5 * t_ijae t_mb Fme_me} -> P^(ab)_(ij) {-0.5 * t_ijae t_mb Fme_me}
    tmp = contract("bm,me->be", t1, Fme)
    first = 0.5 * contract("aeij,be->abij", t2, tmp)
    r_T2 -= first
    r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

    # P(ij) {-t_imab Fmi_mj}  ->  P^(ab)_(ij) {-t_imab Fmi_mj}
    tmp = contract("abim,mj->abij", t2, Fmi)
    r_T2 -= tmp
    r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P(ij) {-0.5 * t_imab t_je Fme_me}  -> P^(ab)_(ij) {-0.5 * t_imab t_je Fme_me}
    tmp = contract("ej,me->jm", t1, Fme)
    first = 0.5 * contract("abim,jm->abij", t2, tmp)
    r_T2 -= first
    r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

    # Build TEI Intermediates
    tmp_tau = build_tau(t1, t2, o, v, np)

    Wmnij = build_Wmnij(u, t1, t2, o, v, np)
    Wmbej = build_Wmbej(u, t1, t2, o, v, np)
    Wmbje = build_Wmbje(u, t1, t2, o, v, np)
    Zmbij = build_Zmbij(u, t1, t2, o, v, np)

    # 0.5 * tau_mnab Wmnij_mnij  -> tau_mnab Wmnij_mnij
    # This also includes the last term in 0.5 * tau_ijef Wabef
    # as Wmnij is modified to include this contribution.
    r_T2 += contract("abmn,mnij->abij", tmp_tau, Wmnij)

    # Wabef used in eqn 2 of reference 1 is very expensive to build and store, so we have
    # broken down the term , 0.5 * tau_ijef * Wabef (eqn. 7) into different components
    # The last term in the contraction 0.5 * tau_ijef * Wabef is already accounted
    # for in the contraction just above.

    # First term: 0.5 * tau_ijef <ab||ef> -> tau_ijef <ab|ef>
    r_T2 += contract("efij,abef->abij", tmp_tau, u[v, v, v, v])

    # Second term: 0.5 * tau_ijef (-P(ab) t_mb <am||ef>)  -> -P^(ab)_(ij) {t_ma * Zmbij_mbij}
    # where Zmbij_mbij = <mb|ef> * tau_ijef
    tmp = contract("am,mbij->abij", t1, Zmbij)
    r_T2 -= tmp
    r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P(ij)P(ab) t_imae Wmbej -> Broken down into three terms below
    # First term: P^(ab)_(ij) {(t_imae - t_imea)* Wmbej_mbej}
    tmp = contract("aeim,mbej->abij", t2, Wmbej)
    tmp -= contract("eaim,mbej->abij", t2, Wmbej)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # Second term: P^(ab)_(ij) t_imae * (Wmbej_mbej + Wmbje_mbje)
    tmp = contract("aeim,mbej->abij", t2, Wmbej)
    tmp += contract("aeim,mbje->abij", t2, Wmbje)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # Third term: P^(ab)_(ij) t_mjae * Wmbje_mbie
    tmp = contract("aemj,mbie->abij", t2, Wmbje)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # -P(ij)P(ab) {-t_ie * t_ma * <mb||ej>} -> P^(ab)_(ij) {-t_ie * t_ma * <mb|ej>
    #                                                      + t_ie * t_mb * <ma|je>}
    tmp = contract("ei,am->imea", t1, t1)
    tmp1 = contract("imea,mbej->abij", tmp, u[o, v, v, o])
    r_T2 -= tmp1
    r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)
    tmp = contract("ei,bm->imeb", t1, t1)
    tmp1 = contract("imeb,maje->abij", tmp, u[o, v, o, v])
    r_T2 -= tmp1
    r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)

    # P(ij) {t_ie <ab||ej>} -> P^(ab)_(ij) {t_ie <ab|ej>}
    tmp = contract("ei,abej->abij", t1, u[v, v, v, o])
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P(ab) {-t_ma <mb||ij>} -> P^(ab)_(ij) {-t_ma <mb|ij>}
    tmp = contract("am,mbij->abij", t1, u[o, v, o, o])
    r_T2 -= tmp
    r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

    return r_T2


def build_tilde_tau(t1, t2, o, v, np):
    ttau = t2.copy()
    tmp = 0.5 * contract("ai,bj->abij", t1, t1)
    ttau += tmp
    return ttau


def build_tau(t1, t2, o, v, np):
    ttau = t2.copy()
    tmp = contract("ai,bj->abij", t1, t1)
    ttau += tmp
    return ttau


def build_Fae(f, u, t1, t2, o, v, np):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Fae = np.zeros((nvirt, nvirt), dtype=t1.dtype)

    Fae += f[v, v]
    Fae -= 0.5 * contract("me,am->ae", f[o, v], t1)
    Fae += 2 * contract("fm,mafe->ae", t1, u[o, v, v, v])
    Fae -= contract("fm,maef->ae", t1, u[o, v, v, v])
    Fae -= 2 * contract(
        "afmn,mnef->ae", build_tilde_tau(t1, t2, o, v, np), u[o, o, v, v]
    )
    Fae += contract(
        "afmn,mnfe->ae", build_tilde_tau(t1, t2, o, v, np), u[o, o, v, v]
    )
    return Fae


def build_Fmi(f, u, t1, t2, o, v, np):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Fmi = np.zeros((nocc, nocc), dtype=t1.dtype)

    Fmi += f[o, o]
    Fmi += 0.5 * contract("ei,me->mi", t1, f[o, v])
    Fmi += 2 * contract("en,mnie->mi", t1, u[o, o, o, v])
    Fmi -= contract("en,mnei->mi", t1, u[o, o, v, o])
    Fmi += 2 * contract(
        "efin,mnef->mi", build_tilde_tau(t1, t2, o, v, np), u[o, o, v, v]
    )
    Fmi -= contract(
        "efin,mnfe->mi", build_tilde_tau(t1, t2, o, v, np), u[o, o, v, v]
    )
    return Fmi


def build_Fme(f, u, t1, o, v, np):
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Fme = np.zeros((nocc, nvirt), dtype=t1.dtype)
    Fme += f[o, v]
    Fme += 2 * contract("fn,mnef->me", t1, u[o, o, v, v])
    Fme -= contract("fn,mnfe->me", t1, u[o, o, v, v])
    return Fme


def build_Wmnij(u, t1, t2, o, v, np):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Wmnij = np.zeros((nocc, nocc, nocc, nocc), dtype=t1.dtype)

    Wmnij += u[o, o, o, o]
    Wmnij += contract("ej,mnie->mnij", t1, u[o, o, o, v])
    Wmnij += contract("ei,mnej->mnij", t1, u[o, o, v, o])
    # prefactor of 1 instead of 0.5 below to fold the last term of
    # 0.5 * tau_ijef Wabef in Wmnij contraction: 0.5 * tau_mnab Wmnij_mnij
    Wmnij += contract(
        "efij,mnef->mnij", build_tau(t1, t2, o, v, np), u[o, o, v, v]
    )
    return Wmnij


def build_Wmbej(u, t1, t2, o, v, np):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Wmbej = np.zeros((nocc, nvirt, nvirt, nocc), dtype=t1.dtype)

    Wmbej += u[o, v, v, o]
    Wmbej += contract("fj,mbef->mbej", t1, u[o, v, v, v])
    Wmbej -= contract("bn,mnej->mbej", t1, u[o, o, v, o])
    tmp = 0.5 * t2
    tmp += contract("fj,bn->fbjn", t1, t1)
    Wmbej -= contract("fbjn,mnef->mbej", tmp, u[o, o, v, v])
    Wmbej += contract("fbnj,mnef->mbej", t2, u[o, o, v, v])
    Wmbej -= 0.5 * contract("fbnj,mnfe->mbej", t2, u[o, o, v, v])
    return Wmbej


def build_Wmbje(u, t1, t2, o, v, np):

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Wmbje = np.zeros((nocc, nvirt, nocc, nvirt), dtype=t1.dtype)

    Wmbje += -1.0 * u[o, v, o, v]
    Wmbje -= contract("fj,mbfe->mbje", t1, u[o, v, v, v])
    Wmbje += contract("bn,mnje->mbje", t1, u[o, o, o, v])
    tmp = 0.5 * t2
    tmp += contract("fj,bn->fbjn", t1, t1)
    Wmbje += contract("fbjn,mnfe->mbje", tmp, u[o, o, v, v])
    return Wmbje


def build_Zmbij(u, t1, t2, o, v, np):
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Zmbij = np.zeros((nocc, nvirt, nocc, nocc), dtype=t1.dtype)

    Zmbij += contract(
        "mbef,efij->mbij", u[o, v, v, v], build_tau(t1, t2, o, v, np)
    )
    return Zmbij
