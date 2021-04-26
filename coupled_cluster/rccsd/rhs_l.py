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

from coupled_cluster.rccsd.cc_hbar import *

"""
l1 and l2  equations can be obtained by taking the derivative of the CCSD Lagrangian wrt. t1 and t2 amplitudes respectively.
(Einstein summation):

l1: <o|Hbar|phi^a_i>   + l_kc * <phi^c_k|Hbar|phi^a_i>   + l_klcd * <phi^cd_kl|Hbar|phi^a_i> = 0
l2: <o|Hbar|phi^ab_ij> + l_kc * <phi^c_k|Hbar|phi^ab_ij> + l_klcd * <phi^cd_kl|Hbar|phi^ab_ij> = 0

In Spin orbitals:
l1:
Hov_ia + l_ie * Hvv_ae - l_ma * Hoo_im + l_me * Hovvo_ieam + 0.5 * l_imef * Hvvvo_efam
- 0.5 * l_mnae Hovoo_iemn - Gvv_ef * Hvovv_eifa - Goo_mn * Hooov_mina = 0

where, Goo_mn = 0.5 * t_mjab * l_njab,  Gvv_ef = - 0.5 * l_ijeb * t_ijfb
Intermediates Goo and Gvv have been built to bypass the construction of
3-body Hbar terms like, l1  <-- Hvvooov_beilka * l_lkbe
                        l1  <-- Hvvooov_cbijna * l_jncb

l2:
 <ij||ab> + P(ab) l_ijae * Hov_eb - P(ij) l_imab * Hoo_jm + 0.5 * l_mnab * Hoooo_ijmn
 + 0.5 * Hvvvv_efab l_ijef + P(ij) l_ie * Hvovv_ejab - P(ab) l_ma * Hooov_ijmb
 + P(ij)P(ab) l_imae * Hovvo_jebm + P(ij)P(ab) l_ia * Hov_jb + P(ab) <ij||ae> * Gvv_be
 - P(ij) <im||ab> * Goo_mj

Here we are using the unitary group approach (UGA) to derive spin adapted equations, please refer to chapter
13 of reference 2 and notes in the current folder for more details. Lambda equations derived using UGA differ
from the regular spin factorizd equations (PSI4) as follows:
l_ia(UGA) = 2.0 * l_ia(PSI4)
l_ijab(UGA) = 2.0 * (2.0 * l_ijab - l_ijba)
The residual equations (without the preconditioner) follow the same relations as above.
Ex. the inhomogenous terms in l1 and l2 equations in UGA are 2 * Hov_ia and 2.0 * (2.0 * <ij|ab> - <ij|ba>)
respectively as opposed to Hov_ia and <ij|ab> in PSI4.
"""


def compute_l_1_amplitudes(f, u, t1, t2, l1, l2, o, v, np, out=None):

    Loovv = build_Loovv(u, o, v)
    Lvovv = build_Lvovv(u, o, v)
    Looov = build_Looov(u, o, v)

    Hoo = build_Hoo(f, Looov, Loovv, t1, t2, o, v)
    Hov = build_Hov(f, Loovv, t1, o, v)
    Hvv = build_Hvv(f, Lvovv, Loovv, t1, t2, o, v)
    Hovvo = build_Hovvo(u, Loovv, t1, t2, o, v)
    Hovov = build_Hovov(u, t1, t2, o, v)
    Hvvvo = build_Hvvvo(f, u, Loovv, Lvovv, t1, t2, o, v)
    Hovoo = build_Hovoo(f, u, Loovv, Looov, t1, t2, o, v)
    Hvovv = build_Hvovv(u, t1, o, v)
    Hooov = build_Hooov(u, t1, o, v)

    # l1 equations
    r_l1 = 2.0 * Hov
    r_l1 += np.einsum("ie,ea->ia", l1, Hvv)
    r_l1 -= np.einsum("im,ma->ia", Hoo, l1)
    r_l1 += 2 * np.einsum("ieam,me->ia", Hovvo, l1)
    r_l1 -= np.einsum("iema,me->ia", Hovov, l1)
    r_l1 += np.einsum("imef,efam->ia", l2, Hvvvo)
    r_l1 -= np.einsum("iemn,mnae->ia", Hovoo, l2)
    r_l1 -= 2 * np.einsum("eifa,ef->ia", Hvovv, build_Gvv(t2, l2, np))
    r_l1 += np.einsum("eiaf,ef->ia", Hvovv, build_Gvv(t2, l2, np))
    r_l1 -= 2 * np.einsum("mina,mn->ia", Hooov, build_Goo(t2, l2, np))
    r_l1 += np.einsum("imna,mn->ia", Hooov, build_Goo(t2, l2, np))

    return r_l1


def compute_l_2_amplitudes(f, u, t1, t2, l1, l2, o, v, np, out=None):

    ################################################
    # These intermediates are common with those used in
    # compute_l1_amplitudes
    Loovv = build_Loovv(u, o, v)
    Lvovv = build_Lvovv(u, o, v)
    Looov = build_Looov(u, o, v)

    Hoo = build_Hoo(f, Looov, Loovv, t1, t2, o, v)
    Hov = build_Hov(f, Loovv, t1, o, v)
    Hvv = build_Hvv(f, Lvovv, Loovv, t1, t2, o, v)

    Hvovv = build_Hvovv(u, t1, o, v)
    Hooov = build_Hooov(u, t1, o, v)
    Hovvo = build_Hovvo(u, Loovv, t1, t2, o, v)
    Hovov = build_Hovov(u, t1, t2, o, v)
    ################################################
    Hoooo = build_Hoooo(u, t1, t2, o, v)
    Hvvvv = build_Hvvvv(u, t1, t2, o, v)

    # l2 equations
    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    r_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)

    r_l2 += Loovv
    r_l2 += 2 * np.einsum("ia,jb->ijab", l1, Hov)
    r_l2 -= np.einsum("ja,ib->ijab", l1, Hov)
    r_l2 += np.einsum("ijeb,ea->ijab", l2, Hvv)
    r_l2 -= np.einsum("im,mjab->ijab", Hoo, l2)
    r_l2 += 0.5 * np.einsum("ijmn,mnab->ijab", Hoooo, l2)
    r_l2 += 0.5 * np.einsum("ijef,efab->ijab", l2, Hvvvv)
    r_l2 += 2 * np.einsum("ie,ejab->ijab", l1, Hvovv)
    r_l2 -= np.einsum("ie,ejba->ijab", l1, Hvovv)
    r_l2 -= 2 * np.einsum("mb,jima->ijab", l1, Hooov)
    r_l2 += np.einsum("mb,ijma->ijab", l1, Hooov)
    r_l2 += 2 * np.einsum("ieam,mjeb->ijab", Hovvo, l2)
    r_l2 -= np.einsum("iema,mjeb->ijab", Hovov, l2)
    r_l2 -= np.einsum("mibe,jema->ijab", l2, Hovov)
    r_l2 -= np.einsum("mieb,jeam->ijab", l2, Hovvo)
    r_l2 += np.einsum("ijeb,ae->ijab", Loovv, build_Gvv(t2, l2, np))
    r_l2 -= np.einsum("mi,mjab->ijab", build_Goo(t2, l2, np), Loovv)

    # Final r_l2_ijab = r_l2_ijab + r_l2_jiba
    r_l2 += r_l2.swapaxes(0, 1).swapaxes(2, 3)
    return r_l2


def build_Goo(t2, l2, np):
    Goo = 0
    Goo += np.einsum("abmj,ijab->mi", t2, l2)
    return Goo


def build_Gvv(t2, l2, np):
    Gvv = 0
    Gvv -= np.einsum("ijab,ebij->ae", l2, t2)
    return Gvv
