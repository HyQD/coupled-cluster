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
    https://github.com/psi4/psi4numpy/blob/cbef6ddcb32ccfbf773befea6dc4aaae2b428776/Coupled-Cluster/RHF/helper_cclambda.py
"""

from coupled_cluster.rccsd.cc_hbar import *
from opt_einsum import contract

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
    r_l1 += contract("ie,ea->ia", l1, Hvv)
    r_l1 -= contract("im,ma->ia", Hoo, l1)
    r_l1 += 2 * contract("ieam,me->ia", Hovvo, l1)
    r_l1 -= contract("iema,me->ia", Hovov, l1)
    r_l1 += contract("imef,efam->ia", l2, Hvvvo)
    r_l1 -= contract("iemn,mnae->ia", Hovoo, l2)
    r_l1 -= 2 * contract("eifa,ef->ia", Hvovv, build_Gvv(t2, l2, np))
    r_l1 += contract("eiaf,ef->ia", Hvovv, build_Gvv(t2, l2, np))
    r_l1 -= 2 * contract("mina,mn->ia", Hooov, build_Goo(t2, l2, np))
    r_l1 += contract("imna,mn->ia", Hooov, build_Goo(t2, l2, np))

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
    r_l2 += 2 * contract("ia,jb->ijab", l1, Hov)
    r_l2 -= contract("ja,ib->ijab", l1, Hov)
    r_l2 += contract("ijeb,ea->ijab", l2, Hvv)
    r_l2 -= contract("im,mjab->ijab", Hoo, l2)
    r_l2 += 0.5 * contract("ijmn,mnab->ijab", Hoooo, l2)
    r_l2 += 0.5 * contract("ijef,efab->ijab", l2, Hvvvv)
    r_l2 += 2 * contract("ie,ejab->ijab", l1, Hvovv)
    r_l2 -= contract("ie,ejba->ijab", l1, Hvovv)
    r_l2 -= 2 * contract("mb,jima->ijab", l1, Hooov)
    r_l2 += contract("mb,ijma->ijab", l1, Hooov)
    r_l2 += 2 * contract("ieam,mjeb->ijab", Hovvo, l2)
    r_l2 -= contract("iema,mjeb->ijab", Hovov, l2)
    r_l2 -= contract("mibe,jema->ijab", l2, Hovov)
    r_l2 -= contract("mieb,jeam->ijab", l2, Hovvo)
    r_l2 += contract("ijeb,ae->ijab", Loovv, build_Gvv(t2, l2, np))
    r_l2 -= contract("mi,mjab->ijab", build_Goo(t2, l2, np), Loovv)

    # Final r_l2_ijab = r_l2_ijab + r_l2_jiba
    r_l2 += r_l2.swapaxes(0, 1).swapaxes(2, 3)
    return r_l2


def build_Goo(t2, l2, np):
    Goo = 0
    Goo += contract("abmj,ijab->mi", t2, l2)
    return Goo


def build_Gvv(t2, l2, np):
    Gvv = 0
    Gvv -= contract("ijab,ebij->ae", l2, t2)
    return Gvv
