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

from opt_einsum import contract


def compute_l_2_amplitudes(f, u, t2, l2, o, v, np, out=None):

    Loovv = build_Loovv(u, o, v, np)

    Hoo = build_Hoo(f, Loovv, t2, o, v, np)
    Hvv = build_Hvv(f, Loovv, t2, o, v, np)

    Hovvo = build_Hovvo(u, Loovv, t2, o, v, np)
    Hovov = build_Hovov(u, t2, o, v, np)

    Hoooo = build_Hoooo(u, t2, o, v, np)

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    r_l2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t2.dtype)

    r_l2 += Loovv
    r_l2 += contract("ijeb,ea->ijab", l2, Hvv)
    r_l2 -= contract("im,mjab->ijab", Hoo, l2)

    r_l2 += 0.5 * contract("ijmn,mnab->ijab", Hoooo, l2)

    ###########################################################################
    # Avoid explicit construction og Hvvvv
    r_l2 += 0.5 * contract("ijef, efab->ijab", l2, u[v, v, v, v])

    tmp_ijmn = contract("ijef, efmn->ijmn", l2, t2)
    r_l2 += 0.5 * contract("ijmn, mnab->ijab", tmp_ijmn, u[o, o, v, v])
    ###########################################################################

    r_l2 += 2 * contract("ieam,mjeb->ijab", Hovvo, l2)
    r_l2 -= contract("iema,mjeb->ijab", Hovov, l2)
    r_l2 -= contract("mibe,jema->ijab", l2, Hovov)
    r_l2 -= contract("mieb,jeam->ijab", l2, Hovvo)

    r_l2 += contract("ijeb,ae->ijab", Loovv, build_Gvv(t2, l2, np))
    r_l2 -= contract("mi,mjab->ijab", build_Goo(t2, l2, np), Loovv)

    r_l2 += r_l2.swapaxes(0, 1).swapaxes(2, 3)

    return r_l2


def build_Loovv(u, o, v, np):
    tmp = u[o, o, v, v].copy()
    Loovv = 2.0 * tmp - tmp.swapaxes(2, 3)
    return Loovv


def build_Hoo(f, Loovv, t2, o, v, np):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hoo = np.zeros((nocc, nocc), dtype=t2.dtype)

    Hoo += f[o, o]
    Hoo += contract("efin,mnef->mi", t2, Loovv)
    return Hoo


def build_Hvv(f, Loovv, t2, o, v, np):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hvv = np.zeros((nvirt, nvirt), dtype=t2.dtype)

    Hvv += f[v, v]
    Hvv -= contract("famn,mnfe->ae", t2, Loovv)
    return Hvv


def build_Hoooo(u, t2, o, v, np):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hoooo = np.zeros((nocc, nocc, nocc, nocc), dtype=t2.dtype)

    Hoooo += u[o, o, o, o]
    Hoooo += contract("efij,mnef->mnij", t2, u[o, o, v, v])
    return Hoooo


def build_Hvvvv(u, t2, o, v, np):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hvvvv = np.zeros((nvirt, nvirt, nvirt, nvirt), dtype=t2.dtype)

    Hvvvv += u[v, v, v, v]
    Hvvvv += contract("abmn,mnef->abef", t2, u[o, o, v, v])
    return Hvvvv


def build_Hovvo(u, Loovv, t2, o, v, np):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hovvo = np.zeros((nocc, nvirt, nvirt, nocc), dtype=t2.dtype)

    Hovvo += u[o, v, v, o]
    Hovvo -= contract("fbjn,nmfe->mbej", t2, u[o, o, v, v])
    Hovvo += contract("bfjn,nmfe->mbej", t2, Loovv)
    return Hovvo


def build_Hovov(u, t2, o, v, np):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hovov = np.zeros((nocc, nvirt, nocc, nvirt), dtype=t2.dtype)

    Hovov += u[o, v, o, v]
    Hovov -= contract("fbjn,nmef->mbje", t2, u[o, o, v, v])
    return Hovov


def build_Goo(t2, l2, np):
    Goo = 0
    Goo += contract("abmj,ijab->mi", t2, l2)
    return Goo


def build_Gvv(t2, l2, np):
    Gvv = 0
    Gvv -= contract("ijab,ebij->ae", l2, t2)
    return Gvv
