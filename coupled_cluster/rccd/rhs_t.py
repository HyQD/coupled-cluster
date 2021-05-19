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


def compute_t_2_amplitudes(f, u, t2, o, v, np, out=None):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]

    Fae = build_Fae(f, u, t2, o, v, np)
    Fmi = build_Fmi(f, u, t2, o, v, np)

    r_T2 = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t2.dtype)
    r_T2 += u[v, v, o, o]

    tmp = contract("aeij,be->abij", t2, Fae)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    tmp = contract("abim,mj->abij", t2, Fmi)
    r_T2 -= tmp
    r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

    Wmnij = build_Wmnij(u, t2, o, v, np)
    Wmbej = build_Wmbej(u, t2, o, v, np)
    Wmbje = build_Wmbje(u, t2, o, v, np)

    r_T2 += contract("abmn,mnij->abij", t2, Wmnij)

    r_T2 += contract("efij,abef->abij", t2, u[v, v, v, v])

    tmp = contract("aeim,mbej->abij", t2, Wmbej)
    tmp -= contract("eaim,mbej->abij", t2, Wmbej)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    tmp = contract("aeim,mbej->abij", t2, Wmbej)
    tmp += contract("aeim,mbje->abij", t2, Wmbje)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    tmp = contract("aemj,mbie->abij", t2, Wmbje)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    return r_T2


def build_Fae(f, u, t2, o, v, np):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Fae = np.zeros((nvirt, nvirt), dtype=t2.dtype)

    Fae += f[v, v]
    Fae -= 2 * contract("afmn,mnef->ae", t2, u[o, o, v, v])
    Fae += contract("afmn,mnfe->ae", t2, u[o, o, v, v])
    return Fae


def build_Fmi(f, u, t2, o, v, np):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Fmi = np.zeros((nocc, nocc), dtype=t2.dtype)

    Fmi += f[o, o]
    Fmi += 2 * contract("efin,mnef->mi", t2, u[o, o, v, v])
    Fmi -= contract("efin,mnfe->mi", t2, u[o, o, v, v])
    return Fmi


def build_Wmnij(u, t2, o, v, np):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Wmnij = np.zeros((nocc, nocc, nocc, nocc), dtype=t2.dtype)

    Wmnij += u[o, o, o, o]

    Wmnij += contract("efij,mnef->mnij", t2, u[o, o, v, v])
    return Wmnij


def build_Wmbej(u, t2, o, v, np):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Wmbej = np.zeros((nocc, nvirt, nvirt, nocc), dtype=t2.dtype)

    Wmbej += u[o, v, v, o]

    Wmbej -= 0.5 * contract("fbjn,mnef->mbej", t2, u[o, o, v, v])
    Wmbej += contract("fbnj,mnef->mbej", t2, u[o, o, v, v])
    Wmbej -= 0.5 * contract("fbnj,mnfe->mbej", t2, u[o, o, v, v])
    return Wmbej


def build_Wmbje(u, t2, o, v, np):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Wmbje = np.zeros((nocc, nvirt, nocc, nvirt), dtype=t2.dtype)

    Wmbje -= u[o, v, o, v]
    Wmbje += 0.5 * contract("fbjn,mnfe->mbje", t2, u[o, o, v, v])

    return Wmbje
