def compute_t_2_amplitudes(f, u, t2, o, v, np, out=None):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]

    Fae = build_Fae(f, u, t2, o, v)
    Fmi = build_Fmi(f, u, t2, o, v)

    r_T2 = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t2.dtype)
    r_T2 += u[v, v, o, o]

    tmp = np.einsum("aeij,be->abij", t2, Fae)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    tmp = np.einsum("abim,mj->abij", t2, Fmi)
    r_T2 -= tmp
    r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

    Wmnij = build_Wmnij(u, t2, o, v)
    Wmbej = build_Wmbej(u, t2, o, v)
    Wmbje = build_Wmbje(u, t2, o, v)

    r_T2 += np.einsum("abmn,mnij->abij", t2, Wmnij)

    r_T2 += np.einsum("efij,abef->abij", t2, u[v, v, v, v])

    tmp = np.einsum("aeim,mbej->abij", t2, Wmbej)
    tmp -= np.einsum("eaim,mbej->abij", t2, Wmbej)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    tmp = np.einsum("aeim,mbej->abij", t2, Wmbej)
    tmp += np.einsum("aeim,mbje->abij", t2, Wmbje)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    tmp = np.einsum("aemj,mbie->abij", t2, Wmbje)
    r_T2 += tmp
    r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

    return r_T2


import numpy as np


def build_Fae(f, u, t2, o, v):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Fae = np.zeros((nvirt, nvirt), dtype=t2.dtype)

    Fae += f[v, v]
    Fae -= 2 * np.einsum("afmn,mnef->ae", t2, u[o, o, v, v])
    Fae += np.einsum("afmn,mnfe->ae", t2, u[o, o, v, v])
    return Fae


def build_Fmi(f, u, t2, o, v):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Fmi = np.zeros((nocc, nocc), dtype=t2.dtype)

    Fmi += f[o, o]
    Fmi += 2 * np.einsum("efin,mnef->mi", t2, u[o, o, v, v])
    Fmi -= np.einsum("efin,mnfe->mi", t2, u[o, o, v, v])
    return Fmi


def build_Wmnij(u, t2, o, v):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Wmnij = np.zeros((nocc, nocc, nocc, nocc), dtype=t2.dtype)

    Wmnij += u[o, o, o, o]

    Wmnij += np.einsum("efij,mnef->mnij", t2, u[o, o, v, v])
    return Wmnij


def build_Wmbej(u, t2, o, v):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Wmbej = np.zeros((nocc, nvirt, nvirt, nocc), dtype=t2.dtype)

    Wmbej += u[o, v, v, o]

    Wmbej -= 0.5 * np.einsum("fbjn,mnef->mbej", t2, u[o, o, v, v])
    Wmbej += np.einsum("fbnj,mnef->mbej", t2, u[o, o, v, v])
    Wmbej -= 0.5 * np.einsum("fbnj,mnfe->mbej", t2, u[o, o, v, v])
    return Wmbej


def build_Wmbje(u, t2, o, v):

    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Wmbje = np.zeros((nocc, nvirt, nocc, nvirt), dtype=t2.dtype)

    Wmbje -= u[o, v, o, v]
    Wmbje += 0.5 * np.einsum("fbjn,mnfe->mbje", t2, u[o, o, v, v])

    return Wmbje
