# Labelling of the different terms comes from the book "Many-Body Methods in
# Chemistry and Physics" by I. Shavitt and R. J. Bartlett.


# Diagrams for CC2 amplitude equations

import coupled_cluster.ccs.rhs_t as ccs_t
import coupled_cluster.ccd.rhs_t as ccd_t


def compute_t_1_amplitudes(F, F_t, W_t, t1, t2, o, v, np, intermediates=None, out=None):

    """
    if out is None:
        out = np.zeros_like(t_1)
    """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]

    tau0 = np.zeros((nocc, nvirt, nvirt, nvirt), dtype=t1.dtype)
    tau0 += 2 * np.einsum("iabc->iabc", W_t[o, v, v, v], optimize=True)
    path1 = np.einsum_path("iabc->iabc", W_t[o, v, v, v], optimize=True)
    print(path1[1])
    tau0 -= np.einsum("iacb->iabc", W_t[o, v, v, v], optimize=True)
    path2 = np.einsum_path("iacb->iabc", W_t[o, v, v, v], optimize=True)
    print(path2[1])
    Omega = np.zeros((nvirt, nocc), dtype=t1.dtype)
    Omega += np.einsum("bcji,jabc->ai", t2, tau0, optimize=True)
    path3 = np.einsum_path("bcji,jabc->ai", t2, tau0, optimize=True)
    print(path3[1])
    del tau0
    tau2 = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)
    tau2 -= np.einsum("jkia->ijka", W_t[o, o, o, v], optimize=True)
    path4 = np.einsum_path("jkia->ijka", W_t[o, o, o, v], optimize=True)
    print(path4[1])
    tau2 += 2 * np.einsum("kjia->ijka", W_t[o, o, o, v], optimize=True)
    path5 = 2 * np.einsum_path("kjia->ijka", W_t[o, o, o, v], optimize=True)
    print(path5[1])
    Omega -= np.einsum("bajk,ijkb->ai", t2, tau2, optimize=True)
    path_6 = np.einsum_path("bajk,ijkb->ai", t2, tau2, optimize=True)
    print(path_6[1])
    del tau2
    tau5 = np.zeros((nocc, nvirt), dtype=t1.dtype)
    tau5 += np.einsum("ia->ia", F_t[o, v], optimize=True)
    path_7 = np.einsum_path("ia->ia", F_t[o, v], optimize=True)
    print(path_7[1])
    tau6 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau6 += 2 * np.einsum("abji->ijab", t2, optimize=True)
    path_8 = 2 * np.einsum_path("abji->ijab", t2, optimize=True)
    print(path_8[1])
    tau6 -= np.einsum("baji->ijab", t2, optimize=True)
    path_9 = np.einsum_path("baji->ijab", t2, optimize=True)
    Omega += np.einsum("jb,jiab->ai", tau5, tau6, optimize=True)
    path_10 = np.einsum_path("jb,jiab->ai", tau5, tau6, optimize=True)
    print(path_10)
    del tau5
    del tau6
    Omega += np.einsum("ai->ai", F_t[v, o], optimize=True)
    path_11 = np.einsum_path("ai->ai", F_t[v, o], optimize=True)
    print(path_11[1])

    return Omega


def compute_t_2_amplitudes(F, F_t, W_t, t1, t2, o, v, np, intermediates=None, out=None):
    """
    if out is None:
        out = np.zeros_like(t_2)
    """

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Omegb = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t1.dtype)
    tau2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau2 += np.einsum("ki,abjk->ijab", F[o, o], t2, optimize=True)
    path_1 = np.einsum_path("ki,abjk->ijab", F[o, o], t2, optimize=True)
    print(path_1)
    tau16 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau16 -= np.einsum("ijab->ijab", tau2, optimize=True)
    path_2 = np.einsum_path("ijab->ijab", tau2, optimize=True)
    print(path_2)
    del tau2
    tau3 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau3 += np.einsum("ac,bcij->ijab", F[v, v], t2, optimize=True)
    path_3 = np.einsum_path("ac,bcij->ijab", F[v, v], t2, optimize=True)
    print(path_3)
    tau16 += np.einsum("ijab->ijab", tau3, optimize=True)
    path_4 = np.einsum_path("ijab->ijab", tau3, optimize=True)
    print(path_4[1])
    del tau3
    Omegb += np.einsum("ijba->abij", tau16, optimize=True)
    path_5 = np.einsum_path("ijba->abij", tau16, optimize=True)
    print(path_5)
    Omegb += np.einsum("jiab->abij", tau16, optimize=True)
    path_6 = np.einsum_path("jiab->abij", tau16, optimize=True)
    print(path_6)
    del tau16

    Omegb += np.einsum("baji->abij", W_t[v, v, o, o], optimize=True)  # Keep this
    path_7 = np.einsum_path("baji->abij", W_t[v, v, o, o], optimize=True)
    print(path_7)

    return Omegb
