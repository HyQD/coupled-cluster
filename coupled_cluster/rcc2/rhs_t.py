# Labelling of the different terms comes from the book "Many-Body Methods in
# Chemistry and Physics" by I. Shavitt and R. J. Bartlett.


# Diagrams for CC2 amplitude equations

from opt_einsum import contract


def compute_t_1_amplitudes(
    F, F_t, W_t, t1, t2, o, v, np, intermediates=None, out=None
):

    """
    if out is None:
        out = np.zeros_like(t_1)
    """
    nocc = t1.shape[1]
    nvirt = t1.shape[0]

    tau0 = np.zeros((nocc, nvirt, nvirt, nvirt), dtype=t1.dtype)
    tau0 += 2 * contract("iabc->iabc", W_t[o, v, v, v], optimize=True)
    tau0 -= contract("iacb->iabc", W_t[o, v, v, v], optimize=True)
    Omega = np.zeros((nvirt, nocc), dtype=t1.dtype)
    Omega += contract("bcji,jabc->ai", t2, tau0, optimize=True)
    del tau0
    tau2 = np.zeros((nocc, nocc, nocc, nvirt), dtype=t1.dtype)
    tau2 -= contract("jkia->ijka", W_t[o, o, o, v], optimize=True)
    tau2 += 2 * contract("kjia->ijka", W_t[o, o, o, v], optimize=True)
    Omega -= contract("bajk,ijkb->ai", t2, tau2, optimize=True)
    del tau2
    tau5 = np.zeros((nocc, nvirt), dtype=t1.dtype)
    tau5 += contract("ia->ia", F_t[o, v], optimize=True)
    tau6 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau6 += 2 * contract("abji->ijab", t2, optimize=True)
    tau6 -= contract("baji->ijab", t2, optimize=True)
    Omega += contract("jb,jiab->ai", tau5, tau6, optimize=True)
    del tau5
    del tau6
    Omega += contract("ai->ai", F_t[v, o], optimize=True)

    return Omega


def compute_t_2_amplitudes(
    F, F_t, W_t, t1, t2, o, v, np, intermediates=None, out=None
):
    """
    if out is None:
        out = np.zeros_like(t_2)
    """

    nocc = t1.shape[1]
    nvirt = t1.shape[0]
    Omegb = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t1.dtype)
    tau2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau2 += contract("ki,abjk->ijab", F[o, o], t2, optimize=True)
    tau16 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau16 -= contract("ijab->ijab", tau2, optimize=True)
    del tau2
    tau3 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=t1.dtype)
    tau3 += contract("ac,bcij->ijab", F[v, v], t2, optimize=True)
    tau16 += contract("ijab->ijab", tau3, optimize=True)
    del tau3
    Omegb += contract("ijba->abij", tau16, optimize=True)
    Omegb += contract("jiab->abij", tau16, optimize=True)
    del tau16
    Omegb += contract("baji->abij", W_t[v, v, o, o], optimize=True)

    return Omegb
