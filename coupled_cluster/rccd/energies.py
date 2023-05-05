from opt_einsum import contract


def compute_rccd_correlation_energy(f, u, t, o, v, np):
    r"""Ground state correlation energy for the coupled cluster doubles method

    \Delta E_{CCD} = 0.25 * t^{ab}_{ij} u^{ij}_{ab}.
    """

    rccd_corr = 2 * contract("abij,ijab->", t, u[o, o, v, v])

    rccd_corr -= contract("abij,ijba->", t, u[o, o, v, v])

    return rccd_corr


def compute_lagrangian_functional(f, u, t, l, o, v, np):
    energy = -contract("ijab,acik,kbjc->", l, t, u[o, v, o, v])

    energy -= contract("ijab,acki,kbcj->", l, t, u[o, v, v, o])

    energy -= contract("ijba,acki,kbjc->", l, t, u[o, v, o, v])

    energy += 2 * contract("ijab,acik,kbcj->", l, t, u[o, v, v, o])

    energy -= contract("ij,jkab,abik->", f[o, o], l, t)

    energy += 0.5 * contract("ijab,abkm,kmij->", l, t, u[o, o, o, o])

    energy += 0.5 * contract("ijab,cdij,abcd->", l, t, u[v, v, v, v])

    energy += contract("ab,ijac,bcij->", f[v, v], l, t)

    energy += 0.5 * contract("ijab,abkm,cdij,kmcd->", l, t, t, u[o, o, v, v])

    energy += 0.5 * contract("ijab,acki,bdmj,kmcd->", l, t, t, u[o, o, v, v])

    energy += 0.5 * contract("ijab,admj,bcki,kmdc->", l, t, t, u[o, o, v, v])

    energy += contract("ijab,abik,cdjm,kmdc->", l, t, t, u[o, o, v, v])

    energy += contract("ijab,acik,bdmj,kmdc->", l, t, t, u[o, o, v, v])

    energy += 2.0 * contract("ijab,acik,bdjm,kmcd->", l, t, t, u[o, o, v, v])

    energy += contract("ijab,acij,bdkm,kmdc->", l, t, t, u[o, o, v, v])

    energy -= 2 * contract("ijab,abik,cdjm,kmcd->", l, t, t, u[o, o, v, v])

    energy -= 2 * contract("ijab,acij,bdkm,kmcd->", l, t, t, u[o, o, v, v])

    energy -= 2 * contract("ijab,acik,bdmj,kmcd->", l, t, t, u[o, o, v, v])

    energy -= contract("ijab,acik,bdjm,kmdc->", l, t, t, u[o, o, v, v])

    energy += 0.5 * contract("ijab,abij->", l, u[v, v, o, o])

    energy -= contract("abij,ijba->", t, u[o, o, v, v])

    energy += 2 * contract("abij,ijab->", t, u[o, o, v, v])

    return energy
