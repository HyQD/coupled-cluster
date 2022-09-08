from opt_einsum import contract


def compute_rccd_correlation_energy(f, u, t, o, v, np):
    r"""Ground state correlation energy for the coupled cluster doubles method

    \Delta E_{CCD} = 0.25 * t^{ab}_{ij} u^{ij}_{ab}.
    """

    rccd_corr = 2 * contract("abij,ijab->", t, u[o, o, v, v])

    rccd_corr -= contract("abij,ijba->", t, u[o, o, v, v])

    return rccd_corr
