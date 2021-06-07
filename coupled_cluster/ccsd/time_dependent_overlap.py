from opt_einsum import contract


def compute_time_dependent_overlap(
    t_1_0, t_2_0, l_1_0, l_2_0, t_0, t_1, t_2, l_1, l_2, np, use_old=False
):
    psi_t_0 = 1
    psi_t_0 += contract("ia, ai ->", l_1, t_1_0)
    psi_t_0 -= contract("ia, ai ->", l_1, t_1)
    psi_t_0 += 0.25 * contract("ijab, abij ->", l_2, t_2_0)
    psi_t_0 -= 0.5 * contract(
        "ijab, aj, bi ->", l_2, t_1_0, t_1_0, optimize=True
    )
    psi_t_0 -= contract("ijab, ai, bj ->", l_2, t_1, t_1_0, optimize=True)
    psi_t_0 -= 0.5 * contract("ijab, aj, bi ->", l_2, t_1, t_1, optimize=True)
    psi_t_0 -= 0.25 * contract("ijab, abij ->", l_2, t_2)

    psi_0_t = 1
    psi_0_t += contract("ia, ai ->", l_1_0, t_1)
    psi_0_t -= contract("ia, ai ->", l_1_0, t_1_0)
    psi_0_t += 0.25 * contract("ijab, abij ->", l_2_0, t_2)
    psi_0_t -= 0.5 * contract(
        "ijab, aj, bi ->", l_2_0, t_1_0, t_1_0, optimize=True
    )
    psi_0_t -= contract("ijab, ai, bj ->", l_2_0, t_1, t_1_0)
    psi_0_t -= 0.5 * contract("ijab, aj, bi ->", l_2_0, t_1, t_1, optimize=True)
    psi_0_t -= 0.25 * contract("ijab, abij ->", l_2_0, t_2_0)

    # This computation is taken from Pedersen & Kvaal (2018), eq 18
    # auto_corr = 0.5 * (psi_t_0 + psi_0_t.conj())
    auto_corr = 0.5 * (psi_t_0 * np.exp(-t_0) + (psi_0_t * np.exp(t_0)).conj())
    auto_corr = np.abs(auto_corr) ** 2

    if use_old:
        auto_corr = psi_t_0 * psi_0_t

    return auto_corr
