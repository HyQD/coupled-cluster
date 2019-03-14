def time_dependent_overlap(t_1, t_2, l_1, l_2, t_t, l_t, np):
    # ? t_1_0, t_2_0 = _t_0
    # ? l_1_0, l_2_0 = _l_0
    psi_t_0 = 1
    psi_t_0 += np.einsum("ia, ai ->", l_1, t_1_0)
    psi_t_0 -= np.einsum("ia, ai ->", l_1, t_1)
    psi_t_0 += 0.25 * np.einsum("ijab, abij ->", l_2, t_2_0)
    psi_t_0 -= 0.5 * np.einsum(
        "ijab, aj, bi ->", l_2, t_1_0, t_1_0, optimize=True
    )
    psi_t_0 -= np.einsum(
        "ijab, ai, bj ->", l_2, t_1, t_1_0, optimize=True
    )
    psi_t_0 -= 0.5 * np.einsum(
        "ijab, aj, bi ->", l_2, t_1, t_1, optimize=True
    )
    psi_t_0 -= 0.25 * np.einsum("ijab, abij ->", l_2, t_2)
    psi_0_t = 1
    psi_0_t += np.einsum("ia, ai ->", l_1_0, t_1)
    psi_0_t -= np.einsum("ia, ai ->", l_1_0, t_1_0)
    psi_0_t += 0.25 * np.einsum("ijab, abij ->", l_2_0, t_2)
    psi_0_t -= 0.5 * np.einsum(
        "ijab, aj, bi ->", l_2_0, t_1_0, t_1_0, optimize=True
    )
    psi_0_t -= np.einsum("ijab, ai, bj ->", l_2_0, t_1, t_1_0)
    psi_0_t -= 0.5 * np.einsum(
        "ijab, aj, bi ->", l_2_0, t_1, t_1, optimize=True
    )
    psi_0_t -= 0.25 * np.einsum("ijab, abij ->", l_2_0, t_2_0)
    return psi_t_0 * psi_0_t

