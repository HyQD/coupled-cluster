from opt_einsum import contract


def compute_time_dependent_overlap(
    t_1_t1,
    t_2_t1,
    l_1_t1,
    l_2_t1,
    t_0_t2,
    t_1_t2,
    t_2_t2,
    l_1_t2,
    l_2_t2,
    np,
    use_old=False,
):

    """
    t_1_t1 refer to the t_1 amplitude at time t1
    t_1_t1 refer to the t_1 amplitude at time t2
    and so on.
    """

    psi_t1_t2 = 1

    psi_t1_t2 += 0.5 * contract("ijab,abij->", l_2_t1, t_2_t2)
    psi_t1_t2 -= 0.5 * contract("ijab,abij->", l_2_t1, t_2_t1)

    psi_t1_t2 += 0.5 * contract(
        "ai,bj,ijab->", t_1_t1, t_1_t1, l_2_t1, optimize=True
    )

    psi_t1_t2 += 0.5 * contract(
        "ai,bj,ijab->", t_1_t2, t_1_t2, l_2_t1, optimize=True
    )

    psi_t1_t2 -= contract("ai,bj,ijab->", t_1_t1, t_1_t2, l_2_t1, optimize=True)

    psi_t1_t2 += contract("ia,ai->", l_1_t1, t_1_t2)
    psi_t1_t2 -= contract("ia,ai->", l_1_t1, t_1_t1)

    psi_t2_t1 = 1
    psi_t2_t1 += 0.5 * contract("ijab,abij->", l_2_t2, t_2_t1)
    psi_t2_t1 -= 0.5 * contract("ijab,abij->", l_2_t2, t_2_t2)

    psi_t2_t1 += 0.5 * contract(
        "ai,bj,ijab->", t_1_t2, t_1_t2, l_2_t2, optimize=True
    )

    psi_t2_t1 += 0.5 * contract(
        "ai,bj,ijab->", t_1_t1, t_1_t1, l_2_t2, optimize=True
    )

    psi_t2_t1 -= contract("ai,bj,ijab->", t_1_t2, t_1_t1, l_2_t2, optimize=True)

    psi_t2_t1 += contract("ia,ai->", l_1_t2, t_1_t1)
    psi_t2_t1 -= contract("ia,ai->", l_1_t2, t_1_t2)

    auto_corr = 0.5 * (
        psi_t1_t2 * np.exp(t_0_t2) + (psi_t2_t1 * np.exp(-t_0_t2)).conj()
    )

    return auto_corr
