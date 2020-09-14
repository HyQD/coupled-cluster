from coupled_cluster.cc_helper import compute_reference_energy


def compute_ccd_ground_state_energy(f, u, t, o, v, np):
    energy = compute_reference_energy(f, u, o, v, np=np)
    energy += compute_ccd_ground_state_energy_correction(u, t, o, v, np=np)

    return energy


def compute_ccd_ground_state_energy_correction(u, t, o, v, np):
    r"""Ground state correlation energy for the coupled cluster doubles method

    \Delta E_{CCD} = 0.25 * t^{ab}_{ij} u^{ij}_{ab}.
    """

    return 0.25 * np.tensordot(
        t, u[o, o, v, v], axes=((0, 1, 2, 3), (2, 3, 0, 1))
    )


def compute_time_dependent_energy(f, u, t, l, o, v, np):
    energy = compute_reference_energy(f, u, o, v, np=np)
    energy += compute_lagrangian_functional(f, u, t, l, o, v, np=np)

    return energy


def compute_lagrangian_functional(f, u, t, l, o, v, np):
    """
    Eq [A8] in Kvaal sans the reference energy

        E_ref = h^i_i + 0.5 u^{ij}_{ij}
            = f^{i}_{i} - 0.5 u^{ij}_{ij}
    """
    # result = 0.25 * np.einsum(
    #    "lkdc,dclk->", l, u[v, v, o, o], optimize=["einsum_path", (0, 1)]
    # )

    energy = 0.25 * np.tensordot(
        l, u[v, v, o, o], axes=((0, 1, 2, 3), (2, 3, 0, 1))
    )
    # np.testing.assert_allclose(result, energy)

    # result += 0.25 * np.einsum(
    #    "dclk,lkdc->", t, u[o, o, v, v], optimize=["einsum_path", (0, 1)]
    # )

    energy += 0.25 * np.tensordot(
        t, u[o, o, v, v], axes=((0, 1, 2, 3), (2, 3, 0, 1))
    )
    # np.testing.assert_allclose(result, energy)

    # result += np.einsum(
    #    "lkdc,eckm,dmel->",
    #    l,
    #    t,
    #    u[v, o, v, o],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_ldem = np.tensordot(l, t, axes=((1, 3), (2, 1)))
    energy += np.tensordot(
        temp_ldem, u[v, o, v, o], axes=((0, 1, 2, 3), (3, 0, 2, 1))
    )
    # np.testing.assert_allclose(result, energy)

    # result += 0.5 * np.einsum(
    #    "lkdc,dckm,ml->",
    #    l,
    #    t,
    #    f[o, o],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_dckl = 0.5 * np.tensordot(t, f[o, o], axes=((3), (0)))
    energy += np.tensordot(l, temp_dckl, axes=((0, 1, 2, 3), (3, 2, 0, 1)))
    # np.testing.assert_allclose(result, energy)

    # result += 0.5 * np.einsum(
    #    "lkdc,eclk,de->",
    #    l,
    #    t,
    #    f[v, v],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_dclk = 0.5 * np.tensordot(f[v, v], t, axes=((1), (0)))
    energy += np.tensordot(temp_dclk, l, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    # np.testing.assert_allclose(result, energy)

    # result += 0.125 * np.einsum(
    #    "lkdc,dcmn,mnlk->",
    #    l,
    #    t,
    #    u[o, o, o, o],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_dclk = 0.125 * np.tensordot(t, u[o, o, o, o], axes=((2, 3), (0, 1)))
    energy += np.tensordot(temp_dclk, l, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    # np.testing.assert_allclose(result, energy)

    # result += 0.125 * np.einsum(
    #    "lkdc,eflk,dcef->",
    #    l,
    #    t,
    #    u[v, v, v, v],
    #    optimize=["einsum_path", (0, 2), (0, 1)],
    # )

    temp_dclk = 0.125 * np.tensordot(u[v, v, v, v], t, axes=((2, 3), (0, 1)))
    energy += np.tensordot(temp_dclk, l, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    # np.testing.assert_allclose(result, energy)

    # result += 0.5 * np.einsum(
    #    "lkdc,dfkn,eclm,mnef->",
    #    l,
    #    t,
    #    t,
    #    u[o, o, v, v],
    #    optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    # )

    temp_clnf = 0.5 * np.tensordot(t, u[o, o, v, v], axes=((0, 3), (2, 0)))
    temp_kdnf = np.tensordot(l, temp_clnf, axes=((0, 3), (1, 0)))
    energy += np.tensordot(t, temp_kdnf, axes=((0, 1, 2, 3), (1, 3, 0, 2)))
    # np.testing.assert_allclose(result, energy)

    # result += -0.25 * np.einsum(
    #    "lkdc,dflk,ecmn,mnef->",
    #    l,
    #    t,
    #    t,
    #    u[o, o, v, v],
    #    optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    # )

    temp_cf = -0.25 * np.tensordot(
        t, u[o, o, v, v], axes=((0, 2, 3), (2, 0, 1))
    )
    temp_lkdf = np.tensordot(l, temp_cf, axes=((3), (0)))
    energy += np.tensordot(temp_lkdf, t, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    # np.testing.assert_allclose(result, energy)

    # result += -0.125 * np.einsum(
    #    "lkdc,dckn,eflm,mnef->",
    #    l,
    #    t,
    #    t,
    #    u[o, o, v, v],
    #    optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    # )

    temp_ln = -0.125 * np.tensordot(
        t, u[o, o, v, v], axes=((0, 1, 3), (2, 3, 0))
    )
    temp_dckl = np.tensordot(t, temp_ln, axes=((3), (1)))
    energy += np.tensordot(l, temp_dckl, axes=((0, 1, 2, 3), (3, 2, 0, 1)))
    # np.testing.assert_allclose(result, energy)

    # result += -0.125 * np.einsum(
    #    "lkdc,dclm,efkn,mnef->",
    #    l,
    #    t,
    #    t,
    #    u[o, o, v, v],
    #    optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    # )

    temp_km = -0.125 * np.tensordot(l, t, axes=((0, 2, 3), (2, 0, 1)))
    temp_knef = np.tensordot(temp_km, u[o, o, v, v], axes=((1), (0)))
    energy += np.tensordot(temp_knef, t, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    # np.testing.assert_allclose(result, energy)

    # result += 0.0625 * np.einsum(
    #    "lkdc,dcmn,eflk,mnef->",
    #    l,
    #    t,
    #    t,
    #    u[o, o, v, v],
    #    optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    # )

    if o.stop >= v.stop // 2:
        temp_dcef = 0.0625 * np.tensordot(
            t, u[o, o, v, v], axes=((2, 3), (0, 1))
        )
        temp_efdc = np.tensordot(t, l, axes=((2, 3), (0, 1)))
        energy += np.tensordot(
            temp_efdc, temp_dcef, axes=((0, 1, 2, 3), (2, 3, 0, 1))
        )
    else:
        temp_lkmn = 0.0625 * np.tensordot(l, t, axes=((2, 3), (0, 1)))
        temp_mnlk = np.tensordot(u[o, o, v, v], t, axes=((2, 3), (0, 1)))
        energy += np.tensordot(
            temp_lkmn, temp_mnlk, axes=((0, 1, 2, 3), (2, 3, 0, 1))
        )
    # np.testing.assert_allclose(result, energy)

    # return result
    return energy


# def lagrangian_funtional_trans(T2, L2, F, W, o, v, np):
#    """
#   Eq [A8] in Kvaal sans the reference energy
#
#       E_ref = h^i_i + 0.5 u^{ij}_{ij}
#           = f^{i}_{i} - 0.5 u^{ij}_{ij}
#   """
#    result = 0.25 * np.einsum(
#        "lkdc,dclk->", L2, W[v, v, o, o], optimize=["einsum_path", (0, 1)]
#    )
#    result += 0.25 * np.einsum(
#        "lkdc,lkdc->", T2, W[o, o, v, v], optimize=["einsum_path", (0, 1)]
#    )
#    result += np.einsum(
#        "lkdc,kmec,dmel->",
#        L2,
#        T2,
#        W[v, o, v, o],
#        optimize=["einsum_path", (0, 1), (0, 1)],
#    )
#    result += 0.5 * np.einsum(
#        "lkdc,kmdc,ml->",
#        L2,
#        T2,
#        F[o, o],
#        optimize=["einsum_path", (0, 1), (0, 1)],
#    )
#    result += 0.5 * np.einsum(
#        "lkdc,lkec,de->",
#        L2,
#        T2,
#        F[v, v],
#        optimize=["einsum_path", (0, 1), (0, 1)],
#    )
#    result += 0.125 * np.einsum(
#        "lkdc,mndc,mnlk->",
#        L2,
#        T2,
#        W[o, o, o, o],
#        optimize=["einsum_path", (0, 1), (0, 1)],
#    )
#    result += 0.125 * np.einsum(
#        "lkdc,lkef,dcef->",
#        L2,
#        T2,
#        W[v, v, v, v],
#        optimize=["einsum_path", (0, 2), (0, 1)],
#    )
#    result += 0.5 * np.einsum(
#        "lkdc,kndf,lmec,mnef->",
#        L2,
#        T2,
#        T2,
#        W[o, o, v, v],
#        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
#    )
#    result += -0.25 * np.einsum(
#        "lkdc,lkdf,mnec,mnef->",
#        L2,
#        T2,
#        T2,
#        W[o, o, v, v],
#        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
#    )
#    result += -0.125 * np.einsum(
#        "lkdc,kndc,lmef,mnef->",
#        L2,
#        T2,
#        T2,
#        W[o, o, v, v],
#        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
#    )
#    result += -0.125 * np.einsum(
#        "lkdc,lmdc,knef,mnef->",
#        L2,
#        T2,
#        T2,
#        W[o, o, v, v],
#        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
#    )
#    result += 0.0625 * np.einsum(
#        "lkdc,mndc,lkef,mnef->",
#        L2,
#        T2,
#        T2,
#        W[o, o, v, v],
#        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
#    )
#
#    return result
