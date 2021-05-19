from opt_einsum import contract


def compute_ground_state_energy_correction(f, u, t_1, t_2, o, v, np):
    """

    f^{i}_{a} t^{a}_{i}
    + (0.25) u^{ij}_{ab} t^{ab}_{ij}
    + (0.5) t^{ij}_{ab} t^{a}_{i} t^{b}_{j}

    """

    e_corr = 2 * contract("ia,ai->", f[o, v], t_1)

    e_corr += 2 * contract("abij,ijab->", t_2, u[o, o, v, v])
    e_corr -= contract("abij,ijba->", t_2, u[o, o, v, v])

    e_corr += 2 * contract(
        "ai,bj,ijab->", t_1, t_1, u[o, o, v, v], optimize=True
    )
    e_corr -= contract("ai,bj,ijba->", t_1, t_1, u[o, o, v, v], optimize=True)

    return e_corr


def compute_time_dependent_energy(
    f, f_transform, u_transform, t_1, t_2, l_1, l_2, o, v, np
):
    energy = lagrangian_functional(
        f, f_transform, u_transform, t_1, t_2, l_1, l_2, o, v, np=np
    )
    return energy


def lagrangian_functional(
    f, f_transform, u_transform, t1, t2, l1, l2, o, v, np
):

    no = t1.shape[1]
    nv = t1.shape[0]

    no = t1.shape[1]
    nv = t1.shape[0]

    lagrangian = 0 + 0j

    I7_l1 = np.zeros((no, no), dtype=t2.dtype)
    I2_l1 = np.zeros((no, no), dtype=t2.dtype)
    I2_l1 += contract("kiba,bakj->ij", l2, t2)
    I7_l1 += contract("ij->ij", I2_l1)
    lagrangian -= contract("ji,ij->", I2_l1, f[o, o])  # f*l2*t2
    del I2_l1
    I3_l1 = np.zeros((nv, nv), dtype=t2.dtype)
    I3_l1 += contract("jica,cbji->ab", l2, t2)
    lagrangian += contract("ba,ba->", I3_l1, f[v, v])  # f*l2*t2
    I4_l1 = np.zeros((no, no, nv, nv), dtype=t2.dtype)
    I5_l1 = np.zeros((no, no, nv, nv), dtype=t2.dtype)
    I5_l1 -= contract("abji->ijab", t2)
    I5_l1 += contract("baji->ijab", t2)
    I11_l1 = np.zeros((no, no, nv, nv), dtype=t2.dtype)
    I11_l1 -= contract("kjcb,kica->ijab", I4_l1, I5_l1)
    del I4_l1
    del I5_l1
    I9_l1 = np.zeros((no, no, nv, nv), dtype=t2.dtype)
    I9_l1 -= contract("abji->ijab", t2)
    I9_l1 += 2 * contract("baji->ijab", t2)
    I14_l1 = np.zeros((no, no, nv, nv), dtype=t2.dtype)
    I39_l1 = np.zeros((no, nv), dtype=t2.dtype)
    I39_l1 += contract("jibc,jabc->ia", I9_l1, u_transform[o, v, v, v])
    I39_l1 -= contract("jkab,jkib->ia", I9_l1, u_transform[o, o, o, v])
    del I9_l1
    I11_l1 -= contract("baji->ijab", t2)
    I12_l1 = np.zeros((no, no, nv, nv), dtype=t2.dtype)
    I12_l1 -= contract("jiab->ijab", u_transform[o, o, v, v])
    I12_l1 += 2 * contract("jiba->ijab", u_transform[o, o, v, v])
    lagrangian -= contract("ijab,ijab->", I11_l1, I12_l1)  # (l2*t2*t2)*u
    del I11_l1
    del I12_l1
    I14_l1 += contract("baji->ijab", u_transform[v, v, o, o])
    lagrangian += (
        contract("ijab,ijab->", I14_l1, l2) / 2
    )  # (u*t2+ u + u*t2*t2)*l2 ### #####The u*l2 survives
    del I14_l1
    I15_l1 = np.zeros((no, no, nv, nv), dtype=t2.dtype)
    I15_l1 += 2 * contract("jaib->ijab", u_transform[o, v, o, v])
    I15_l1 -= contract("acki,kjbc->ijab", t2, u_transform[o, o, v, v])
    I16_l1 = np.zeros((no, no, nv, nv), dtype=t2.dtype)
    del I15_l1
    I16_l1 += 2 * contract("ia,jb->ijab", f_transform[o, v], l1)
    lagrangian -= contract("ijab,abji->", I16_l1, t2) / 2  # f*l1*t2
    del I16_l1
    I17_l1 = np.zeros((no, no, no, no), dtype=t2.dtype)
    I17_l1 += contract("abij,lkba->ijkl", t2, u_transform[o, o, v, v])
    I18_l1 = np.zeros((no, no, no, no), dtype=t2.dtype)
    I18_l1 += contract("lkji->ijkl", I17_l1)
    del I17_l1
    I18_l1 += contract("jilk->ijkl", u_transform[o, o, o, o])
    I19_l1 = np.zeros((no, no, no, no), dtype=t2.dtype)
    del I18_l1
    del I19_l1
    I37_l1 = np.zeros((no, no, nv, nv), dtype=t2.dtype)
    I37_l1 += 2 * contract("jiab->ijab", u_transform[o, o, v, v])
    I37_l1 -= contract("jiba->ijab", u_transform[o, o, v, v])
    I38_l1 = np.zeros((nv, nv), dtype=t2.dtype)
    I38_l1 += contract("ijcb,acij->ab", I37_l1, t2)
    del I3_l1
    del I38_l1
    I41_l1 = np.zeros((no, no), dtype=t2.dtype)
    I41_l1 += contract("kiab,bakj->ij", I37_l1, t2)
    del I37_l1
    I39_l1 += contract("ai->ia", f_transform[v, o])
    I39_l1 += 2 * contract("jb,abij->ia", f_transform[o, v], t2)
    lagrangian += contract(
        "ia,ia->", I39_l1, l1
    )  # l1*u(o,o,o,v)*t2 + l1*u(o,v,v,v)*t2 +l1*f + l1*f*t2)
    del I39_l1
    del I41_l1
    del I7_l1

    return lagrangian
