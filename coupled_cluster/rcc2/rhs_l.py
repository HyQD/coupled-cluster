from opt_einsum import contract


def compute_l_1_amplitudes(
    f, f_t, u_t, t1, t2, l1, l2, o, v, np, intermediates=None, out=None
):
    """
    if out is None:
    out = np.zeros_like(l_1)
    """
    no = t1.shape[1]
    nv = t1.shape[0]

    rhs = np.zeros((no, nv), dtype=t1.dtype)

    I7_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I14_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)
    I16_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I17_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I18_l1 = np.zeros((no, nv, nv, nv), dtype=t1.dtype)
    I19_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)
    I20_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I22_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)
    I25_l1 = np.zeros((no, no, no, nv), dtype=t1.dtype)
    I31_l1 = np.zeros((no, nv), dtype=t1.dtype)
    I32_l1 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I34_l1 = np.zeros((nv, nv), dtype=t1.dtype)
    I35_l1 = np.zeros((no, no), dtype=t1.dtype)
    I36_l1 = np.zeros((no, no), dtype=t1.dtype)

    I14_l1 -= contract("iakj->ijka", u_t[o, v, o, o])
    rhs += contract("ijkb,jkba->ia", I14_l1, l2)  # l2*ut(o,v,o,o)

    I18_l1 += contract("abic->iabc", u_t[v, v, o, v])
    rhs += contract("jbca,jibc->ia", I18_l1, l2)  # l2*ut(v,v,o,v)

    I19_l1 += contract("ib,bakj->ijka", l1, t2)
    I22_l1 -= 2 * contract("ijka->ijka", I19_l1)
    I22_l1 += contract("ikja->ijka", I19_l1)
    rhs += contract(
        "ijkb,jkba->ia", I22_l1, u_t[o, o, v, v]
    )  # u(o,o,v,v)(l1*t2)  = s10c

    I16_l1 += 2 * contract("abji->ijab", t2)
    I16_l1 -= contract("baji->ijab", t2)
    I31_l1 -= contract("jb,jiab->ia", l1, I16_l1)
    I7_l1 += 2 * contract("jiab->ijab", u_t[o, o, v, v])
    I7_l1 -= contract("jiba->ijab", u_t[o, o, v, v])
    rhs -= contract(
        "jb,jiab->ia", I31_l1, I7_l1
    )  # (l1*t2)*(u_t[o, o, v, v])  = s7

    I32_l1 += 2 * contract("iabj->ijab", u_t[o, v, v, o])
    I32_l1 -= contract("iajb->ijab", u_t[o, v, o, v])
    rhs += contract("jb,ijba->ia", l1, I32_l1)  # l1*u_t[o, v, o, v]

    I34_l1 += contract("ab->ab", f_t[v, v])
    rhs += contract("ba,ib->ia", I34_l1, l1)  # l1*f[v, v]

    I20_l1 -= contract("abji->ijab", t2)
    I20_l1 += 2 * contract("baji->ijab", t2)

    I35_l1 += contract("ij->ij", f_t[o, o])
    rhs -= contract("ja,ij->ia", l1, I35_l1)  # Made this myself

    I36_l1 += contract(
        "kjab,kiab->ij", I20_l1, u_t[o, o, v, v]
    )  # = s10d  Handmade, not generated
    rhs -= contract(
        "ij,ja->ia", I36_l1, l1
    )  # l1*u_t*t2) # Handmade, not generatet

    rhs += 2 * contract("ia->ia", f_t[o, v])  # f_t[o, v]

    del I7_l1
    del I14_l1
    del I17_l1
    del I16_l1
    del I18_l1
    del I19_l1
    del I22_l1
    del I20_l1
    del I31_l1
    del I32_l1
    del I34_l1
    del I35_l1
    del I36_l1

    return rhs


def compute_l_2_amplitudes(
    f, f_t, u_t, t1, t2, l1, l2, o, v, np, intermediates=None, out=None
):
    """
    if out is None:
        out = np.zeros_like(l_2)
    """
    no = t1.shape[1]
    nv = t1.shape[0]

    I0_l2 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I1_l2 = np.zeros((no, no, no, nv), dtype=t1.dtype)
    I52_l2 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I54_l2 = np.zeros((no, no, no, no), dtype=t1.dtype)
    I24_l2 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I53_l2 = np.zeros((no, no, no, no), dtype=t1.dtype)
    I35_l2 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I13_l2 = np.zeros((no, no), dtype=t1.dtype)
    I14_l2 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I16_l2 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I25_l2 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I42_l2 = np.zeros((nv, nv), dtype=t1.dtype)
    I34_l2 = np.zeros((no, nv), dtype=t1.dtype)

    I0_l2 += contract("ca,jicb->ijab", f[v, v], l2)
    I15_l2 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I15_l2 -= contract("ijab->ijab", I0_l2)
    del I0_l2
    rhs = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I13_l2 += contract("ij->ij", f[o, o])
    I14_l2 += contract("ik,kjab->ijab", I13_l2, l2)
    del I13_l2
    I15_l2 += contract("ijba->ijab", I14_l2)
    del I14_l2
    rhs -= contract("ijba->ijab", I15_l2)
    rhs -= contract("jiab->ijab", I15_l2)
    del I15_l2
    I16_l2 += contract("ic,jcab->ijab", l1, u_t[o, v, v, v])
    I24_l2 -= contract("ijab->ijab", I16_l2)
    del I16_l2
    rhs += contract("ijab->ijab", I24_l2)
    rhs -= 2 * contract("ijba->ijab", I24_l2)
    rhs -= 2 * contract("jiab->ijab", I24_l2)
    rhs += contract("jiba->ijab", I24_l2)
    del I24_l2
    I25_l2 += contract("ka,ijkb->ijab", l1, u_t[o, o, o, v])
    I35_l2 += contract("ijab->ijab", I25_l2)
    del I25_l2
    I35_l2 -= contract("ia,jb->ijab", f_t[o, v], l1)
    rhs -= 2 * contract("ijab->ijab", I35_l2)
    rhs += contract("ijba->ijab", I35_l2)
    rhs += contract("jiab->ijab", I35_l2)
    rhs -= 2 * contract("jiba->ijab", I35_l2)
    del I35_l2

    rhs -= 2 * contract("jiab->ijab", u_t[o, o, v, v])
    rhs += 4 * contract("jiba->ijab", u_t[o, o, v, v])

    return rhs
