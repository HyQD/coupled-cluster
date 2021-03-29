
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

    I14_l1 -= np.einsum("iakj->ijka", u_t[o, v, o, o])
    rhs += np.einsum("ijkb,jkba->ia", I14_l1, l2)  # l2*ut(o,v,o,o)

    I18_l1 += np.einsum("abic->iabc", u_t[v, v, o, v])
    rhs += np.einsum("jbca,jibc->ia", I18_l1, l2)  # l2*ut(v,v,o,v)

    I19_l1 += np.einsum("ib,bakj->ijka", l1, t2)
    I22_l1 -= 2 * np.einsum("ijka->ijka", I19_l1)
    I22_l1 += np.einsum("ikja->ijka", I19_l1)
    rhs += np.einsum(
        "ijkb,jkba->ia", I22_l1, u_t[o, o, v, v]
    )  # u(o,o,v,v)(l1*t2)  = s10c

    I16_l1 += 2 * np.einsum("abji->ijab", t2)
    I16_l1 -= np.einsum("baji->ijab", t2)
    I31_l1 -= np.einsum("jb,jiab->ia", l1, I16_l1)
    I7_l1 += 2 * np.einsum("jiab->ijab", u_t[o, o, v, v])
    I7_l1 -= np.einsum("jiba->ijab", u_t[o, o, v, v])
    rhs -= np.einsum("jb,jiab->ia", I31_l1, I7_l1)  # (l1*t2)*(u_t[o, o, v, v])  = s7

    I32_l1 += 2 * np.einsum("iabj->ijab", u_t[o, v, v, o])
    I32_l1 -= np.einsum("iajb->ijab", u_t[o, v, o, v])
    rhs += np.einsum("jb,ijba->ia", l1, I32_l1)  # l1*u_t[o, v, o, v]

    I34_l1 += np.einsum("ab->ab", f_t[v, v])
    rhs += np.einsum("ba,ib->ia", I34_l1, l1)  # l1*f[v, v]

    I20_l1 -= np.einsum("abji->ijab", t2)
    I20_l1 += 2 * np.einsum("baji->ijab", t2)

    I35_l1 += np.einsum("ij->ij", f_t[o, o])
    rhs -= np.einsum("ja,ij->ia", l1, I35_l1)  # Made this myself

    I36_l1 += np.einsum(
        "kjab,kiab->ij", I20_l1, u_t[o, o, v, v]
    )  # = s10d  Handmade, not generated
    rhs -= np.einsum("ij,ja->ia", I36_l1, l1)  # l1*u_t*t2) # Handmade, not generatet

    rhs += 2 * np.einsum("ia->ia", f_t[o, v])  # f_t[o, v]

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

    I0_l2 += np.einsum("ca,jicb->ijab", f[v, v], l2)
    I15_l2 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I15_l2 -= np.einsum("ijab->ijab", I0_l2)
    del I0_l2
    rhs = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    I13_l2 += np.einsum("ij->ij", f[o, o])
    I14_l2 += np.einsum("ik,kjab->ijab", I13_l2, l2)
    del I13_l2
    I15_l2 += np.einsum("ijba->ijab", I14_l2)
    del I14_l2
    rhs -= np.einsum("ijba->ijab", I15_l2)
    rhs -= np.einsum("jiab->ijab", I15_l2)
    del I15_l2
    I16_l2 += np.einsum("ic,jcab->ijab", l1, u_t[o, v, v, v])
    I24_l2 -= np.einsum("ijab->ijab", I16_l2)
    del I16_l2
    rhs += np.einsum("ijab->ijab", I24_l2)
    rhs -= 2 * np.einsum("ijba->ijab", I24_l2)
    rhs -= 2 * np.einsum("jiab->ijab", I24_l2)
    rhs += np.einsum("jiba->ijab", I24_l2)
    del I24_l2
    I25_l2 += np.einsum("ka,ijkb->ijab", l1, u_t[o, o, o, v])
    I35_l2 += np.einsum("ijab->ijab", I25_l2)
    del I25_l2
    I35_l2 -= np.einsum("ia,jb->ijab", f_t[o, v], l1)
    rhs -= 2 * np.einsum("ijab->ijab", I35_l2)
    rhs += np.einsum("ijba->ijab", I35_l2)
    rhs += np.einsum("jiab->ijab", I35_l2)
    rhs -= 2 * np.einsum("jiba->ijab", I35_l2)
    del I35_l2

    rhs -= 2 * np.einsum("jiab->ijab", u_t[o, o, v, v])
    rhs += 4 * np.einsum("jiba->ijab", u_t[o, o, v, v])

    return rhs
