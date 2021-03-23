import coupled_cluster.ccs.rhs_l as ccs_l
import coupled_cluster.ccd.rhs_l as ccd_l


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
    path1 = np.einsum_path("iakj->ijka", u_t[o, v, o, o])
    print(path1)
    rhs += np.einsum("ijkb,jkba->ia", I14_l1, l2)  # l2*ut(o,v,o,o)
    path2 = np.einsum_path("ijkb,jkba->ia", I14_l1, l2)
    print(path2)
    I18_l1 += np.einsum("abic->iabc", u_t[v, v, o, v])
    path3 = np.einsum_path("abic->iabc", u_t[v, v, o, v])
    print(path3)
    rhs += np.einsum("jbca,jibc->ia", I18_l1, l2)  # l2*ut(v,v,o,v)
    path4 = np.einsum_path("jbca,jibc->ia", I18_l1, l2)
    print(path4)

    I19_l1 += np.einsum("ib,bakj->ijka", l1, t2)
    path5 = np.einsum_path("ib,bakj->ijka", l1, t2)
    print(path5)
    I22_l1 -= 2 * np.einsum("ijka->ijka", I19_l1)
    path6 = 2 * np.einsum_path("ijka->ijka", I19_l1)
    print(path6)
    I22_l1 += np.einsum("ikja->ijka", I19_l1)
    path7 = np.einsum_path("ikja->ijka", I19_l1)
    print(path7)
    rhs += np.einsum(
        "ijkb,jkba->ia", I22_l1, u_t[o, o, v, v]
    )  # u(o,o,v,v)(l1*t2)  = s10c
    path8 = np.einsum_path("ijkb,jkba->ia", I22_l1, u_t[o, o, v, v])
    print(path8)

    I16_l1 += 2 * np.einsum("abji->ijab", t2)
    path9 = 2 * np.einsum_path("abji->ijab", t2)
    print(path9)
    I16_l1 -= np.einsum("baji->ijab", t2)
    path10 = np.einsum_path("baji->ijab", t2)
    print(path10)
    I31_l1 -= np.einsum("jb,jiab->ia", l1, I16_l1)
    path11 = np.einsum_path("jb,jiab->ia", l1, I16_l1)
    print(path11)
    I7_l1 += 2 * np.einsum("jiab->ijab", u_t[o, o, v, v])
    path12 = 2 * np.einsum_path("jiab->ijab", u_t[o, o, v, v])
    print(path12)
    I7_l1 -= np.einsum("jiba->ijab", u_t[o, o, v, v])
    path13 = np.einsum_path("jiba->ijab", u_t[o, o, v, v])
    print(path13)
    rhs -= np.einsum("jb,jiab->ia", I31_l1, I7_l1)  # (l1*t2)*(u_t[o, o, v, v])  = s7
    path14 = np.einsum_path("jb,jiab->ia", I31_l1, I7_l1)
    print(path14)

    I32_l1 += 2 * np.einsum("iabj->ijab", u_t[o, v, v, o])
    path15 = 2 * np.einsum_path("iabj->ijab", u_t[o, v, v, o])
    print(path15)
    I32_l1 -= np.einsum("iajb->ijab", u_t[o, v, o, v])
    path16 = np.einsum_path("iajb->ijab", u_t[o, v, o, v])
    print(path16)
    rhs += np.einsum("jb,ijba->ia", l1, I32_l1)  # l1*u_t[o, v, o, v]
    path17 = np.einsum_path("jb,ijba->ia", l1, I32_l1)
    print(path17)

    I34_l1 += np.einsum("ab->ab", f_t[v, v])
    path18 = np.einsum_path("ab->ab", f_t[v, v])
    print(path18)
    rhs += np.einsum("ba,ib->ia", I34_l1, l1)  # l1*f[v, v]
    path19 = np.einsum_path("ba,ib->ia", I34_l1, l1)
    print(path19)

    I20_l1 -= np.einsum("abji->ijab", t2)
    path19 = np.einsum_path("abji->ijab", t2)
    print(path19)
    I20_l1 += 2 * np.einsum("baji->ijab", t2)
    path20 = 2 * np.einsum_path("baji->ijab", t2)
    print(path20)

    I35_l1 += np.einsum("ij->ij", f_t[o, o])
    path21 = np.einsum_path("ij->ij", f_t[o, o])
    print(path21)
    rhs -= np.einsum("ja,ij->ia", l1, I35_l1)
    path21 = np.einsum_path("ja,ij->ia", l1, I35_l1)
    print(path21)

    I36_l1 += np.einsum("kjab,kiab->ij", I20_l1, u_t[o, o, v, v])
    path22 = np.einsum_path("kjab,kiab->ij", I20_l1, u_t[o, o, v, v])  # = s10d
    print(path22)
    rhs -= np.einsum("ij,ja->ia", I36_l1, l1)  # l1*u_t*t2)
    path23 = np.einsum_path("ij,ja->ia", I36_l1, l1)
    print(path23)

    rhs += 2 * np.einsum("ia->ia", f_t[o, v])  # f_t[o, v]
    path24 = 2 * np.einsum_path("ia->ia", f_t[o, v])
    print(path24)

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
    I15_l2 = np.zeros((no, no, nv, nv), dtype=t1.dtype)
    rhs = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    I0_l2 += np.einsum("ca,jicb->ijab", f[v, v], l2)
    path1 = np.einsum_path("ca,jicb->ijab", f[v, v], l2)

    I15_l2 -= np.einsum("ijab->ijab", I0_l2)  # 2
    path2 = np.einsum_path("ijab->ijab", I0_l2)

    I13_l2 += np.einsum("ij->ij", f[o, o])
    path3 = np.einsum_path("ij->ij", f[o, o])
    I14_l2 += np.einsum("ik,kjab->ijab", I13_l2, l2)
    path4 = np.einsum_path("ik,kjab->ijab", I13_l2, l2)

    I15_l2 += np.einsum("ijba->ijab", I14_l2)  # 3
    path5 = np.einsum_path("ijba->ijab", I14_l2)

    rhs -= np.einsum("ijba->ijab", I15_l2)
    rhs -= np.einsum("jiab->ijab", I15_l2)
    path6 = np.einsum_path("jiab->ijab", I15_l2)

    I16_l2 += np.einsum("ic,jcab->ijab", l1, u_t[o, v, v, v])
    path7 = np.einsum("ic,jcab->ijab", l1, u_t[o, v, v, v])
    I24_l2 -= np.einsum("ijab->ijab", I16_l2)
    path8 = np.einsum_path("ijab->ijab", I16_l2)

    rhs += np.einsum("ijab->ijab", I24_l2)
    rhs -= 2 * np.einsum("ijba->ijab", I24_l2)
    rhs -= 2 * np.einsum("jiab->ijab", I24_l2)
    rhs += np.einsum("jiba->ijab", I24_l2)
    path9 = np.einsum_path("jiba->ijab", I24_l2)

    I25_l2 += np.einsum("ka,ijkb->ijab", l1, u_t[o, o, o, v])
    path10 = np.einsum_path("ka,ijkb->ijab", l1, u_t[o, o, o, v])
    I35_l2 += np.einsum("ijab->ijab", I25_l2)  # 4
    path11 = np.einsum_path("ijab->ijab", I25_l2)

    I35_l2 -= np.einsum("ia,jb->ijab", f_t[o, v], l1)  # 6
    path12 = np.einsum_path("ia,jb->ijab", f_t[o, v], l1)
    rhs -= 2 * np.einsum("ijab->ijab", I35_l2)
    rhs += np.einsum("ijba->ijab", I35_l2)
    rhs += np.einsum("jiab->ijab", I35_l2)
    rhs -= 2 * np.einsum("jiba->ijab", I35_l2)
    path13 = 2 * np.einsum_path("jiba->ijab", I35_l2)

    rhs -= 2 * np.einsum("jiab->ijab", u_t[o, o, v, v])  # 1
    rhs += 4 * np.einsum("jiba->ijab", u_t[o, o, v, v])
    path14 = 4 * np.einsum_path("jiba->ijab", u_t[o, o, v, v])

    del I0_l2
    del I35_l2
    del I25_l2
    del I24_l2
    del I16_l2
    del I15_l2
    del I14_l2
    del I13_l2

    print(path1)
    print(path2)
    print(path3)
    print(path4)
    print(path5)
    print(path6)
    print(path7)
    print(path8)
    print(path9)
    print(path10)
    print(path11)
    print(path12)
    print(path13)
    print(path14)

    return rhs
