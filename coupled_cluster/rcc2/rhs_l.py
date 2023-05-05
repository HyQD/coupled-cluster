from opt_einsum import contract


def compute_l_1_amplitudes(
    f, f_t, V_t, u_t, t1, t2, l1, l2, o, v, np, intermediates=None, out=None
):
    no = t1.shape[1]
    nv = t1.shape[0]

    r_L1 = np.zeros((no, nv), dtype=t1.dtype)

    r_L1 += 2 * f_t[o, v]

    r_L1 += contract("ba, ib->ia", f_t[v, v], l1)
    r_L1 -= contract("ij, ja->ia", f_t[o, o], l1)
    r_L1 += 2 * contract("jb, ibaj->ia", l1, u_t[o, v, v, o])
    r_L1 -= contract("jb, ibja->ia", l1, u_t[o, v, o, v])

    Loovv = 2 * u_t[o, o, v, v] - u_t[o, o, v, v].swapaxes(2, 3)
    tmp_ck = 2 * contract("jb, bcjk->ck", l1, t2)
    r_L1 += contract("ck, ikac->ia", tmp_ck, Loovv)

    tmp_ck = contract("jb, bckj->ck", l1, t2)
    r_L1 -= contract("ck, ikac->ia", tmp_ck, Loovv)

    tt = 2 * t2 - t2.swapaxes(2, 3)
    tmp_ij = contract("ikbc, bcjk->ij", u_t[o, o, v, v], tt)
    r_L1 -= contract("ja, ij->ia", l1, tmp_ij)

    tmp_ba = contract("jkac, bcjk->ba", u_t[o, o, v, v], tt)
    r_L1 -= contract("ib, ba->ia", l1, tmp_ba)

    r_L1 += contract("ijbc, bcaj->ia", l2, u_t[v, v, v, o])
    r_L1 -= contract("jkab, ibjk->ia", l2, u_t[o, v, o, o])

    tmp_ba = contract("jkac, bcjk->ba", l2, t2)
    r_L1 -= contract("ib, ba->ia", V_t[o, v], tmp_ba)

    tmp_ji = contract("ikbc, bcjk->ji", l2, t2)
    r_L1 -= contract("ja, ji->ia", V_t[o, v], tmp_ji)

    return r_L1


def compute_l_2_amplitudes(
    f, f_t, u_t, t1, t2, l1, l2, o, v, np, intermediates=None, out=None
):
    no = t1.shape[1]
    nv = t1.shape[0]

    r_L2 = np.zeros((no, no, nv, nv), dtype=t1.dtype)

    Loovv = 2 * u_t[o, o, v, v] - u_t[o, o, v, v].swapaxes(2, 3)
    r_L2 += 2 * Loovv

    r_L2 += contract("ca, ijcb->ijab", f[v, v], l2)
    r_L2 += contract("cb, ijac->ijab", f[v, v], l2)

    r_L2 -= contract("jk, ikab->ijab", f[o, o], l2)
    r_L2 -= contract("ik, kjab->ijab", f[o, o], l2)

    r_L2 += 2 * contract("ia, jb->ijab", f_t[o, v], l1)
    r_L2 += 2 * contract("jb, ia->ijab", f_t[o, v], l1)
    r_L2 -= contract("ib, ja->ijab", f_t[o, v], l1)
    r_L2 -= contract("ja, ib->ijab", f_t[o, v], l1)

    r_L2 -= 2 * contract("ijkb, ka->ijab", u_t[o, o, o, v], l1)
    r_L2 += contract("ijbk, ka->ijab", u_t[o, o, v, o], l1)

    r_L2 -= 2 * contract("ijak, kb->ijab", u_t[o, o, v, o], l1)
    r_L2 += contract("ijka, kb->ijab", u_t[o, o, o, v], l1)

    r_L2 += 2 * contract("ic, cjab->ijab", l1, u_t[v, o, v, v])
    r_L2 -= contract("ic,cjba->ijab", l1, u_t[v, o, v, v])

    r_L2 += 2 * contract("jc, icab->ijab", l1, u_t[o, v, v, v])
    r_L2 -= contract("jc, icba->ijab", l1, u_t[o, v, v, v])

    return r_L2
