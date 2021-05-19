from opt_einsum import contract


def compute_t_2_amplitudes(f, u, t, o, v, np, out=None):

    rhs_T2 = u[v, v, o, o].copy()

    Pabij = contract("ac, bcji->abij", f[v, v], t)
    Pabij -= contract("ki, abkj->abij", f[o, o], t)
    rhs_T2 += Pabij
    rhs_T2 += Pabij.swapaxes(0, 1).swapaxes(2, 3)

    return rhs_T2


def compute_l_2_amplitudes(f, u, t2, l2, o, v, np, out=None):
    no = t2.shape[2]
    nv = t2.shape[0]

    L_ijab = 2 * u[o, o, v, v] - u[o, o, v, v].transpose(0, 1, 3, 2)
    rhs_L2 = 2 * L_ijab

    Pabij = contract("ca, ijcb->ijab", f[v, v], l2)
    Pabij -= contract("ik, kjab->ijab", f[o, o], l2)
    rhs_L2 += Pabij
    rhs_L2 += Pabij.swapaxes(0, 1).swapaxes(2, 3)

    return rhs_L2
