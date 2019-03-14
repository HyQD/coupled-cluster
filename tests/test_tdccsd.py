import numpy as np
from coupled_cluster.ccsd.energies import lagrangian_functional


def test_lagrangian_functional(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    f = cs.f
    u = cs.u
    o = cs.o
    v = cs.v

    result = np.einsum("ai, ia->", f[v, o], l_1, optimize=True)
    result += np.einsum("ia, ai->", f[o, v], t_1, optimize=True) 
    
    result += np.einsum("ab, ia, bi->", f[v, v], l_1, t_1) 
    result += np.einsum("ia, jb, abij->", f[o, v], l_1, t_2, optimize=True)

    result += (0.5) * np.einsum("ia, abjk, jkbi->", l_1, t_2, u[o, o, v, o], optimize=True)
    result += (0.5) * np.einsum("ia, bcij, ajbc->", l_1, t_2, u[v, o, v, v], optimize=True)
    result += (0.5) * np.einsum("ijab, ak, bkij->", l_2, t_1, u[v, o, o, o], optimize=True)
    result += (0.5) * np.einsum("ijab, ci, abcj->", l_2, t_1, u[v, v, v, o], optimize=True)

    result += (-1) * np.einsum("ji, ia, aj->", f[o, o], l_1, t_1, optimize=True)
    result += (-1) * np.einsum("ia, bj, ajbi->", l_1, t_1, u[v, o, v, o], optimize=True)

    result += (-0.5) * np.einsum("aj, bi, ijab->", t_1, t_1, u[o, o, v, v], optimize=True)

    result += np.einsum("ia, aj, bk, jkbi->", l_1, t_1, t_1, u[o, o, v, o], optimize=True)
    result += np.einsum("ia, bi, cj, ajbc->", l_1, t_1, t_1, u[v, o, v, v], optimize=True)
    result += np.einsum("ia, bj, acik, jkbc->", l_1, t_1, t_2, u[o, o, v, v], optimize=True)
    result += np.einsum("ijab, ak, ci, bkcj->", l_2, t_1, t_1, u[v, o, v, o], optimize=True)

    result += (-1) * np.einsum("ia, jb, aj, bi->", f[o, v], l_1, t_1, t_1, optimize=True)
    result += (-1) * np.einsum("ijab, ak, bcil, klcj->", l_2, t_1, t_2, u[o, o, v, o], optimize=True)
    result += (-1) * np.einsum("ijab, ci, adjk, bkcd->", l_2, t_1, t_2, u[v, o, v, v], optimize=True) 

    result += (-0.5) * np.einsum("ia, jkbc, aj, bcik->", f[o, v], l_2, t_1, t_2, optimize=True)
    result += (-0.5) * np.einsum("ia, jkbc, bi, acjk->", f[o, v], l_2, t_1, t_2, optimize=True)
    result += (-0.5) * np.einsum("ia, aj, bcik, jkbc->", l_1, t_1, t_2, u[o, o, v, v], optimize=True)
    result += (-0.5) * np.einsum("ia, bi, acjk, jkbc->", l_1, t_1, t_2, u[o, o, v, v], optimize=True)
    result += (-0.5) * np.einsum("ijab, ck, abil, klcj->", l_2, t_1, t_2, u[o, o, v, o], optimize=True)
    result += (-0.5) * np.einsum("ijab, ck, adij, bkcd->", l_2, t_1, t_2, u[v, o, v, v], optimize=True)

    result += (-0.25) * np.einsum("ijab, al, bk, klij->", l_2, t_1, t_1, u[o, o, o, o], optimize=True)
    result += (-0.25) * np.einsum("ijab, cj, di, abcd->", l_2, t_1, t_1, u[v, v, v, v], optimize=True)
    result += (0.25) * np.einsum("ijab, ak, cdij, bkcd->", l_2, t_1, t_2, u[v, o, v, v], optimize=True)
    result += (0.25) * np.einsum("ijab, ci, abkl, klcj->", l_2, t_1, t_2, u[o, o, v, o], optimize=True)

    result += (-1) * np.einsum("ia, ak, bj, ci, jkbc->", l_1, t_1, t_1, t_1, u[o, o, v, v], optimize=True)
    result += (-1) * np.einsum("ijab, ak, ci, bdjl, klcd->", l_2, t_1, t_1, t_2, u[o, o, v, v], optimize=True)

    result += (-0.5) * np.einsum("ijab, ak, cj, di, bkcd->", l_2, t_1, t_1, t_1, u[v, o, v, v], optimize=True)
    result += (-0.5) * np.einsum("ijab, ak, cl, bdij, klcd->", l_2, t_1, t_1, t_2, u[o, o, v, v], optimize=True)
    result += (-0.5) * np.einsum("ijab, al, bk, ci, klcj->", l_2, t_1, t_1, t_1, u[o, o, v, o], optimize=True)
    result += (-0.5) * np.einsum("ijab, ci, dk, abjl, klcd->", l_2, t_1, t_1, t_2, u[o, o, v, v], optimize=True)

    result += (-0.125) * np.einsum("ijab, al, bk, cdij, klcd->", l_2, t_1, t_1, t_2, u[o, o, v, v], optimize=True)
    result += (-0.125) * np.einsum("ijab, cj, di, abkl, klcd->", l_2, t_1, t_1, t_2, u[o, o, v, v], optimize=True)

    result += (0.25) * np.einsum("ijab, al, bk, cj, di, klcd->", l_2, t_1, t_1, t_1, t_1, u[o, o, v, v], optimize=True)


    energy = lagrangian_functional(f, u, t_1, t_2, l_1, l_2, o, v, np=np, test=True)

    assert abs(result - energy) < 1e-8