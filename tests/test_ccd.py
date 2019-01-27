import pytest
import numpy as np
from coupled_cluster.ccd import CoupledClusterDoubles
from coupled_cluster.ccd.rhs_t import (
    add_d1_t,
    add_d2a_t,
    add_d2b_t,
    add_d2c_t,
    add_d2d_t,
    add_d2e_t,
    add_d3a_t,
    add_d3b_t,
    add_d3c_t,
    add_d3d_t,
    compute_t_2_amplitudes,
)
from coupled_cluster.ccd.rhs_l import (
    add_d1_l,
    add_d2a_l,
    add_d2b_l,
    add_d2c_l,
    add_d2d_l,
    add_d2e_l,
    add_d3a_l,
    add_d3b_l,
    add_d3c_l,
    add_d3d_l,
    add_d3e_l,
    add_d3f_l,
    add_d3g_l,
    compute_l_2_amplitudes,
)
from coupled_cluster.ccd.time_dependent_overlap import (
    compute_time_dependent_overlap,
)
from coupled_cluster.ccd.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)


@pytest.fixture
def iterated_ccd_amplitudes(helium_system, beryllium_system, neon_system):

    ccd_list = []
    for system in [helium_system, beryllium_system, neon_system]:
        ccd = CoupledClusterDoubles(system, verbose=False)
        ccd.iterate_t_amplitudes(theta=0.9)
        ccd.iterate_l_amplitudes(theta=0.9)

        ccd_list.append(ccd)

    return ccd_list


def test_add_d1_t(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(t)
    add_d1_t(u, o, v, out, np=np)

    np.testing.assert_allclose(out, u[v, v, o, o], atol=1e-10)


def test_add_d2a_t(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    f = large_system_ccd.f
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(t)
    add_d2a_t(f, t, o, v, out, np=np)
    out_e = np.einsum("bc, acij -> abij", f[v, v], t)
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2b_t(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    f = large_system_ccd.f
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(t)
    add_d2b_t(f, t, o, v, out, np=np)
    out_e = -np.einsum("kj, abik -> abij", f[o, o], t)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2c_t(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(t)
    add_d2c_t(u, t, o, v, out, np=np)
    out_e = 0.5 * np.einsum("cdij, abcd -> abij", t, u[v, v, v, v])

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2d_t(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(t)
    add_d2d_t(u, t, o, v, out, np=np)
    out_e = 0.5 * np.einsum("abkl, klij -> abij", t, u[o, o, o, o])

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2e_t(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(t)
    add_d2e_t(u, t, o, v, out, np=np)
    out_e = np.einsum("acik, kbcj -> abij", t, u[o, v, v, o])
    out_e -= out_e.swapaxes(0, 1)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3a_t(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(t)
    add_d3a_t(u, t, o, v, out, np=np)
    out_e = 0.25 * np.einsum(
        "cdij, abkl, klcd -> abij", t, t, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3b_t(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(t)
    add_d3b_t(u, t, o, v, out, np=np)
    out_e = np.einsum(
        "acik, bdjl, klcd -> abij", t, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3c_t(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(t)
    add_d3c_t(u, t, o, v, out, np=np)
    out_e = -0.5 * np.einsum(
        "ablj, dcik, klcd -> abij", t, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3d_t(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(t)
    add_d3d_t(u, t, o, v, out, np=np)
    out_e = -0.5 * np.einsum(
        "aclk, dbij, klcd -> abij", t, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_full_t_amplitudes(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    T2 = t.copy().transpose(2, 3, 0, 1)
    F = large_system_ccd.f
    W = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    np.testing.assert_allclose(t, -t.transpose(1, 0, 2, 3), atol=1e-10)
    np.testing.assert_allclose(t, -t.transpose(0, 1, 3, 2), atol=1e-10)
    np.testing.assert_allclose(t, t.transpose(1, 0, 3, 2), atol=1e-10)

    # d1
    result = -1.0 * np.einsum(
        "BAIJ->IJAB", W[v, v, o, o], optimize=["einsum_path", (0,)]
    )

    # d2a
    temp = np.einsum(
        "IJAc,Bc->IJAB", T2, F[v, v], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)

    # d2b
    temp = np.einsum(
        "IkBA,kJ->IJAB", T2, F[o, o], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)

    # d2c
    result += -0.5 * np.einsum(
        "IJdc,BAdc->IJAB", T2, W[v, v, v, v], optimize=["einsum_path", (0, 1)]
    )

    # d2d
    result += -0.5 * np.einsum(
        "lkBA,lkIJ->IJAB", T2, W[o, o, o, o], optimize=["einsum_path", (0, 1)]
    )

    # d2e
    temp = np.einsum(
        "IkAc,BkJc->IJAB", T2, W[v, o, o, v], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)

    # d3a
    result += -0.25 * np.einsum(
        "lkBA,IJdc,lkdc->IJAB",
        T2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )

    # d3b
    temp = np.einsum(
        "IkAc,JlBd,lkdc->IJAB",
        T2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)

    # d3c
    temp = 0.5 * np.einsum(
        "IlBA,Jkdc,lkdc->IJAB",
        T2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)

    # d3d
    temp = 0.5 * np.einsum(
        "IJAc,lkBd,lkdc->IJAB",
        T2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)

    out = compute_t_2_amplitudes(F, W, t, o, v, np=np)
    np.testing.assert_allclose(out, result.transpose(2, 3, 0, 1), atol=1e-10)


def test_add_d1_l(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(l)
    add_d1_l(u, o, v, out, np=np)
    out_e = u[o, o, v, v].copy()

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2a_l(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(l)
    add_d2a_l(u, l, o, v, out, np=np)
    out_e = 0.5 * np.einsum("klab, ijkl -> ijab", l, u[o, o, o, o])

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2b_l(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(l)
    add_d2b_l(u, l, o, v, out, np=np)
    out_e = 0.5 * np.einsum("ijdc, dcab -> ijab", l, u[v, v, v, v])

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2c_l(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    f = large_system_ccd.f
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(l)
    add_d2c_l(f, l, o, v, out, np=np)
    out_e = -np.einsum("ijbc, ca -> ijab", l, f[v, v])
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2d_l(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    f = large_system_ccd.f
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(l)
    add_d2d_l(f, l, o, v, out, np=np)
    out_e = np.einsum("jkab, ik -> ijab", l, f[o, o])
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2e_l(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(l)
    add_d2e_l(u, l, o, v, out, np=np)
    out_e = np.einsum("jkbc, icak -> ijab", l, u[o, v, v, o])
    out_e -= out_e.swapaxes(0, 1)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3a_l(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(l)
    add_d3a_l(u, t, l, o, v, out, np=np)
    out_e = -0.5 * np.einsum(
        "ijbc, dckl, klad -> ijab", l, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3b_l(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(l)
    add_d3b_l(u, t, l, o, v, out, np=np)
    out_e = 0.25 * np.einsum(
        "ijdc, dckl, klab -> ijab", l, t, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3c_l(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(l)
    add_d3c_l(u, t, l, o, v, out, np=np)
    out_e = 0.5 * np.einsum(
        "jkab, dckl, ildc -> ijab", l, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3d_l(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(l)
    add_d3d_l(u, t, l, o, v, out, np=np)
    out_e = -np.einsum(
        "jkbc, dckl, ilad -> ijab", l, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3e_l(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(l)
    add_d3e_l(u, t, l, o, v, out, np=np)
    out_e = 0.5 * np.einsum(
        "jkdc, dckl, ilab -> ijab", l, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3f_l(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(l)
    add_d3f_l(u, t, l, o, v, out, np=np)
    out_e = 0.25 * np.einsum(
        "klab, dckl, ijdc -> ijab", l, t, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3g_l(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    u = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    out = np.zeros_like(l)
    add_d3g_l(u, t, l, o, v, out, np=np)
    out_e = -0.5 * np.einsum(
        "klbc, dckl, ijad -> ijab", l, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_full_l_amplitudes(large_system_ccd):
    t, l, large_system_ccd = large_system_ccd
    T2 = t.copy().transpose(2, 3, 0, 1)
    L2 = l.copy()
    F = large_system_ccd.f
    W = large_system_ccd.u
    o = large_system_ccd.o
    v = large_system_ccd.v

    # d1
    result = -1.0 * np.einsum(
        "IJBA->IJAB", W[o, o, v, v], optimize=["einsum_path", (0,)]
    )

    # d2a
    result += -0.5 * np.einsum(
        "lkBA,IJlk->IJAB", L2, W[o, o, o, o], optimize=["einsum_path", (0, 1)]
    )

    # d2b
    result += -0.5 * np.einsum(
        "IJdc,dcBA->IJAB", L2, W[v, v, v, v], optimize=["einsum_path", (0, 1)]
    )

    # d2c
    temp = np.einsum(
        "IJAc,cB->IJAB", L2, F[v, v], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)

    # d2d
    temp = np.einsum(
        "IkBA,Jk->IJAB", L2, F[o, o], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)

    # d2e
    temp = np.einsum(
        "IkAc,JcBk->IJAB", L2, W[o, v, v, o], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)

    # d3a
    temp = 0.5 * np.einsum(
        "IJAc,lkdc,lkBd->IJAB",
        L2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)

    # d3b
    result += -0.25 * np.einsum(
        "IJdc,lkdc,lkBA->IJAB",
        L2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )

    # d3c
    temp = -0.5 * np.einsum(
        "IkBA,lkdc,Jldc->IJAB",
        L2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)

    # d3d
    temp = np.einsum(
        "IkAc,lkdc,JlBd->IJAB",
        L2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)

    # d3e
    temp = -0.5 * np.einsum(
        "Ikdc,lkdc,JlBA->IJAB",
        L2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)

    # d3f
    result += -0.25 * np.einsum(
        "lkBA,lkdc,IJdc->IJAB",
        L2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )

    # d3g
    temp = 0.5 * np.einsum(
        "lkAc,lkdc,IJBd->IJAB",
        L2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)

    out = compute_l_2_amplitudes(F, W, t, l, o, v, np=np)
    np.testing.assert_allclose(out, result, atol=1e-10)


def test_compute_time_dependent_overlap():
    n = 10
    m = 20

    t = np.random.random((m, m, n, n)) + 1j * np.random.random((m, m, n, n))
    t_t = np.random.random((m, m, n, n)) + 1j * np.random.random((m, m, n, n))
    l = np.random.random((n, n, m, m)) + 1j * np.random.random((n, n, m, m))
    l_t = np.random.random((n, n, m, m)) + 1j * np.random.random((n, n, m, m))

    overlap = compute_time_dependent_overlap(t, l, t_t, l_t, np=np)

    tilde_t = 1
    tilde_t += 0.25 * np.einsum("ijab, abij ->", l_t, t)
    tilde_t -= 0.25 * np.einsum("ijab, abij ->", l_t, t_t)

    tilde_0 = 1
    tilde_0 += 0.25 * np.einsum("ijab, abij ->", l, t_t)
    tilde_0 -= 0.25 * np.einsum("ijab, abij ->", l, t)

    overlap_e = tilde_t * tilde_0

    assert abs(overlap_e - overlap) < 1e-8


def test_reference_energy(tdho, ref_energy):
    tol = 1e-4

    cc_scheme = CoupledClusterDoubles(tdho, verbose=True)
    e_ref = cc_scheme.compute_reference_energy()

    assert abs(e_ref - ref_energy) < tol


def test_ccd_energy(tdho, ccd_energy):
    tol = 1e-4

    cc_scheme = CoupledClusterDoubles(tdho, verbose=True)
    cc_scheme.iterate_t_amplitudes(tol=tol)
    energy = cc_scheme.compute_energy()

    assert abs(energy - ccd_energy) < tol
