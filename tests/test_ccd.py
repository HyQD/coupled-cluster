import pytest
import numpy as np
from coupled_cluster.ccd import CoupledClusterDoubles
from coupled_cluster.ccd.rhs import (
    add_d1,
    add_d2a,
    add_d2b,
    add_d2c,
    add_d2d,
    add_d2e,
    add_d3a,
    add_d3b,
    add_d3c,
    add_d3d,
)


# def test_pppp_intermediate(large_system, large_t):
#    u = large_system.u
#    o = large_system.o
#    v = large_system.v
#    t = large_t
#
#    W_pppp = construct_pppp_intermediate(u, t, o, v, np=np)
#    W_pppp_e = (
#        0.25 * np.einsum("abkl, klcd -> abcd", t, u[o, o, v, v])
#        + 0.5 * u[v, v, v, v]
#    )
#
#    np.testing.assert_allclose(W_pppp, W_pppp_e, atol=1e-10)
#
#
# def test_hh_intermediate(large_system, large_t):
#    u = large_system.u
#    o = large_system.o
#    v = large_system.v
#    t = large_t
#
#    W_hh = construct_hh_intermediate(u, t, o, v, np=np)
#    W_hh_e = 0.5 * np.einsum("cdjk, klcd -> lj", t, u[o, o, v, v])
#
#    np.testing.assert_allclose(W_hh, W_hh_e, atol=1e-10)
#
#
# def test_pp_intermediate(large_system, large_t):
#    u = large_system.u
#    o = large_system.o
#    v = large_system.v
#    t = large_t
#
#    W_pp = construct_pp_intermediate(u, t, o, v, np=np)
#    W_pp_e = 0.5 * np.einsum("bdkl, klcd -> bc", t, u[o, o, v, v])
#
#    np.testing.assert_allclose(W_pp, W_pp_e, atol=1e-10)
#
#
# def test_phhp_intermediate(large_system, large_t):
#    u = large_system.u
#    o = large_system.o
#    v = large_system.v
#    t = large_t
#
#    W_phhp = construct_phhp_intermediate(u, t, o, v, np=np)
#    W_phhp_e = (
#        0.5 * np.einsum("bdjl, klcd -> bkjc", t, u[o, o, v, v]) + u[v, o, o, v]
#    )
#
#    np.testing.assert_allclose(W_phhp, W_phhp_e, atol=1e-10)


def test_add_d1(large_system, large_t):
    u = large_system.u
    o = large_system.o
    v = large_system.v
    t = large_t

    out = np.zeros_like(t)
    add_d1(u, o, v, out, np=np)

    np.testing.assert_allclose(out, u[v, v, o, o], atol=1e-10)


def test_add_d2a(large_system, large_t):
    f = large_system.f
    o = large_system.o
    v = large_system.v
    t = large_t

    out = np.zeros_like(t)
    add_d2a(f, t, o, v, out, np=np)
    out_e = np.einsum("bc, acij -> abij", f[v, v], t)
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2b(large_system, large_t):
    f = large_system.f
    o = large_system.o
    v = large_system.v
    t = large_t

    out = np.zeros_like(t)
    add_d2b(f, t, o, v, out, np=np)
    out_e = -np.einsum("kj, abik -> abij", f[o, o], t)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2c(large_system, large_t):
    u = large_system.u
    o = large_system.o
    v = large_system.v
    t = large_t

    out = np.zeros_like(t)
    add_d2c(u, t, o, v, out, np=np)
    out_e = 0.5 * np.einsum("cdij, abcd -> abij", t, u[v, v, v, v])

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2d(large_system, large_t):
    u = large_system.u
    o = large_system.o
    v = large_system.v
    t = large_t

    out = np.zeros_like(t)
    add_d2d(u, t, o, v, out, np=np)
    out_e = 0.5 * np.einsum("abkl, klij -> abij", t, u[o, o, o, o])

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2e(large_system, large_t):
    u = large_system.u
    o = large_system.o
    v = large_system.v
    t = large_t

    out = np.zeros_like(t)
    add_d2e(u, t, o, v, out, np=np)
    out_e = np.einsum("acik, bkjc -> abij", t, u[v, o, o, v])
    out_e -= out_e.swapaxes(0, 1)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3a(large_system, large_t):
    u = large_system.u
    o = large_system.o
    v = large_system.v
    t = large_t

    out = np.zeros_like(t)
    add_d3a(u, t, o, v, out, np=np)
    out_e = 0.25 * np.einsum(
        "cdij, abkl, klcd -> abij", t, t, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3b(large_system, large_t):
    u = large_system.u
    o = large_system.o
    v = large_system.v
    t = large_t

    out = np.zeros_like(t)
    add_d3b(u, t, o, v, out, np=np)
    out_e = np.einsum(
        "acik, bdjl, klcd -> abij", t, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3c(large_system, large_t):
    u = large_system.u
    o = large_system.o
    v = large_system.v
    t = large_t

    out = np.zeros_like(t)
    add_d3c(u, t, o, v, out, np=np)
    out_e = -0.5 * np.einsum(
        "ablj, dcik, klcd -> abij", t, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3d(large_system, large_t):
    u = large_system.u
    o = large_system.o
    v = large_system.v
    t = large_t

    out = np.zeros_like(t)
    add_d3d(u, t, o, v, out, np=np)
    out_e = -0.5 * np.einsum(
        "aclk, dbij, klcd -> abij", t, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


# def test_add_d2c_and_d3a(large_system, large_t):
#    f = large_system.f
#    u = large_system.u
#    o = large_system.o
#    v = large_system.v
#    t = large_t
#
#    W_pppp = construct_pppp_intermediate(u, t, o, v, np=np)
#    out = np.zeros_like(t)
#    add_d2c_and_d3a(t, W_pppp, o, v, out, np=np)
#
#    out_d2c = 0.5 * np.einsum("cdij, abcd -> abij", t, u[v, v, v, v])
#    out_d3a = 0.25 * np.einsum(
#        "cdij, abkl, klcd -> abij", t, t, u[o, o, v, v], optimize=True
#    )
#
#    np.testing.assert_allclose(out, out_d2c + out_d3a, atol=1e-10)
#
#    out_ei = np.einsum("cdij, abcd -> abij", t, W_pppp)
#
#    np.testing.assert_allclose(out, out_ei, atol=1e-10)
#
#
# def test_add_d3c(large_system, large_t):
#    u = large_system.u
#    o = large_system.o
#    v = large_system.v
#    t = large_t
#
#    W_hh = construct_hh_intermediate(u, t, o, v, np=np)
#    out = np.zeros_like(t)
#    add_d3c(u, t, W_hh, o, v, out, np=np)
#
#    out_e = 0.5 * np.einsum(
#        "cdjk, abil, klcd -> abij", t, t, u[o, o, v, v], optimize=True
#    )
#    out_e -= out_e.swapaxes(2, 3)
#
#    np.testing.assert_allclose(out, out_e, atol=1e-10)
#
#    out_ei = np.einsum("abil, lj -> abij", t, W_hh)
#    out_ei -= out_ei.swapaxes(2, 3)
#
#    np.testing.assert_allclose(out, out_ei, atol=1e-10)
#
#
# def test_add_d2e_and_d3b(large_system, large_t):
#    u = large_system.u
#    o = large_system.o
#    v = large_system.v
#    t = large_t
#
#    W_phhp = construct_phhp_intermediate(u, t, o, v, np=np)
#    out = np.zeros_like(t)
#    add_d2e_and_d3b(u, t, W_phhp, o, v, out, np=np)
#
#    out_d2e = np.einsum(
#        "acik, bkjc -> abij", t, u[v, o, o, v]
#    )
#    out_d2e -= out_d2e.swapaxes(0, 1)
#    out_d2e -= out_d2e.swapaxes(2, 3)
#
#    out_d3b = np.einsum(
#        "acik, bdjl, klcd -> abij", t, t, u[o, o, v, v], optimize=True
#    )
#    out_d3b -= out_d3b.swapaxes(0, 1)
#
#    np.testing.assert_allclose(
#        out, out_d2e + out_d3b, atol=1e-10
#    )
#
#    out_ei = np.einsum(
#        "acik, bkjc -> abij", t, W_phhp
#    )
#    out_ei -= out_ei.swapaxes(0, 1)
#    out_ei -= out_ei.swapaxes(2, 3)
#
#    np.testing.assert_allclose(
#        out, out_ei, atol=1e-10
#    )


def test_reference_energy(tdho, ref_energy):
    tol = 1e-4

    cc_scheme = CoupledClusterDoubles(tdho, verbose=True)
    e_ref = cc_scheme.compute_reference_energy()

    assert abs(e_ref - ref_energy) < tol


def test_ccd_energy(tdho, ccd_energy):
    tol = 1e-4

    cc_scheme = CoupledClusterDoubles(tdho, verbose=True)
    energy, _ = cc_scheme.compute_ground_state_energy(tol=tol)

    assert abs(energy - ccd_energy) < tol
