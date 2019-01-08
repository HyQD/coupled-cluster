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
)


def test_add_d1_t(large_system):
    t, large_system = large_system
    u = large_system.u
    o = large_system.o
    v = large_system.v

    out = np.zeros_like(t)
    add_d1_t(u, o, v, out, np=np)

    np.testing.assert_allclose(out, u[v, v, o, o], atol=1e-10)


def test_add_d2a_t(large_system):
    t, large_system = large_system
    f = large_system.f
    o = large_system.o
    v = large_system.v

    out = np.zeros_like(t)
    add_d2a_t(f, t, o, v, out, np=np)
    out_e = np.einsum("bc, acij -> abij", f[v, v], t)
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2b_t(large_system):
    t, large_system = large_system
    f = large_system.f
    o = large_system.o
    v = large_system.v

    out = np.zeros_like(t)
    add_d2b_t(f, t, o, v, out, np=np)
    out_e = -np.einsum("kj, abik -> abij", f[o, o], t)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2c_t(large_system):
    t, large_system = large_system
    u = large_system.u
    o = large_system.o
    v = large_system.v

    out = np.zeros_like(t)
    add_d2c_t(u, t, o, v, out, np=np)
    out_e = 0.5 * np.einsum("cdij, abcd -> abij", t, u[v, v, v, v])

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2d_t(large_system):
    t, large_system = large_system
    u = large_system.u
    o = large_system.o
    v = large_system.v

    out = np.zeros_like(t)
    add_d2d_t(u, t, o, v, out, np=np)
    out_e = 0.5 * np.einsum("abkl, klij -> abij", t, u[o, o, o, o])

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d2e_t(large_system):
    t, large_system = large_system
    u = large_system.u
    o = large_system.o
    v = large_system.v

    out = np.zeros_like(t)
    add_d2e_t(u, t, o, v, out, np=np)
    out_e = np.einsum("acik, bkjc -> abij", t, u[v, o, o, v])
    out_e -= out_e.swapaxes(0, 1)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3a_t(large_system):
    t, large_system = large_system
    u = large_system.u
    o = large_system.o
    v = large_system.v

    out = np.zeros_like(t)
    add_d3a_t(u, t, o, v, out, np=np)
    out_e = 0.25 * np.einsum(
        "cdij, abkl, klcd -> abij", t, t, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3b_t(large_system):
    t, large_system = large_system
    u = large_system.u
    o = large_system.o
    v = large_system.v

    out = np.zeros_like(t)
    add_d3b_t(u, t, o, v, out, np=np)
    out_e = np.einsum(
        "acik, bdjl, klcd -> abij", t, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3c_t(large_system):
    t, large_system = large_system
    u = large_system.u
    o = large_system.o
    v = large_system.v

    out = np.zeros_like(t)
    add_d3c_t(u, t, o, v, out, np=np)
    out_e = -0.5 * np.einsum(
        "ablj, dcik, klcd -> abij", t, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d3d_t(large_system):
    t, large_system = large_system
    u = large_system.u
    o = large_system.o
    v = large_system.v

    out = np.zeros_like(t)
    add_d3d_t(u, t, o, v, out, np=np)
    out_e = -0.5 * np.einsum(
        "aclk, dbij, klcd -> abij", t, t, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


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
