import pytest
import numpy as np

from coupled_cluster.ccsd import CoupledClusterSinglesDoubles
from coupled_cluster.ccsd.rhs_t import (
    add_s1_t,
    add_s2a_t,
    add_s2b_t,
    add_s2c_t,
    add_s3a_t,
    add_s3b_t,
    add_s3c_t,
    add_s4a_t,
    add_s4b_t,
    add_s4c_t,
    add_s5a_t,
    add_s5b_t,
    add_s5c_t,
    add_s6_t,
    add_d4a_t,
    add_d4b_t,
    add_d5a_t,
)


def test_add_s1_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s1_t(f, o, v, out, np=np)

    np.testing.assert_allclose(out, f[v, o], atol=1e-10)


def test_add_s2a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s2a_t(f, t_2, o, v, out, np=np)
    out_e = np.einsum("kc, acik->ai", f[o, v], t_2)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s2b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s2b_t(u, t_2, o, v, out, np=np)
    out_e = 0.5 * np.einsum("akcd, cdik->ai", u[v, o, v, v], t_2)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s2c_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s2c_t(u, t_2, o, v, out, np=np)
    out_e = -0.5 * np.einsum("klic, ackl->ai", u[o, o, o, v], t_2)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s3a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s3a_t(f, t_1, o, v, out, np=np)
    out_e = np.einsum("ac, ci->ai", f[v, v], t_1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s3b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s3b_t(f, t_1, o, v, out, np=np)
    out_e = -np.einsum("ki, ak->ai", f[o, o], t_1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s3c_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s3c_t(u, t_1, o, v, out, np=np)
    out_e = np.einsum("akic, ck->ai", u[v, o, o, v], t_1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s4a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s4a_t(u, t_1, t_2, o, v, out, np=np)
    out_e = -0.5 * np.einsum("klcd, ci, adkl->ai", u[o, o, v, v], t_1, t_2)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s4b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s4b_t(u, t_1, t_2, o, v, out, np=np)
    out_e = -0.5 * np.einsum("klcd, ak, cdil->ai", u[o, o, v, v], t_1, t_2)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s4c_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s4c_t(u, t_1, t_2, o, v, out, np=np)
    out_e = np.einsum("klcd, ck, dali->ai", u[o, o, v, v], t_1, t_2)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s5a_t(f, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum("kc, ci, ak->ai", f[o, v], t_1, t_1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s5b_t(u, t_1, o, v, out, np=np)
    out_e = np.einsum("akcd, ci, dk->ai", u[v, o, v, v], t_1, t_1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5c_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s5c_t(u, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum("klic, ak, cl->ai", u[o, o, o, v], t_1, t_1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s6_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s6_t(u, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "klcd, ci, ak, dl->ai", u[o, o, v, v], t_1, t_1, t_1
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d4a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d4a_t(u, t_1, o, v, out, np=np)
    out_e = np.einsum("abcj, ci->abij", u[v, v, v, o], t_1)
    out_e -= out_e.swapaxes(2, 3)
    
    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d4b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d4b_t(u, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum("kbij, ak->abij", u[o, v, o, o], t_1)
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)

def test_add_d5a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d5a_t(f, t_1, t_2, o, v, out, np=np)
    out_e = -np.einsum("kc, ci, abkj->abij", f[o, v], t_1, t_2)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_mbpt_enegy(tdho):

    cc_scheme = CoupledClusterSinglesDoubles(tdho, verbose=True)
    energy = cc_scheme.compute_energy()

    assert True


def test_ccsd_energy(tdho, ccsd_energy):
    tol = 1e-4

    cc_scheme = CoupledClusterSinglesDoubles(tdho, verbose=True)
    cc_scheme.iterate_t_amplitudes(tol=tol)
    energy = cc_scheme.compute_energy()

    assert abs(energy - ccsd_energy) < tol


def test_lambda_amplitude_iterations(tdho):
    cc_scheme = CoupledClusterSinglesDoubles(tdho, verbose=True)

    cc_scheme.iterate_t_amplitudes()
    energy = cc_scheme.compute_energy()
    cc_scheme.iterate_t_amplitudes()

    assert True
