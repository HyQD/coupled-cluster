import pytest
import warnings
import numpy as np

from quantum_systems import construct_pyscf_system_rhf

from coupled_cluster.mix import DIIS
from coupled_cluster import CCSD

from coupled_cluster.ccsd.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
    add_s2a_t,
    add_s2b_t,
    add_s2c_t,
    add_s4a_t,
    add_s4b_t,
    add_s4c_t,
    add_d4a_t,
    add_d4b_t,
    add_d5a_t,
    add_d5b_t,
    add_d5c_t,
    add_d5e_t,
    add_d5g_t,
    add_d5d_t,
    add_d5f_t,
    add_d5h_t,
    add_d6a_t,
    add_d6b_t,
    add_d6c_t,
    add_d7a_t,
    add_d7b_t,
    add_d7c_t,
    add_d7d_t,
    add_d7e_t,
    add_d8a_t,
    add_d8b_t,
    add_d9_t,
)
from coupled_cluster.ccsd.rhs_l import (
    compute_l_1_amplitudes,
    compute_l_2_amplitudes,
    add_s4a_l,
    add_s4b_l,
    add_s6a_l,
    add_s6b_l,
    add_s6c_l,
    add_s6d_l,
    add_s7_l,
    add_s9a_l,
    add_s9b_l,
    add_s9c_l,
    add_s10a_l,
    add_s10b_l,
    add_s10c_l,
    add_s10d_l,
    add_s10e_l,
    add_s10f_l,
    add_s10g_l,
    add_s11a_l,
    add_s11b_l,
    add_s11c_l,
    add_s11d_l,
    add_s11e_l,
    add_s11i_l,
    add_s11j_l,
    add_s11k_l,
    add_s11l_l,
    add_s11m_l,
    add_s11n_l,
    add_s11o_l,
    add_s12a_l,
    add_s12b_l,
    add_d4a_l,
    add_d4b_l,
    add_d5a_l,
    add_d5b_l,
    add_d7a_l,
    add_d7b_l,
    add_d7c_l,
    add_d8a_l,
    add_d8b_l,
    add_d8c_l,
    add_d8d_l,
    add_d10a_l,
    add_d10b_l,
    add_d11a_l,
    add_d11b_l,
    add_d11c_l,
    add_d12a_l,
    add_d12b_l,
    add_d12c_l,
)
from coupled_cluster.ccsd.density_matrices import (
    compute_one_body_density_matrix,
    add_rho_ba,
    add_rho_ia,
    add_rho_ai,
    add_rho_ji,
)


@pytest.fixture
def iterated_ccsd_amplitudes():
    ccsd_list = []
    for system in [
        construct_pyscf_system_rhf("he"),
        construct_pyscf_system_rhf("be"),
        construct_pyscf_system_rhf("ne"),
    ]:
        ccsd = CCSD(system, mixer=DIIS, verbose=True)
        ccsd.iterate_t_amplitudes()
        ccsd.iterate_l_amplitudes()

        ccsd_list.append(ccsd)

    return ccsd_list


def test_add_s2a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s2a_t(f, t_2, o, v, out, np=np)
    out_e = np.einsum("kc, acik->ai", f[o, v], t_2, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s2b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s2b_t(u, t_2, o, v, out, np=np)
    out_e = 0.5 * np.einsum("akcd, cdik->ai", u[v, o, v, v], t_2, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s2c_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s2c_t(u, t_2, o, v, out, np=np)
    out_e = -0.5 * np.einsum(
        "klic, ackl->ai", u[o, o, o, v], t_2, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s4a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s4a_t(u, t_1, t_2, o, v, out, np=np)
    out_e = -0.5 * np.einsum(
        "klcd, ci, adkl->ai", u[o, o, v, v], t_1, t_2, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s4b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s4b_t(u, t_1, t_2, o, v, out, np=np)
    out_e = -0.5 * np.einsum(
        "klcd, ak, cdil->ai", u[o, o, v, v], t_1, t_2, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s4c_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s4c_t(u, t_1, t_2, o, v, out, np=np)
    out_e = np.einsum(
        "klcd, ck, dali->ai", u[o, o, v, v], t_1, t_2, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d4a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d4a_t(u, t_1, o, v, out, np=np)
    out_e = np.einsum("abcj, ci->abij", u[v, v, v, o], t_1, optimize=True)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d4b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d4b_t(u, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "kbij, ak->abij", u[o, v, o, o], t_1, optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d5a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d5a_t(f, t_1, t_2, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "kc, ci, abkj->abij", f[o, v], t_1, t_2, optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d5b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d5b_t(f, t_1, t_2, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "kc, ak, cbij->abij", f[o, v], t_1, t_2, optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d5c_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d5c_t(u, t_1, t_2, o, v, out, np=np)
    out_e = np.einsum(
        "akcd, ci, dbkj->abij", u[v, o, v, v], t_1, t_2, optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d5e_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d5e_t(u, t_1, t_2, o, v, out, np=np)
    out_e = (-0.5) * np.einsum(
        "kbcd, ak, cdij->abij", u[o, v, v, v], t_1, t_2, optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d5g_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d5g_t(u, t_1, t_2, o, v, out, np=np)
    out_e = np.einsum(
        "kacd, ck, dbij->abij", u[o, v, v, v], t_1, t_2, optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d5d_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d5d_t(u, t_1, t_2, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "klic, ak, cblj->abij", u[o, o, o, v], t_1, t_2, optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d5f_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d5f_t(u, t_1, t_2, o, v, out, np=np)
    out_e = (0.5) * np.einsum(
        "klcj, ci, abkl->abij", u[o, o, v, o], t_1, t_2, optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d5h_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d5h_t(u, t_1, t_2, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "klci, ck, ablj->abij", u[o, o, v, o], t_1, t_2, optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d6a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d6a_t(u, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "abcd, ci, dj->abij", u[v, v, v, v], t_1, t_1, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d6b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d6b_t(u, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "klij, ak, bl->abij", u[o, o, o, o], t_1, t_1, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d6c_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d6c_t(u, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "kbcj, ci, ak->abij", u[o, v, v, o], t_1, t_1, optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d7a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d7a_t(u, t_1, t_2, o, v, out, np=np)
    out_e = (0.5) * np.einsum(
        "klcd, ci, abkl, dj->abij", u[o, o, v, v], t_1, t_2, t_1, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d7b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d7b_t(u, t_1, t_2, o, v, out, np=np)
    out_e = (0.5) * np.einsum(
        "klcd, ak, cdij, bl->abij", u[o, o, v, v], t_1, t_2, t_1, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d7c_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d7c_t(u, t_1, t_2, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "klcd, ci, ak, dblj->abij", u[o, o, v, v], t_1, t_1, t_2, optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d7d_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d7d_t(u, t_1, t_2, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "klcd, ck, di, ablj->abij", u[o, o, v, v], t_1, t_1, t_2, optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d7e_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d7e_t(u, t_1, t_2, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "klcd, ck, al, dbij->abij", u[o, o, v, v], t_1, t_1, t_2, optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d8a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d8a_t(u, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "bkcd, ci, ak, dj->abij", u[v, o, v, v], t_1, t_1, t_1, optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d9b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d8b_t(u, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "klcj, ci, ak, bl->abij", u[o, o, v, o], t_1, t_1, t_1, optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d9_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_2)
    add_d9_t(u, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "klcd, ci, dj, ak, bl", u[o, o, v, v], t_1, t_1, t_1, t_1, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


# L diagrams


def test_add_s4a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s4a_l(u, l_2, o, v, out, np=np)
    out_e = (0.5) * np.einsum(
        "ijbc, bcaj->ia", l_2, u[v, v, v, o], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s4b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s4b_l(u, l_2, o, v, out, np=np)
    out_e = (-0.5) * np.einsum(
        "jkab, ibjk->ia", l_2, u[o, v, o, o], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s6a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s6a_l(u, l_2, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "ijbc, bk, ckaj->ia", l_2, t_1, u[v, o, v, o], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s6b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s6b_l(u, l_2, t_1, o, v, out, np=np)
    out_e = (0.5) * np.einsum(
        "ijbc, dj, bcad->ia", l_2, t_1, u[v, v, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s6c_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s6c_l(u, l_2, t_1, o, v, out, np=np)
    out_e = (0.5) * np.einsum(
        "jkab, bl, iljk->ia", l_2, t_1, u[o, o, o, o], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s6d_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s6d_l(u, l_2, t_2, o, v, out, np=np)
    out_e = (0.5) * np.einsum(
        "jkbc, bdjk, icad->ia", l_2, t_2, u[o, v, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s7_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s7_l(u, l_1, t_2, o, v, out, np=np)
    out_e = np.einsum(
        "jb, bcjk, ikac->ia", l_1, t_2, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s9a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s9a_l(u, l_2, t_2, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "ijbc, bdjk, ckad->ia", l_2, t_2, u[v, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s9b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s9b_l(u, l_2, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "jkab, cj, ibck->ia", l_2, t_1, u[o, v, v, o], optimize=True
    )


def test_add_s9c_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s9c_l(u, l_2, t_2, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "jkab, bcjl, ilck->ia", l_2, t_2, u[o, o, v, o], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s10a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s10a_l(f, l_2, t_2, o, v, out, np=np)
    out_e = (-0.5) * np.einsum(
        "ib, jkac, bcjk->ia", f[o, v], l_2, t_2, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s10b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s10b_l(f, l_2, t_2, o, v, out, np=np)
    out_e = (-0.5) * np.einsum(
        "ja, ikbc, bcjk->ia", f[o, v], l_2, t_2, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s10c_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s10c_l(u, l_1, t_2, o, v, out, np)
    out_e = (-0.5) * np.einsum(
        "ib, bcjk, jkac->ia", l_1, t_2, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s10d_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s10d_l(u, l_1, t_2, o, v, out, np=np)
    out_e = (-0.5) * np.einsum(
        "ja, bcjk, ikbc->ia", l_1, t_2, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s10e_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s10e_l(u, l_2, t_2, o, v, out, np=np)
    out_e = (-0.5) * np.einsum(
        "jkbc, bcjl, ilak->ia", l_2, t_2, u[o, o, v, o], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s10f_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s10f_l(u, l_2, t_2, o, v, out, np=np)
    out_e = (-0.25) * np.einsum(
        "jkab, cdjk, ibcd->ia", l_2, t_2, u[o, v, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s10g_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s10g_l(u, l_2, t_2, o, v, out, np=np)
    out_e = (0.25) * np.einsum(
        "ijbc, bckl, klaj->ia", l_2, t_2, u[o, o, v, o], optimize=True
    )


def test_add_s11a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11a_l(u, l_2, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "ijbc, bk, dj, ckad->ia", l_2, t_1, t_1, u[v, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11b_l(u, l_2, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "jkab, bl, cj, ilck->ia", l_2, t_1, t_1, u[o, o, v, o], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11c_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11c_l(u, l_2, t_1, o, v, out, np=np)
    out_e = (0.5) * np.einsum(
        "jkab, ck, dj, ibcd->ia", l_2, t_1, t_1, u[o, v, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11d_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11d_l(u, l_2, t_1, t_2, o, v, out, np=np)
    out_e = (0.5) * np.einsum(
        "jkbc, bl, cdjk, ilad->ia", l_2, t_1, t_2, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11e_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11e_l(u, l_2, t_1, t_2, o, v, out, np=np)
    out_e = (0.5) * np.einsum(
        "jkbc, dj, bckl, ilad->ia", l_2, t_1, t_2, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11i_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11i_l(u, l_2, t_1, t_2, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "ijbc, bk, cdjl, klad->ia", l_2, t_1, t_2, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11j_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11j_l(u, l_2, t_1, t_2, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "jkab, cj, bdkl, ilcd->ia", l_2, t_1, t_2, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11k_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11k_l(u, l_2, t_1, o, v, out, np=np)
    out_e = (-0.5) * np.einsum(
        "ijbc, bl, ck, klaj->ia", l_2, t_1, t_1, u[o, o, v, o], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11l_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11l_l(u, l_2, t_1, t_2, o, v, out, np=np)
    out_e = (-0.5) * np.einsum(
        "ijbc, dk, bcjl, klad->ia", l_2, t_1, t_2, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11m_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11m_l(u, l_2, t_1, t_2, o, v, out, np=np)
    out_e = (-0.5) * np.einsum(
        "jkab, cl, bdjk, ilcd->ia", l_2, t_1, t_2, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11n_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11n_l(u, l_2, t_1, t_2, o, v, out, np=np)
    out_e = (0.25) * np.einsum(
        "ijbc, dj, bckl, klad->ia", l_2, t_1, t_2, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11o_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11o_l(u, l_2, t_1, t_2, o, v, out, np=np)
    out_e = (0.25) * np.einsum(
        "jkab, bl, cdjk, ilcd->ia", l_2, t_1, t_2, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s12a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s12a_l(u, l_2, t_1, o, v, out, np=np)
    out_e = (-0.5) * np.einsum(
        "ijbc, bl, ck, dj, klad->ia",
        l_2,
        t_1,
        t_1,
        t_1,
        u[o, o, v, v],
        optimize=True,
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s12b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s12b_l(u, l_2, t_1, o, v, out, np=np)
    out_e = (-0.5) * np.einsum(
        "jkab, bl, ck, dj, ilcd->ia",
        l_2,
        t_1,
        t_1,
        t_1,
        u[o, o, v, v],
        optimize=True,
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d4a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d4a_l(u, l_2, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "ijcd, ck, dkab->ijab", l_2, t_1, u[v, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d4b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d4b_l(u, l_2, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "klab, ck, ijcl->ijab", l_2, t_1, u[o, o, v, o], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d5a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d5a_l(u, l_1, o, v, out, np=np)
    out_e = np.einsum("ka, ijbk->ijab", l_1, u[o, o, v, o])
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d5b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d5b_l(u, l_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "ic, jcab->ijab", l_1, u[o, v, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d7a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d7a_l(f, l_1, o, v, out, np=np)
    out_e = np.einsum("ia, jb->ijab", f[o, v], l_1, optimize=True)
    out_e -= out_e.swapaxes(2, 3)
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d7b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d7b_l(f, l_2, t_1, o, v, out, np=np)
    out_e = np.einsum("ic, jkab, ck->ijab", f[o, v], l_2, t_1, optimize=True)
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d7c_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d7c_l(f, l_2, t_1, o, v, out, np=np)
    out_e = np.einsum("ka, ijbc, ck->ijab", f[o, v], l_2, t_1, optimize=True)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d8a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d8a_l(u, l_1, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "ic, ck, jkab->ijab", l_1, t_1, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d8b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d8b_l(u, l_1, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "ka, ck, ijbc->ijab", l_1, t_1, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d8c_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d8c_l(u, l_2, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "ijac, dk, ckbd->ijab", l_2, t_1, u[v, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d8d_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d8d_l(u, l_2, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "ikab, cl, jlck->ijab", l_2, t_1, u[o, o, v, o], optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d10a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d10a_l(u, l_2, t_1, o, v, out, np=np)
    out_e = (-0.5) * np.einsum(
        "klab, cl, dk, ijcd->ijab", l_2, t_1, t_1, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d10b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d10b_l(u, l_2, t_1, o, v, out, np=np)
    out_e = (-0.5) * np.einsum(
        "ijcd, cl, dk, klab->ijab", l_2, t_1, t_1, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d11a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d11a_l(u, l_1, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "ia, ck, jkbc->ijab", l_1, t_1, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d11b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d11b_l(u, l_2, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "ikac, dk, jcbd->ijab", l_2, t_1, u[o, v, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d11c_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d11c_l(u, l_2, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "ikac, cl, jlbk->ijab", l_2, t_1, u[o, o, v, o], optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d12a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d12a_l(u, l_2, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "ijac, ck, dl, klbd->ijab", l_2, t_1, t_1, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(2, 3)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d12b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d12b_l(u, l_2, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "ikab, ck, dl, jlcd->ijab", l_2, t_1, t_1, u[o, o, v, v], optimize=True
    )
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_d12c_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d12c_l(u, l_2, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "ikac, cl, dk, jlbd->ijab", l_2, t_1, t_1, u[o, o, v, v], optimize=True
    )

    out_e -= out_e.swapaxes(2, 3)
    out_e -= out_e.swapaxes(0, 1)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_auto_gen_rhs(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    f = cs.f
    u = cs.u
    o = cs.o
    v = cs.v

    t_1_auto = T1_RHS(t_1.T.copy(), t_2.transpose(2, 3, 0, 1).copy(), f, u)
    t_1_diag = compute_t_1_amplitudes(f, u, t_1, t_2, o, v, np=np)

    np.testing.assert_allclose(t_1_diag, t_1_auto.T)

    t_2_auto = T2_RHS(t_1.T.copy(), t_2.transpose(2, 3, 0, 1).copy(), f, u)
    t_2_diag = compute_t_2_amplitudes(f, u, t_1, t_2, o, v, np=np)

    np.testing.assert_allclose(
        t_2_diag, t_2_auto.transpose(2, 3, 0, 1), atol=1e-10
    )

    l_1_auto = L1_RHS(
        t_1.T.copy(), t_2.transpose(2, 3, 0, 1).copy(), l_1, l_2, f, u
    )
    l_1_diag = compute_l_1_amplitudes(f, u, t_1, t_2, l_1, l_2, o, v, np=np)

    np.testing.assert_allclose(l_1_diag, l_1_auto)

    l_2_auto = L2_RHS(
        t_1.T.copy(), t_2.transpose(2, 3, 0, 1).copy(), l_1, l_2, f, u
    )
    l_2_diag = compute_l_2_amplitudes(f, u, t_1, t_2, l_1, l_2, o, v, np=np)

    np.testing.assert_allclose(l_2_diag, l_2_auto, atol=1e-8)


# Density matrix tests


def test_one_body_density_matrix(iterated_ccsd_amplitudes):
    ccsd_list = iterated_ccsd_amplitudes

    for ccsd in ccsd_list:
        rho_qp = ccsd.compute_one_body_density_matrix()

        assert abs(np.trace(rho_qp) - ccsd.n) < 1e-8


# def test_mbpt_enegy(tdho):
#
#     cc_scheme = CCSD(tdho, verbose=True)
#     energy = cc_scheme.compute_energy()
#
#     assert True


# def test_ccsd_energy(tdho, ccsd_energy):
#     tol = 1e-4
#
#     cc_scheme = CCSD(tdho, verbose=True)
#     cc_scheme.iterate_t_amplitudes(tol=tol)
#     energy = cc_scheme.compute_energy()
#
#     assert abs(energy - ccsd_energy) < tol


# def test_lambda_amplitude_iterations(tdho):
#     cc_scheme = CCSD(tdho, verbose=True)
#
#     cc_scheme.iterate_t_amplitudes()
#     energy = cc_scheme.compute_energy()
#     cc_scheme.iterate_t_amplitudes()
#
#     assert True


def T1_RHS(T1, T2, F, W):
    N, L = T1.shape[0], F.shape[0]
    o, v = slice(0, N), slice(N, L)

    result = np.einsum(
        "Ic,Ac->IA", T1, F[v, v], optimize=["einsum_path", (0, 1)]
    )
    result += np.einsum(
        "kc,AkIc->IA", T1, W[v, o, o, v], optimize=["einsum_path", (0, 1)]
    )
    result += np.einsum(
        "IkAc,kc->IA", T2, F[o, v], optimize=["einsum_path", (0, 1)]
    )
    result += 0.5 * np.einsum(
        "Ikdc,Akdc->IA", T2, W[v, o, v, v], optimize=["einsum_path", (0, 1)]
    )
    result += -1.0 * np.einsum(
        "kA,kI->IA", T1, F[o, o], optimize=["einsum_path", (0, 1)]
    )
    result += -0.5 * np.einsum(
        "lkAc,lkIc->IA", T2, W[o, o, o, v], optimize=["einsum_path", (0, 1)]
    )
    result += np.einsum(
        "kc,Id,Akdc->IA",
        T1,
        T1,
        W[v, o, v, v],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += np.einsum(
        "kc,IlAd,lkdc->IA",
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "kA,Ildc,lkdc->IA",
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "Ic,lkAd,lkdc->IA",
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "kA,Ic,kc->IA",
        T1,
        T1,
        F[o, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "lA,kc,lkIc->IA",
        T1,
        T1,
        W[o, o, o, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "lA,kc,Id,lkdc->IA",
        T1,
        T1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 3), (1, 2), (0, 1)],
    )
    result += np.einsum("AI->IA", F[v, o], optimize=["einsum_path", (0,)])
    return result


def T2_RHS(T1, T2, F, W):
    N, L = T1.shape[0], F.shape[0]
    o, v = slice(0, N), slice(N, L)

    result = 0.5 * np.einsum(
        "lkAB,lkIJ->IJAB", T2, W[o, o, o, o], optimize=["einsum_path", (0, 1)]
    )
    result += 0.5 * np.einsum(
        "IJdc,ABdc->IJAB", T2, W[v, v, v, v], optimize=["einsum_path", (0, 1)]
    )
    temp = np.einsum(
        "kA,BkIJ->IJAB", T1, W[v, o, o, o], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = np.einsum(
        "IJAc,Bc->IJAB", T2, F[v, v], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += -1.0 * np.einsum(
        "kA,lB,lkIJ->IJAB",
        T1,
        T1,
        W[o, o, o, o],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "Ic,Jd,ABdc->IJAB",
        T1,
        T1,
        W[v, v, v, v],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    temp = -1.0 * np.einsum(
        "Ic,ABJc->IJAB", T1, W[v, v, o, v], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    temp = -1.0 * np.einsum(
        "IkAB,kJ->IJAB", T2, F[o, o], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += 0.25 * np.einsum(
        "lkAB,IJdc,lkdc->IJAB",
        T2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    temp = np.einsum(
        "kA,IJBc,kc->IJAB",
        T1,
        T2,
        F[o, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = np.einsum(
        "Ic,JkAB,kc->IJAB",
        T1,
        T2,
        F[o, v],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    temp = np.einsum(
        "kc,IJAd,Bkdc->IJAB",
        T1,
        T2,
        W[v, o, v, v],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = np.einsum(
        "IkAc,BkJc->IJAB", T2, W[v, o, o, v], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)
    temp = 0.5 * np.einsum(
        "kA,IJdc,Bkdc->IJAB",
        T1,
        T2,
        W[v, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = 0.5 * np.einsum(
        "IJAc,lkBd,lkdc->IJAB",
        T2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = -1.0 * np.einsum(
        "kc,IlAB,lkJc->IJAB",
        T1,
        T2,
        W[o, o, o, v],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    temp = -1.0 * np.einsum(
        "JkAc,IlBd,lkdc->IJAB",
        T2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += -0.5 * np.einsum(
        "kA,lB,IJdc,lkdc->IJAB",
        T1,
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (2, 3), (0, 2), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "Ic,Jd,lkAB,lkdc->IJAB",
        T1,
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 3), (0, 2), (0, 1)],
    )
    temp = -0.5 * np.einsum(
        "Ic,lkAB,lkJc->IJAB",
        T1,
        T2,
        W[o, o, o, v],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    temp = -0.5 * np.einsum(
        "IlAB,Jkdc,lkdc->IJAB",
        T2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += np.einsum(
        "kA,lB,Ic,Jd,lkdc->IJAB",
        T1,
        T1,
        T1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (2, 4), (2, 3), (0, 2), (0, 1)],
    )
    temp = np.einsum(
        "kA,lB,Ic,lkJc->IJAB",
        T1,
        T1,
        T1,
        W[o, o, o, v],
        optimize=["einsum_path", (2, 3), (0, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    temp = np.einsum(
        "lA,kc,IJBd,lkdc->IJAB",
        T1,
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 3), (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = np.einsum(
        "Ic,JkAd,Bkdc->IJAB",
        T1,
        T2,
        W[v, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)
    temp = np.einsum(
        "kc,Id,JlAB,lkdc->IJAB",
        T1,
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 3), (0, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    temp = -1.0 * np.einsum(
        "kA,Ic,Jd,Bkdc->IJAB",
        T1,
        T1,
        T1,
        W[v, o, v, v],
        optimize=["einsum_path", (1, 3), (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = -1.0 * np.einsum(
        "kA,Ic,BkJc->IJAB",
        T1,
        T1,
        W[v, o, o, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)
    temp = -1.0 * np.einsum(
        "kA,IlBc,lkJc->IJAB",
        T1,
        T2,
        W[o, o, o, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)
    temp = -1.0 * np.einsum(
        "kA,Ic,JlBd,lkdc->IJAB",
        T1,
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (2, 3), (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)
    result += np.einsum(
        "ABIJ->IJAB", W[v, v, o, o], optimize=["einsum_path", (0,)]
    )
    return result


def L1_RHS(T1, T2, L1, L2, F, W):
    N, L = T1.shape[0], F.shape[0]
    o, v = slice(0, N), slice(N, L)

    result = np.einsum(
        "Ic,cA->IA", L1, F[v, v], optimize=["einsum_path", (0, 1)]
    )
    result += np.einsum(
        "kc,IcAk->IA", L1, W[o, v, v, o], optimize=["einsum_path", (0, 1)]
    )
    result += np.einsum(
        "kc,IkAc->IA", T1, W[o, o, v, v], optimize=["einsum_path", (0, 1)]
    )
    result += 0.5 * np.einsum(
        "Ikdc,dcAk->IA", L2, W[v, v, v, o], optimize=["einsum_path", (0, 1)]
    )
    result += -1.0 * np.einsum(
        "kA,Ik->IA", L1, F[o, o], optimize=["einsum_path", (0, 1)]
    )
    result += -0.5 * np.einsum(
        "lkAc,Iclk->IA", L2, W[o, v, o, o], optimize=["einsum_path", (0, 1)]
    )
    result += np.einsum(
        "Ic,kd,ckAd->IA",
        L1,
        T1,
        W[v, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += np.einsum(
        "kA,lc,Ilck->IA",
        L1,
        T1,
        W[o, o, v, o],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += np.einsum(
        "kc,kd,IcAd->IA",
        L1,
        T1,
        W[o, v, v, v],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += np.einsum(
        "kc,lkdc,IlAd->IA",
        L1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += np.einsum(
        "lkAc,kd,Icdl->IA",
        L2,
        T1,
        W[o, v, v, o],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += np.einsum(
        "lkAc,mkdc,Imdl->IA",
        L2,
        T2,
        W[o, o, v, o],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "Ic,lkdc,lkAd->IA",
        L1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "kA,lkdc,Ildc->IA",
        L1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "Ikdc,ke,dcAe->IA",
        L2,
        T1,
        W[v, v, v, v],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "lkAc,mc,Imlk->IA",
        L2,
        T1,
        W[o, o, o, o],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "Ic,kc,kA->IA",
        L1,
        T1,
        F[o, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "kA,kc,Ic->IA",
        L1,
        T1,
        F[o, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "kc,lc,IlAk->IA",
        L1,
        T1,
        W[o, o, v, o],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "Ikdc,lc,dlAk->IA",
        L2,
        T1,
        W[v, o, v, o],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "Ikdc,lkce,dlAe->IA",
        L2,
        T2,
        W[v, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "Ikdc,lkdc,lA->IA",
        L2,
        T2,
        F[o, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "lkAc,lkdc,Id->IA",
        L2,
        T2,
        F[o, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "lkdc,lkce,IdAe->IA",
        L2,
        T2,
        W[o, v, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "lkdc,mkdc,ImAl->IA",
        L2,
        T2,
        W[o, o, v, o],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += -0.25 * np.einsum(
        "lkAc,lkde,Icde->IA",
        L2,
        T2,
        W[o, v, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += 0.25 * np.einsum(
        "Ikdc,lmdc,lmAk->IA",
        L2,
        T2,
        W[o, o, v, o],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += np.einsum(
        "Ic,kc,ld,lkAd->IA",
        L1,
        T1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (2, 3), (0, 1), (0, 1)],
    )
    result += np.einsum(
        "kA,kc,ld,Ildc->IA",
        L1,
        T1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (2, 3), (1, 2), (0, 1)],
    )
    result += np.einsum(
        "lkAc,kd,lmce,Imde->IA",
        L2,
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 2), (1, 2), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "Ikdc,mc,ld,lmAk->IA",
        L2,
        T1,
        T1,
        W[o, o, v, o],
        optimize=["einsum_path", (0, 1), (1, 2), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "Ikdc,le,mkdc,lmAe->IA",
        L2,
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 2), (0, 1), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "kc,lc,kd,IlAd->IA",
        L1,
        T1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "Ikdc,lc,ke,dlAe->IA",
        L2,
        T1,
        T1,
        W[v, o, v, v],
        optimize=["einsum_path", (0, 1), (1, 2), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "Ikdc,lc,mkde,lmAe->IA",
        L2,
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 2), (1, 2), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "lkAc,mc,kd,Imdl->IA",
        L2,
        T1,
        T1,
        W[o, o, v, o],
        optimize=["einsum_path", (2, 3), (0, 2), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "lkAc,ld,ke,Icde->IA",
        L2,
        T1,
        T1,
        W[o, v, v, v],
        optimize=["einsum_path", (1, 3), (0, 2), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "lkAc,md,lkce,Imde->IA",
        L2,
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 2), (0, 1), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "lkdc,mc,lkde,ImAe->IA",
        L2,
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 2), (0, 2), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "lkdc,ke,lmdc,ImAe->IA",
        L2,
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 2), (0, 1), (0, 1)],
    )
    result += 0.25 * np.einsum(
        "Ikdc,ke,lmdc,lmAe->IA",
        L2,
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 2), (1, 2), (0, 1)],
    )
    result += 0.25 * np.einsum(
        "lkAc,mc,lkde,Imde->IA",
        L2,
        T1,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (2, 3), (0, 2), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "Ikdc,mc,ld,ke,lmAe->IA",
        L2,
        T1,
        T1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 3), (1, 2), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "lkAc,mc,ld,ke,Imde->IA",
        L2,
        T1,
        T1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 2), (1, 2), (0, 1)],
    )
    result += np.einsum("IA->IA", F[o, v], optimize=["einsum_path", (0,)])
    return result


def L2_RHS(T1, T2, L1, L2, F, W):
    N, L = T1.shape[0], F.shape[0]
    o, v = slice(0, N), slice(N, L)

    result = 0.5 * np.einsum(
        "IJdc,dcAB->IJAB", L2, W[v, v, v, v], optimize=["einsum_path", (0, 1)]
    )
    result += 0.5 * np.einsum(
        "lkAB,IJlk->IJAB", L2, W[o, o, o, o], optimize=["einsum_path", (0, 1)]
    )
    temp = np.einsum(
        "kA,IJBk->IJAB", L1, W[o, o, v, o], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = np.einsum(
        "IJAc,cB->IJAB", L2, F[v, v], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = -1.0 * np.einsum(
        "Ic,JcAB->IJAB", L1, W[o, v, v, v], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += -1.0 * np.einsum(
        "IJdc,kc,dkAB->IJAB",
        L2,
        T1,
        W[v, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    temp = -1.0 * np.einsum(
        "IkAB,Jk->IJAB", L2, F[o, o], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += -1.0 * np.einsum(
        "lkAB,kc,IJcl->IJAB",
        L2,
        T1,
        W[o, o, v, o],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += 0.25 * np.einsum(
        "IJdc,lkdc,lkAB->IJAB",
        L2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += 0.25 * np.einsum(
        "lkAB,lkdc,IJdc->IJAB",
        L2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    temp = np.einsum(
        "IA,JB->IJAB", L1, F[o, v], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)
    temp = np.einsum(
        "Ic,kc,JkAB->IJAB",
        L1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    temp = np.einsum(
        "kA,kc,IJBc->IJAB",
        L1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = np.einsum(
        "IJAc,kd,ckBd->IJAB",
        L2,
        T1,
        W[v, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = np.einsum(
        "IkAB,lc,Jlck->IJAB",
        L2,
        T1,
        W[o, o, v, o],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    temp = np.einsum(
        "IkAc,JcBk->IJAB", L2, W[o, v, v, o], optimize=["einsum_path", (0, 1)]
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)
    temp = 0.5 * np.einsum(
        "IJAc,lkdc,lkBd->IJAB",
        L2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = 0.5 * np.einsum(
        "IkAB,lkdc,Jldc->IJAB",
        L2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    temp = 0.5 * np.einsum(
        "Ikdc,lkdc,JlAB->IJAB",
        L2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    temp = 0.5 * np.einsum(
        "lkAc,lkdc,IJBd->IJAB",
        L2,
        T2,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = -1.0 * np.einsum(
        "IJAc,kc,kB->IJAB",
        L2,
        T1,
        F[o, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = -1.0 * np.einsum(
        "IkAB,kc,Jc->IJAB",
        L2,
        T1,
        F[o, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += -0.5 * np.einsum(
        "IJdc,lc,kd,lkAB->IJAB",
        L2,
        T1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "lkAB,lc,kd,IJdc->IJAB",
        L2,
        T1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 3), (1, 2), (0, 1)],
    )
    temp = np.einsum(
        "IA,kc,JkBc->IJAB",
        L1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)
    temp = np.einsum(
        "IJAc,kc,ld,lkBd->IJAB",
        L2,
        T1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (2, 3), (0, 1), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 2, 3)
    temp = np.einsum(
        "IkAB,kc,ld,Jldc->IJAB",
        L2,
        T1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (2, 3), (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    temp = np.einsum(
        "IkAc,kd,JcBd->IJAB",
        L2,
        T1,
        W[o, v, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)
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
    temp = -1.0 * np.einsum(
        "IkAc,lc,JlBk->IJAB",
        L2,
        T1,
        W[o, o, v, o],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)
    temp = -1.0 * np.einsum(
        "IkAc,lc,kd,JlBd->IJAB",
        L2,
        T1,
        T1,
        W[o, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    )
    result += temp
    result += (-1) * np.swapaxes(temp, 0, 1)
    result += (-1) * np.swapaxes(temp, 2, 3)
    result += (1) * np.swapaxes(np.swapaxes(temp, 2, 3), 0, 1)
    result += np.einsum(
        "IJAB->IJAB", W[o, o, v, v], optimize=["einsum_path", (0,)]
    )

    return result
