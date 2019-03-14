import pytest
import warnings
import numpy as np
from coupled_cluster.mix import DIIS
from quantum_systems import construct_psi4_system
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles
import coupled_cluster.ccd.rhs_l as ccd_l
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
    add_s1_l,
    add_s2a_l,
    add_s2b_l,
    add_s3a_l,
    add_s3b_l,
    add_s4a_l,
    add_s4b_l,
    add_s5a_l,
    add_s5b_l,
    add_s5c_l,
    add_s5d_l,
    add_s6a_l,
    add_s6b_l,
    add_s6c_l,
    add_s6d_l,
    add_s7_l,
    add_s8a_l,
    add_s8b_l,
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
    add_s11f_l,
    add_s11g_l,
    add_s11h_l,
    add_s11i_l,
    add_s11j_l,
    add_s11k_l,
    add_s11l_l,
    add_s11m_l,
    add_s11n_l,
    add_s11o_l,
    add_s12a_l,
    add_s12b_l,
    add_d3b_l,
    add_d4a_l,
    add_d4b_l,
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
def ccsd_helium_system():
    He = """
    He 0.0 0.0 0.0
    symmetry c1
    """
    options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-8}

    return construct_psi4_system(He, options)


@pytest.fixture
def ccsd_beryllium_system():
    Be = """
    Be 0.0 0.0 0.0
    symmetry c1
    """
    options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-8}

    return construct_psi4_system(Be, options)


@pytest.fixture
def ccsd_neon_system():
    Ne = """
    Ne 0.0 0.0 0.0
    symmetry c1
    """
    options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-8}

    return construct_psi4_system(Ne, options)


@pytest.fixture
def iterated_ccsd_amplitudes(
    ccsd_helium_system, ccsd_beryllium_system, ccsd_neon_system
):
    ccsd_list = []
    for system in [ccsd_helium_system, ccsd_beryllium_system, ccsd_neon_system]:
        try:
            from tdhf import HartreeFock

            hf = HartreeFock(system)
            C = hf.scf(tolerance=1e-8)
            system.change_basis(C)
        except ImportError:
            warnings.warn("Running without Hartree-Fock basis")

        ccsd = CoupledClusterSinglesDoubles(system, mixer=DIIS, verbose=False)
        ccsd.iterate_t_amplitudes()
        ccsd.iterate_l_amplitudes()

        ccsd_list.append(ccsd)

    return ccsd_list


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


def test_add_s3a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s3a_t(f, t_1, o, v, out, np=np)
    out_e = np.einsum("ac, ci->ai", f[v, v], t_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s3b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s3b_t(f, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum("ki, ak->ai", f[o, o], t_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s3c_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s3c_t(u, t_1, o, v, out, np=np)
    out_e = np.einsum("akic, ck->ai", u[v, o, o, v], t_1, optimize=True)

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


def test_add_s5a_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s5a_t(f, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum("kc, ci, ak->ai", f[o, v], t_1, t_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5b_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s5b_t(u, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "akcd, ci, dk->ai", u[v, o, v, v], t_1, t_1, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5c_t(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(t_1)
    add_s5c_t(u, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "klic, ak, cl->ai", u[o, o, o, v], t_1, t_1, optimize=True
    )

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


def test_add_s1_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s1_l(f, o, v, out, np)
    out_e = f[o, v]

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s2a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s2a_l(f, l_1, o, v, out, np=np)
    out_e = np.einsum("ba, ib->ia", f[v, v], l_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s2b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s2b_l(f, l_1, o, v, out, np=np)
    out_e = (-1) * np.einsum("ij, ja->ia", f[o, o], l_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s3a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s3a_l(u, l_1, o, v, out, np=np)
    out_e = np.einsum("jb, ibaj->ia", l_1, u[o, v, v, o], optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s3b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s3b_l(u, t_1, o, v, out, np=np)
    out_e = np.einsum("bj, jiba->ia", t_1, u[o, o, v, v], optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


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


def test_add_s5a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s5a_l(u, l_1, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "ib, cj, bjac->ia", l_1, t_1, u[v, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s5b_l(u, l_1, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "ja, bk, ikbj->ia", l_1, t_1, u[o, o, v, o], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5c_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s5c_l(u, l_1, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "jb, cj, ibac->ia", l_1, t_1, u[o, v, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5d_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s5d_l(u, l_1, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "jb, bk, ikaj->ia", l_1, t_1, u[o, o, v, o], optimize=True
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


def test_add_s8a_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s8a_l(f, l_1, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum("ib, ja, bj->ia", f[o, v], l_1, t_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s8b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    f = cs.f
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s8b_l(f, l_1, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum("ja, ib, bj->ia", f[o, v], l_1, t_1, optimize=True)

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


def test_add_s11f_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11f_l(u, l_1, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "ib, bj, ck, jkac->ia", l_1, t_1, t_1, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11g_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11g_l(u, l_1, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "ja, bj, ck, ikbc->ia", l_1, t_1, t_1, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11h_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_1)
    add_s11h_l(u, l_1, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "jb, bk, cj, ikac->ia", l_1, t_1, t_1, u[o, o, v, v], optimize=True
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


def test_add_d3b_l(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    u = cs.u
    o = cs.o
    v = cs.v

    out = np.zeros_like(l_2)
    add_d3b_l(u, l_1, o, v, out, np=np)
    out_e = np.einsum("ka, ijbk->ijab", l_1, u[o, o, v, o])
    out_e -= out_e.swapaxes(2, 3)

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


# Density matrix tests


def test_one_body_density_matrix(iterated_ccsd_amplitudes):
    ccsd_list = iterated_ccsd_amplitudes

    for ccsd in ccsd_list:
        rho_qp = ccsd.compute_one_body_density_matrix()

        assert abs(np.trace(rho_qp) - ccsd.n) < 1e-8


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
