import pytest
import warnings
import numpy as np

from quantum_systems import construct_pyscf_system_rhf

from coupled_cluster.mix import DIIS
from coupled_cluster import CCD, OACCD, RCCD, ROACCD, CCSD, RCCSD

from coupled_cluster.ccsd.energies import (
    lagrangian_functional as ccsd_lagrangian,
)
from coupled_cluster.rccsd.energies import (
    lagrangian_functional as rccsd_lagrangian,
)
from coupled_cluster.ccd.energies import (
    compute_lagrangian_functional as ccd_lagrangian,
)
from coupled_cluster.rccd.energies import (
    compute_lagrangian_functional as rccd_lagrangian,
)


@pytest.fixture
def gos_system():
    system = construct_pyscf_system_rhf(
        "li 0.0 0.0 0.0; H 0.0 0.0 3.08",
        basis="cc-pvdz",
        np=np,
        verbose=False,
        add_spin=True,
        anti_symmetrize=True,
    )
    return system


@pytest.fixture
def sos_system():
    system = construct_pyscf_system_rhf(
        "li 0.0 0.0 0.0; H 0.0 0.0 3.08",
        basis="cc-pvdz",
        np=np,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )
    return system


def test_ccd(gos_system):
    ccd = CCD(gos_system, mixer=DIIS, verbose=False)

    conv_tol = 1e-10
    ccd.compute_ground_state(
        t_kwargs=dict(tol=conv_tol), l_kwargs=dict(tol=conv_tol)
    )
    e_ccd = ccd.compute_energy()

    lagrangian = gos_system.compute_reference_energy() + ccd_lagrangian(
        ccd.f,
        gos_system.u,
        ccd.t_2,
        ccd.l_2,
        gos_system.o,
        gos_system.v,
        np,
    )
    assert (
        abs(lagrangian - e_ccd) < conv_tol
    )  # This test is only true if the amplitudes satisfy the cc-equations.

    rho_qp = ccd.compute_one_body_density_matrix()
    rho_rspq = ccd.compute_two_body_density_matrix()
    assert np.trace(rho_qp) - gos_system.n < conv_tol
    assert (
        np.trace(np.trace(rho_rspq, axis1=0, axis2=2))
        - gos_system.n * (gos_system.n - 1)
        < conv_tol
    )

    expec_H = (
        ccd.compute_one_body_expectation_value(
            gos_system.h, make_hermitian=False
        )
        + ccd.compute_two_body_expectation_value(0.5 * gos_system.u, asym=True)
        + gos_system.nuclear_repulsion_energy
    )
    assert abs(expec_H - (lagrangian)) < conv_tol


def test_rccd(sos_system):
    rccd = RCCD(sos_system, mixer=DIIS, verbose=False)

    conv_tol = 1e-10
    rccd.compute_ground_state(
        t_kwargs=dict(tol=conv_tol), l_kwargs=dict(tol=conv_tol)
    )
    e_rccd = rccd.compute_energy()

    lagrangian = sos_system.compute_reference_energy() + rccd_lagrangian(
        rccd.f,
        sos_system.u,
        rccd.t_2,
        rccd.l_2,
        sos_system.o,
        sos_system.v,
        np,
    )
    assert (
        abs(lagrangian - e_rccd) < conv_tol
    )  # This test is only true if the amplitudes satisfy the cc-equations.

    rho_qp = rccd.compute_one_body_density_matrix()
    rho_rspq = rccd.compute_two_body_density_matrix()
    assert np.trace(rho_qp) - 2 * sos_system.n < conv_tol
    assert (
        np.trace(np.trace(rho_rspq, axis1=0, axis2=2))
        - 2 * sos_system.n * (2 * sos_system.n - 1)
        < conv_tol
    )

    expec_H = (
        rccd.compute_one_body_expectation_value(
            sos_system.h, make_hermitian=False
        )
        + rccd.compute_two_body_expectation_value(
            0.5 * sos_system.u, asym=False
        )
        + sos_system.nuclear_repulsion_energy
    )
    assert abs(expec_H - (lagrangian)) < conv_tol


def test_ccsd(gos_system):
    ccsd = CCSD(gos_system, mixer=DIIS, verbose=False)

    conv_tol = 1e-10
    ccsd.compute_ground_state(
        t_kwargs=dict(tol=conv_tol), l_kwargs=dict(tol=conv_tol)
    )
    e_ccsd = ccsd.compute_energy()

    lagrangian = gos_system.compute_reference_energy() + ccsd_lagrangian(
        ccsd.f,
        gos_system.u,
        ccsd.t_1,
        ccsd.t_2,
        ccsd.l_1,
        ccsd.l_2,
        gos_system.o,
        gos_system.v,
        np,
    )
    assert (
        abs(lagrangian - e_ccsd) < conv_tol
    )  # This test is only true if the amplitudes satisfy the cc-equations.

    rho_qp = ccsd.compute_one_body_density_matrix()
    rho_rspq = ccsd.compute_two_body_density_matrix()
    assert np.trace(rho_qp) - gos_system.n < conv_tol
    assert (
        np.trace(np.trace(rho_rspq, axis1=0, axis2=2))
        - gos_system.n * (gos_system.n - 1)
        < conv_tol
    )

    expec_H = (
        ccsd.compute_one_body_expectation_value(
            gos_system.h, make_hermitian=False
        )
        + ccsd.compute_two_body_expectation_value(0.5 * gos_system.u, asym=True)
        + gos_system.nuclear_repulsion_energy
    )
    assert abs(expec_H - (lagrangian)) < conv_tol


def test_rccsd(sos_system):
    rccsd = RCCSD(sos_system, mixer=DIIS, verbose=False)

    conv_tol = 1e-10
    rccsd.compute_ground_state(
        t_kwargs=dict(tol=conv_tol), l_kwargs=dict(tol=conv_tol)
    )
    e_rccsd = rccsd.compute_energy()

    lagrangian = sos_system.compute_reference_energy() + rccsd_lagrangian(
        rccsd.f,
        sos_system.u,
        rccsd.t_1,
        rccsd.t_2,
        rccsd.l_1,
        rccsd.l_2,
        sos_system.o,
        sos_system.v,
        np,
    )
    assert (
        abs(lagrangian - e_rccsd) < conv_tol
    )  # This test is only true if the amplitudes satisfy the cc-equations.

    rho_qp = rccsd.compute_one_body_density_matrix()
    rho_rspq = rccsd.compute_two_body_density_matrix()
    assert np.trace(rho_qp) - 2 * sos_system.n < conv_tol
    assert (
        np.trace(np.trace(rho_rspq, axis1=0, axis2=2))
        - 2 * sos_system.n * (2 * sos_system.n - 1)
        < conv_tol
    )

    expec_H = (
        rccsd.compute_one_body_expectation_value(
            sos_system.h, make_hermitian=False
        )
        + rccsd.compute_two_body_expectation_value(
            0.5 * sos_system.u, asym=False
        )
        + sos_system.nuclear_repulsion_energy
    )
    assert abs(expec_H - (lagrangian)) < conv_tol


def test_oaccd(gos_system):
    oaccd = OACCD(gos_system, mixer=DIIS, verbose=False)

    conv_tol = 1e-10
    oaccd.compute_ground_state(
        max_iterations=100,
        num_vecs=10,
        tol=conv_tol,
        termination_tol=1e-12,
        tol_factor=1e-1,
    )
    # oaccd.compute_energy() uses the density matrices internally to compute the energy, so
    # it is only necessary to compare with the explicit expression for ccd-lagrangian.
    e_oaccd = oaccd.compute_energy()
    lagrangian = gos_system.compute_reference_energy(
        oaccd.h, oaccd.u
    ) + ccd_lagrangian(
        oaccd.f,
        gos_system.u,
        oaccd.t_2,
        oaccd.l_2,
        gos_system.o,
        gos_system.v,
        np,
    )
    assert abs(lagrangian - e_oaccd) < conv_tol

    rho_qp = oaccd.compute_one_body_density_matrix()
    rho_rspq = oaccd.compute_two_body_density_matrix()
    assert np.trace(rho_qp) - gos_system.n < conv_tol
    assert (
        np.trace(np.trace(rho_rspq, axis1=0, axis2=2))
        - gos_system.n * (gos_system.n - 1)
        < conv_tol
    )


def test_roaccd(sos_system):
    roaccd = ROACCD(sos_system, mixer=DIIS, verbose=False)

    conv_tol = 1e-10
    roaccd.compute_ground_state(
        max_iterations=100,
        num_vecs=10,
        tol=conv_tol,
        termination_tol=1e-12,
        tol_factor=1e-1,
    )
    # roaccd.compute_energy() uses the density matrices internally to compute the energy, so
    # it is only necessary to compare with the explicit expression for ccd-lagrangian.
    e_roaccd = roaccd.compute_energy()
    lagrangian = sos_system.compute_reference_energy(
        roaccd.h, roaccd.u
    ) + rccd_lagrangian(
        roaccd.f,
        sos_system.u,
        roaccd.t_2,
        roaccd.l_2,
        sos_system.o,
        sos_system.v,
        np,
    )
    assert abs(lagrangian - e_roaccd) < conv_tol

    rho_qp = roaccd.compute_one_body_density_matrix()
    rho_rspq = roaccd.compute_two_body_density_matrix()
    assert np.trace(rho_qp) - 2 * sos_system.n < conv_tol
    assert (
        np.trace(np.trace(rho_rspq, axis1=0, axis2=2))
        - 2 * sos_system.n * (2 * sos_system.n - 1)
        < conv_tol
    )
