import pytest
import warnings
import numpy as np

from quantum_systems import construct_pyscf_system_rhf

from coupled_cluster.mix import DIIS
from coupled_cluster import CCD, OACCD, RCCD, ROACCD, CCSD, RCCSD

"""
These tests tests if the Hellman-Feynman theorem
	\frac{d}{de}<\tilde{\Psi}|\hat{H}+e*\hat{V}|\Psi> = <\tilde{\Psi}|\hat{V}|\Psi>,
holds with \hat{V} = \hat{r}. 

The derivative is computed with the central finite difference approximation
	\frac{d}{de}<\tilde{\Psi}|\hat{H}+e*\hat{V}|\Psi> \approx (E(e+\Delta e) - E(e - \Delta e)) / (2*\Delta e) + O(\Delta e^2), 
where 
	E(e) = <\tilde{\Psi}|\hat{H}+e*\hat{V}|\Psi>.

Thus, for \hat{V} = \hat{r} we expect that
	<\tilde{\Psi}|\hat{r}|\Psi> \approx (E(e+\Delta e) - E(e - \Delta e)) / (2*\Delta e) + O(\Delta e^2). 	 
"""


def get_spas_system(e_str=0.01):
    system = construct_pyscf_system_rhf(
        "li 0.0 0.0 0.0; H 0.0 0.0 3.08",
        basis="cc-pvdz",
        np=np,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )
    system._basis_set.h = system.h + e_str * (
        system.position[0] + system.position[1] + system.position[2]
    ) / np.sqrt(3)
    return system


def get_gos_system(e_str=0.01):
    system = construct_pyscf_system_rhf(
        "li 0.0 0.0 0.0; H 0.0 0.0 3.08",
        basis="cc-pvdz",
        np=np,
        verbose=False,
        add_spin=True,
        anti_symmetrize=True,
    )
    system._basis_set.h = system.h + e_str * (
        system.position[0] + system.position[1] + system.position[2]
    ) / np.sqrt(3)
    return system


def test_rccd():
    def compute_rccd(e_str):
        system = get_spas_system(e_str)
        rccd = RCCD(system, mixer=DIIS, verbose=False)
        conv_tol = 1e-10
        rccd.compute_ground_state(
            t_kwargs=dict(tol=conv_tol), l_kwargs=dict(tol=conv_tol)
        )
        e_rccd = rccd.compute_energy()

        expec_r = (
            rccd.compute_one_body_expectation_value(
                system.position[0], make_hermitian=False
            )
            + rccd.compute_one_body_expectation_value(
                system.position[1], make_hermitian=False
            )
            + rccd.compute_one_body_expectation_value(
                system.position[2], make_hermitian=False
            )
        ) / np.sqrt(3)

        return e_rccd, expec_r

    e_str = 0.0
    de = np.array([1e-2, 1e-3, 1e-4])
    e_dir = 2

    expec_r = np.zeros(3)
    expec_r_fdm = np.zeros(3)

    print("### RCCD ###")
    for i in range(3):
        e_rccd, expec_r[i] = compute_rccd(e_str=e_str)
        e_rccd_p_de, expec_r_p_de = compute_rccd(e_str=e_str + de[i])
        e_rccd_m_de, expec_r_m_de = compute_rccd(e_str=e_str - de[i])

        expec_r_fdm[i] = (e_rccd_p_de - e_rccd_m_de) / (2 * de[i])
        print(expec_r[i], expec_r_fdm[i])
        print(expec_r[i] - expec_r_fdm[i], de[i])
        print()

    # Test that the errors are quadratic in \Delta e.
    assert abs(expec_r[0] - expec_r_fdm[0]) < 1e-3
    assert abs(expec_r[1] - expec_r_fdm[1]) < 1e-5
    assert abs(expec_r[2] - expec_r_fdm[2]) < 1e-7


def test_rccsd():
    def compute_rccsd(e_str):
        system = get_spas_system(e_str)
        rccsd = RCCSD(system, mixer=DIIS, verbose=False)
        conv_tol = 1e-10
        rccsd.compute_ground_state(
            t_kwargs=dict(tol=conv_tol), l_kwargs=dict(tol=conv_tol)
        )
        e_rccsd = rccsd.compute_energy()
        expec_r = (
            rccsd.compute_one_body_expectation_value(
                system.position[0], make_hermitian=False
            )
            + rccsd.compute_one_body_expectation_value(
                system.position[1], make_hermitian=False
            )
            + rccsd.compute_one_body_expectation_value(
                system.position[2], make_hermitian=False
            )
        ) / np.sqrt(3)

        return e_rccsd, expec_r

    e_str = 0.0
    de = np.array([1e-2, 1e-3, 1e-4])
    e_dir = 2

    expec_r = np.zeros(3)
    expec_r_fdm = np.zeros(3)

    print("### RCCSD ###")
    for i in range(3):
        e_rccsd, expec_r[i] = compute_rccsd(e_str=e_str)
        e_rccsd_p_de, expec_r_p_de = compute_rccsd(e_str=e_str + de[i])
        e_rccsd_m_de, expec_r_m_de = compute_rccsd(e_str=e_str - de[i])

        expec_r_fdm[i] = (e_rccsd_p_de - e_rccsd_m_de) / (2 * de[i])
        print(expec_r[i], expec_r_fdm[i])
        print(expec_r[i] - expec_r_fdm[i], de[i])
        print()

    assert abs(expec_r[0] - expec_r_fdm[0]) < 1e-1
    assert abs(expec_r[1] - expec_r_fdm[1]) < 1e-3
    assert abs(expec_r[2] - expec_r_fdm[2]) < 1e-5


def test_ccd():
    def compute_ccd(e_str):
        system = get_gos_system(e_str)
        ccd = CCD(system, mixer=DIIS, verbose=False)
        conv_tol = 1e-10
        ccd.compute_ground_state(
            t_kwargs=dict(tol=conv_tol), l_kwargs=dict(tol=conv_tol)
        )
        e_ccd = ccd.compute_energy()
        expec_r = (
            ccd.compute_one_body_expectation_value(
                system.position[0], make_hermitian=False
            )
            + ccd.compute_one_body_expectation_value(
                system.position[1], make_hermitian=False
            )
            + ccd.compute_one_body_expectation_value(
                system.position[2], make_hermitian=False
            )
        ) / np.sqrt(3)

        return e_ccd, expec_r

    e_str = 0.0
    de = np.array([1e-2, 1e-3, 1e-4])
    e_dir = 2

    expec_r = np.zeros(3)
    expec_r_fdm = np.zeros(3)

    print("### CCD ###")
    for i in range(3):
        e_ccd, expec_r[i] = compute_ccd(e_str=e_str)
        e_ccd_p_de, expec_r_p_de = compute_ccd(e_str=e_str + de[i])
        e_ccd_m_de, expec_r_m_de = compute_ccd(e_str=e_str - de[i])

        expec_r_fdm[i] = (e_ccd_p_de - e_ccd_m_de) / (2 * de[i])
        print(expec_r[i], expec_r_fdm[i])
        print(expec_r[i] - expec_r_fdm[i], de[i])
        print()

    assert abs(expec_r[0] - expec_r_fdm[0]) < 1e-3
    assert abs(expec_r[1] - expec_r_fdm[1]) < 1e-5
    assert abs(expec_r[2] - expec_r_fdm[2]) < 1e-7


def test_ccsd():
    def compute_ccsd(e_str):
        system = get_gos_system(e_str)
        ccsd = CCSD(system, mixer=DIIS, verbose=False)
        conv_tol = 1e-10
        ccsd.compute_ground_state(
            t_kwargs=dict(tol=conv_tol), l_kwargs=dict(tol=conv_tol)
        )
        e_ccsd = ccsd.compute_energy()
        expec_r = (
            ccsd.compute_one_body_expectation_value(
                system.position[0], make_hermitian=False
            )
            + ccsd.compute_one_body_expectation_value(
                system.position[1], make_hermitian=False
            )
            + ccsd.compute_one_body_expectation_value(
                system.position[2], make_hermitian=False
            )
        ) / np.sqrt(3)

        return e_ccsd, expec_r

    e_str = 0.0
    de = 0.0001
    e_dir = 2

    e_str = 0.0
    de = np.array([1e-2, 1e-3, 1e-4])
    e_dir = 2

    expec_r = np.zeros(3)
    expec_r_fdm = np.zeros(3)

    print("### CCSD ###")
    for i in range(3):
        e_ccsd, expec_r[i] = compute_ccsd(e_str=e_str)
        e_ccsd_p_de, expec_r_p_de = compute_ccsd(e_str=e_str + de[i])
        e_ccsd_m_de, expec_r_m_de = compute_ccsd(e_str=e_str - de[i])

        expec_r_fdm[i] = (e_ccsd_p_de - e_ccsd_m_de) / (2 * de[i])
        print(expec_r[i], expec_r_fdm[i])
        print(expec_r[i] - expec_r_fdm[i], de[i])
        print()

    assert abs(expec_r[0] - expec_r_fdm[0]) < 1e-1
    assert abs(expec_r[1] - expec_r_fdm[1]) < 1e-3
    assert abs(expec_r[2] - expec_r_fdm[2]) < 1e-5


def test_roaccd():
    def compute_roaccd(e_str):
        system = get_spas_system(e_str)
        roaccd = ROACCD(system, mixer=DIIS, verbose=False)
        conv_tol = 1e-10
        roaccd.compute_ground_state(
            max_iterations=100,
            num_vecs=10,
            tol=conv_tol,
            termination_tol=1e-12,
            tol_factor=1e-1,
        )
        e_roaccd = roaccd.compute_energy()
        expec_r = (
            roaccd.compute_one_body_expectation_value(
                system.position[0], make_hermitian=False
            )
            + roaccd.compute_one_body_expectation_value(
                system.position[1], make_hermitian=False
            )
            + roaccd.compute_one_body_expectation_value(
                system.position[2], make_hermitian=False
            )
        ) / np.sqrt(3)

        return e_roaccd, expec_r

    e_str = 0.0
    de = np.array([1e-2, 1e-3, 1e-4])
    e_dir = 2

    expec_r = np.zeros(3)
    expec_r_fdm = np.zeros(3)

    print("### ROACCD ###")
    for i in range(3):
        e_roaccd, expec_r[i] = compute_roaccd(e_str=e_str)
        e_roaccd_p_de, expec_r_p_de = compute_roaccd(e_str=e_str + de[i])
        e_roaccd_m_de, expec_r_m_de = compute_roaccd(e_str=e_str - de[i])

        expec_r_fdm[i] = (e_roaccd_p_de - e_roaccd_m_de) / (2 * de[i])
        print(expec_r[i], expec_r_fdm[i])
        print(expec_r[i] - expec_r_fdm[i], de[i])
        print()

    assert abs(expec_r[0] - expec_r_fdm[0]) < 1e-1
    assert abs(expec_r[1] - expec_r_fdm[1]) < 1e-3
    assert abs(expec_r[2] - expec_r_fdm[2]) < 1e-5


def test_oaccd():
    def compute_oaccd(e_str):
        system = get_gos_system(e_str)
        oaccd = OACCD(system, mixer=DIIS, verbose=False)
        conv_tol = 1e-10
        oaccd.compute_ground_state(
            max_iterations=100,
            num_vecs=10,
            tol=conv_tol,
            termination_tol=1e-12,
            tol_factor=1e-1,
        )
        e_oaccd = oaccd.compute_energy()
        expec_r = (
            oaccd.compute_one_body_expectation_value(
                system.position[0], make_hermitian=False
            )
            + oaccd.compute_one_body_expectation_value(
                system.position[1], make_hermitian=False
            )
            + oaccd.compute_one_body_expectation_value(
                system.position[2], make_hermitian=False
            )
        ) / np.sqrt(3)

        return e_oaccd, expec_r

    e_str = 0.0
    de = np.array([1e-2, 1e-3, 1e-4])
    e_dir = 2

    expec_r = np.zeros(3)
    expec_r_fdm = np.zeros(3)

    print("### OACCD ###")
    for i in range(3):
        e_oaccd, expec_r[i] = compute_oaccd(e_str=e_str)
        e_oaccd_p_de, expec_r_p_de = compute_oaccd(e_str=e_str + de[i])
        e_oaccd_m_de, expec_r_m_de = compute_oaccd(e_str=e_str - de[i])

        expec_r_fdm[i] = (e_oaccd_p_de - e_oaccd_m_de) / (2 * de[i])
        print(expec_r[i], expec_r_fdm[i])
        print(expec_r[i] - expec_r_fdm[i], de[i])
        print()

    assert abs(expec_r[0] - expec_r_fdm[0]) < 1e-1
    assert abs(expec_r[1] - expec_r_fdm[1]) < 1e-3
    assert abs(expec_r[2] - expec_r_fdm[2]) < 1e-5
