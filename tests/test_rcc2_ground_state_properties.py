import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.mix import DIIS
from coupled_cluster.rcc2 import RCC2
from coupled_cluster.rcc2.energies import lagrangian_functional


def compute_ground_state_properties(molecule, basis):

    system = construct_pyscf_system_rhf(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )

    e_ref = system.compute_reference_energy()
    e_nuc = system.nuclear_repulsion_energy

    rcc2 = RCC2(system, mixer=DIIS, verbose=False)

    conv_tol = 1e-10
    t_kwargs = dict(tol=conv_tol)
    l_kwargs = dict(tol=conv_tol)

    rcc2.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)
    e_rcc2 = rcc2.compute_energy()

    """
    Tests the one-body part of the lagrangian functional
    """

    o, v = (system.o, system.v)

    dipole_moment_array = system.dipole_moment
    rcc2_dipole_moment = np.zeros(3)

    for i in range(3):

        dipole_moment_t1_transformed = rcc2.t1_transform_integrals_one_body(
            dipole_moment_array[i]
        )

        rcc2_dipole_moment[i] = lagrangian_functional(
            dipole_moment_t1_transformed,
            dipole_moment_t1_transformed,
            system.u * 0,
            rcc2.t_1,
            rcc2.t_2,
            rcc2.l_1,
            rcc2.l_2,
            rcc2.o,
            rcc2.v,
            np,
        ).real

        rcc2_dipole_moment[i] = rcc2_dipole_moment[i] + 2 * np.trace(
            dipole_moment_t1_transformed[o, o]
        )

    return e_rcc2, rcc2_dipole_moment


def test_rcc2():

    molecule = "li 0.0 0.0 0.0;h 0.0 0.0 3.08"
    basis = "6-31G"
    e_rcc2_psi4 = -7.992515440819747
    dipole_z_rcc2_psi4 = -2.26335
    e_rcc2, dipole_rcc2 = compute_ground_state_properties(molecule, basis)

    np.testing.assert_approx_equal(e_rcc2, e_rcc2_psi4, significant=8)
    np.testing.assert_approx_equal(
        dipole_rcc2[2], dipole_z_rcc2_psi4, significant=4
    )

    molecule = "be 0.0 0.0 0.0"
    basis = "6-31G"
    e_rcc2_psi4 = -14.591159613927022
    dipole_z_rcc2_psi4 = 0.0000
    e_rcc2, dipole_rcc2 = compute_ground_state_properties(molecule, basis)

    np.testing.assert_approx_equal(e_rcc2, e_rcc2_psi4, significant=8)
    np.testing.assert_approx_equal(
        dipole_rcc2[2], dipole_z_rcc2_psi4, significant=8
    )


if __name__ == "__main__":
    test_rcc2()
