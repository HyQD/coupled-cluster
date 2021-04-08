import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.mix import DIIS
from coupled_cluster.cc2 import CC2
from coupled_cluster.cc2.energies import lagrangian_functional


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

    cc2 = CC2(system, mixer=DIIS, verbose=False)

    conv_tol = 1e-10
    t_kwargs = dict(tol=conv_tol)
    l_kwargs = dict(tol=conv_tol)

    cc2.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)
    e_cc2 = cc2.compute_energy() + e_nuc

    """
    Tests the lagrangian functional
    """
    h_tranform, f_transform, u_transform = cc2.t1_transform_integrals

    e_ref_lagrangian = cc2.compute_reference_energy(
            h_transform, u_transform, cc2.o, cc2.v, cc2.np
        )
   

    print("e ref lagrangian")
    print(e_ref_lagrangian)

    """
    Tests the one-body part of the lagrangian functional
    """

    o, v = (system.o, system.v)

    dipole_moment_array = system.dipole_moment
    cc2_dipole_moment = np.zeros(3)

    for i in range(3):

        dipole_moment_t1_transformed = cc2.t1_transform_integrals_one_body(
            dipole_moment_array[i]
        )

        cc2_dipole_moment[i] = lagrangian_functional(
            dipole_moment_t1_transformed,
            dipole_moment_t1_transformed,
            system.u * 0,
            cc2.t_1,
            cc2.t_2,
            cc2.l_1,
            cc2.l_2,
            cc2.o,
            cc2.v,
            np,
        ).real

        cc2_dipole_moment[i] = cc2_dipole_moment[i] + 2 * np.trace(
            dipole_moment_t1_transformed[o, o]
        )

    return e_cc2, cc2_dipole_moment


def test_cc2():

    molecule = "li 0.0 0.0 0.0;h 0.0 0.0 3.08"
    basis = "6-31G"
    e_cc2_psi4 = -7.992515440819747
    dipole_z_cc2_psi4 = -2.26335
    e_cc2, dipole_cc2 = compute_ground_state_properties(molecule, basis)

    np.testing.assert_approx_equal(e_cc2, e_cc2_psi4, significant=8)
    np.testing.assert_approx_equal(
        dipole_cc2[2], dipole_z_cc2_psi4, significant=4
    )

 #   molecule = "be 0.0 0.0 0.0"
 #   basis = "6-31G"
 #   e_rcc2_psi4 = -14.591159613927022
 #   dipole_z_rcc2_psi4 = 0.0000
 #   e_rcc2, dipole_rcc2 = compute_ground_state_properties(molecule, basis)

 #   np.testing.assert_approx_equal(e_cc2, e_cc2_psi4, significant=8)
 #   np.testing.assert_approx_equal(
 #       dipole_cc2[2], dipole_z_cc2_psi4, significant=8
 #   )


if __name__ == "__main__":
    test_cc2()
