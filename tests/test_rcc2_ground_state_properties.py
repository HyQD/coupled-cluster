import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.mix import DIIS, AlphaMixer
from coupled_cluster.rcc2 import RCC2


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
    e_rcc2 = rcc2.compute_energy() + e_nuc + e_ref
    dipole_rcc2 = rcc2.compute_dipole_moment()
    """
    Potentially compute more propeties such as dipole moment
    """
    return e_rcc2, dipole_rcc2


def test_rcc2():

    molecule = "li 0.0 0.0 0.0;h 0.0 0.0 3.08"
    basis = "6-31G"
    e_rcc2_psi4 = -7.992515440819747
    dipole_z_rcc2_psi4 = 2.2634
    e_rcc2, dipole_rcc2 = compute_ground_state_properties(molecule, basis)
    print("energy lithium")
    print(e_rcc2)
    print( e_rcc2_psi4)

    print("dipole moment lithium") 
    print(dipole_rcc2)
    print(dipole_z_rcc2_psi4)

    np.testing.assert_approx_equal(e_rcc2, e_rcc2_psi4, significant=8)
    np.testing.assert_approx_equal(dipole_rcc2[2], dipole_z_rcc2_psi4, significant=4)

    molecule = "be 0.0 0.0 0.0"
    basis = "6-31G"
    e_rcc2_psi4 = -14.591159613927022
    dipole_z_rcc2_psi4 = 0.0000
    e_rcc2, dipole_rcc2 = compute_ground_state_properties(molecule, basis)
   
    print("energy beryllium")
    print(e_rcc2)

    print("dipole moment beryllium")
    print(dipole_rcc2)
    np.testing.assert_approx_equal(e_rcc2, e_rcc2_psi4, significant=8)
    np.testing.assert_approx_equal(dipole_rcc2[2], dipole_z_rcc2_psi4, significant=8)


if __name__ == "__main__":
    test_rcc2()
