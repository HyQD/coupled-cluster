import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.mix import DIIS, AlphaMixer
from coupled_cluster.rccsd import RCCSD


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

    rccsd = RCCSD(system, mixer=DIIS, verbose=False)

    conv_tol = 1e-10
    t_kwargs = dict(tol=conv_tol)
    l_kwargs = dict(tol=conv_tol)

    rccsd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)
    e_rccsd = rccsd.compute_energy()

    """
    Potentially compute more propeties such as dipole moment
    """
    return e_rccsd


def test_rccsd():

    molecule = "li 0.0 0.0 0.0;h 0.0 0.0 3.08"
    basis = "cc-pvdz"
    e_rccsd_dalton = -8.0147418652916809
    e_rccsd = compute_ground_state_properties(molecule, basis)
    np.testing.assert_approx_equal(e_rccsd, e_rccsd_dalton, significant=8)

    molecule = "be 0.0 0.0 0.0"
    basis = "aug-cc-pvdz"
    e_rccsd_dalton = -14.617433363077
    e_rccsd = compute_ground_state_properties(molecule, basis)
    np.testing.assert_approx_equal(e_rccsd, e_rccsd_dalton, significant=8)


if __name__ == "__main__":
    test_rccsd()
