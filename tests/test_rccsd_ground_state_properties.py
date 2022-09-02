import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.mix import DIIS, AlphaMixer
from coupled_cluster.rccsd import RCCSD


def test_two_body_density_matrix():

    molecule = "li 0.0 0.0 0.0; h 0.0 0.0 3.08"
    basis = "cc-pvdz"

    system = construct_pyscf_system_rhf(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )

    rccsd = RCCSD(system, mixer=DIIS, verbose=False)

    conv_tol = 1e-10
    t_kwargs = dict(tol=conv_tol)
    l_kwargs = dict(tol=conv_tol)

    rccsd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)
    e_rccsd = rccsd.compute_energy()

    rho_qp = rccsd.compute_one_body_density_matrix()
    rho_rspq = rccsd.compute_two_body_density_matrix()
    assert (
        np.trace(np.trace(rho_rspq, axis1=0, axis2=2))
        - system.n * (system.n - 1)
        < 1e-10
    )  # This is a minimal (and useful) test, since only the elements rho^{pq}_{pq} contribute to the trace.
    expec_H = np.einsum("pq,qp", system.h, rho_qp) + 0.5 * np.einsum(
        "pqrs, rspq", system.u, rho_rspq
    )

    assert (e_rccsd - (expec_H + system.nuclear_repulsion_energy)) < 1e-10


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
    test_two_body_density_matrix()
    # test_rccsd()
