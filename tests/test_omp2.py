import pytest

from quantum_systems import construct_pyscf_system_rhf

from coupled_cluster.omp2.omp2 import OMP2
from coupled_cluster.mix import DIIS


"""
@pytest.fixture
def he_groundstate_oaccd():
    return -2.887594831090973
"""


def test_omp2_groundstate_pyscf():

    molecule = "b 0.0 0.0 0.0;h 0.0 0.0 2.4"
    basis = "dzp"

    system = construct_pyscf_system_rhf(molecule, basis=basis)

    omp2 = OMP2(system, mixer=DIIS, verbose=False)
    omp2.compute_ground_state(
        max_iterations=100,
        num_vecs=10,
        tol=1e-10,
        termination_tol=1e-12,
        tol_factor=1e-1,
    )

    print(omp2.compute_energy().real + system.nuclear_repulsion_energy)


def test_omp2_groundstate_psi4():

    bh = """
        units au
        b 0.0 0.0 0.0
        h 0.0 0.0 2.4
        symmetry c1
        """
    basis = "dzp"

    system = make_psi4_system(bh, basis=basis)

    omp2 = OMP2(system, mixer=DIIS, verbose=True)
    omp2.compute_ground_state(
        max_iterations=100,
        num_vecs=10,
        tol=1e-10,
        termination_tol=1e-12,
        tol_factor=1e-1,
    )

    print(omp2.compute_energy().real + system.nuclear_repulsion_energy)


def make_psi4_system(geometry, basis):

    import numpy as np
    import psi4

    from quantum_systems import (
        BasisSet,
        SpatialOrbitalSystem,
        GeneralOrbitalSystem,
        QuantumSystem,
    )

    # Psi4 setup
    psi4.set_memory("2 GB")
    psi4.core.set_output_file("output.dat", False)
    mol = psi4.geometry(geometry)

    # roots per irrep must be set to do the eom calculation with psi4
    psi4.set_options(
        {"basis": basis, "e_convergence": 1e-10, "d_convergence": 1e-10}
    )
    rhf_e, rhf_wfn = psi4.energy("SCF", return_wfn=True)
    omp2_e = psi4.energy("omp2")

    wfn = rhf_wfn
    ndocc = wfn.doccpi()[0]
    n_electrons = 2 * ndocc
    nmo = wfn.nmo()
    C = wfn.Ca()
    npC = np.asarray(C)

    mints = psi4.core.MintsHelper(wfn.basisset())
    H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    nmo = H.shape[0]

    # Update H, transform to MO basis
    H = np.einsum("uj,vi,uv", npC, npC, H)
    # Integral generation from Psi4's MintsHelper
    MO = np.asarray(mints.mo_eri(C, C, C, C))
    # Physicist notation
    MO = MO.swapaxes(1, 2)

    dipole_integrals = np.zeros((3, nmo, nmo))
    ints = mints.ao_dipole()
    for n in range(3):
        dipole_integrals[n] = np.einsum(
            "ui,uv,vj->ij", npC, np.asarray(ints[n]), npC, optimize=True
        )

    bs = BasisSet(nmo, dim=3, np=np)
    bs.h = H
    bs.s = np.eye(nmo)
    bs.u = MO
    bs.nuclear_repulsion_energy = mol.nuclear_repulsion_energy()
    bs.dipole_moment = dipole_integrals
    bs.change_module(np=np)
    system = SpatialOrbitalSystem(n_electrons, bs)

    return system.construct_general_orbital_system()


if __name__ == "__main__":
    test_omp2_groundstate_pyscf()
    test_omp2_groundstate_psi4()
