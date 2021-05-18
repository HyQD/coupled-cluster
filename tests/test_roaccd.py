import pytest

from quantum_systems import construct_pyscf_system_rhf

from coupled_cluster.ccd.oaccd import OACCD
from coupled_cluster.rccd.roaccd import ROACCD
from coupled_cluster.mix import DIIS


def test_roaccd_vs_oaccd():
    import numpy as np

    molecule = "b 0.0 0.0 0.0;h 0.0 0.0 2.4"
    basis = "cc-pvdz"

    system = construct_pyscf_system_rhf(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )

    conv_tol = 1e-10
    roaccd = ROACCD(system, mixer=DIIS, verbose=True)
    roaccd.compute_ground_state(
        max_iterations=100,
        num_vecs=10,
        tol=conv_tol,
        termination_tol=conv_tol,
        tol_factor=1e-1,
    )

    e_roaccd = roaccd.compute_energy()

    system = construct_pyscf_system_rhf(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=True,
        anti_symmetrize=True,
    )

    oaccd = OACCD(system, mixer=DIIS, verbose=True)
    oaccd.compute_ground_state(
        max_iterations=100,
        num_vecs=10,
        tol=conv_tol,
        termination_tol=conv_tol,
        tol_factor=1e-1,
    )

    e_oaccd = oaccd.compute_energy()
    assert abs(e_oaccd - e_roaccd) < conv_tol


if __name__ == "__main__":
    test_roaccd_vs_oaccd()
