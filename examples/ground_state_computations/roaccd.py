import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.mix import DIIS, AlphaMixer
from coupled_cluster import ROACCD

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


roaccd = ROACCD(system, verbose=True)
roaccd.compute_ground_state(
    max_iterations=100,
    num_vecs=10,
    tol=1e-10,
    termination_tol=1e-10,
    tol_factor=1e-1,
    change_system_basis=True,
)
print(
    "EOACCD={0}".format(
        roaccd.compute_energy() + system.nuclear_repulsion_energy
    )
)
