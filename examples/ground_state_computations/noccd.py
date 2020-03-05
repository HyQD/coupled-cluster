import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.mix import DIIS, AlphaMixer
from coupled_cluster import OACCD

molecule = "li 0.0 0.0 0.0;h 0.0 0.0 3.08"
basis = "cc-pvdz"
system = construct_pyscf_system_rhf(
    molecule,
    basis=basis,
    np=np,
    verbose=False,
    add_spin=True,
    anti_symmetrize=True,
)


oaccd = OACCD(system, verbose=True)
oaccd.compute_ground_state(
    max_iterations=100,
    num_vecs=6,
    tol=1e-8,
    termination_tol=1e-8,
    tol_factor=1e-1,
    change_system_basis=True,
)
print("EOACCD={0}".format(oaccd.compute_energy()))

dm1 = oaccd.compute_one_body_density_matrix()

dip_mom_z = np.trace(np.dot(dm1, system.dipole_moment[2]))
print(f"dip_mom_z={dip_mom_z}")
