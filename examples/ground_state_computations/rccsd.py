import numpy as np
from quantum_systems import construct_pyscf_system_rhf

molecule = "li 0.0 0.0 0.0;h 0.0 0.0 3.08"
basis = "cc-pvdz"

from coupled_cluster.mix import DIIS, AlphaMixer
from coupled_cluster.rccsd import RCCSD
system = construct_pyscf_system_rhf(
    molecule,
    basis=basis,
    np=np,
    verbose=False,
    add_spin=False,
    anti_symmetrize=False,
)
rccsd = RCCSD(system,mixer=DIIS, verbose=True)

conv_tol = 1e-8
t_kwargs = dict(tol=conv_tol)
l_kwargs = dict(tol=conv_tol)

rccsd.compute_ground_state(t_kwargs=t_kwargs,l_kwargs=l_kwargs)
print("Ground state energy: {0}".format(rccsd.compute_energy()))
dm1 = rccsd.compute_one_body_density_matrix()
for i in range(3):
    dip_xi = np.trace(np.dot(dm1,system.dipole_moment[i]))
    print(f"dipole moment x{i}={dip_xi}")