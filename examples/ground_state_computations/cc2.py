import numpy as np
from quantum_systems import construct_pyscf_system_rhf

molecule = "li 0.0 0.0 0.0;h 0.0 0.0 3.08"
basis = "6-31G"

from coupled_cluster.mix import DIIS, AlphaMixer
from coupled_cluster.rcc2 import RCC2

system = construct_pyscf_system_rhf(
    molecule,
    basis=basis,
    np=np,
    verbose=False,
    add_spin=False,
    anti_symmetrize=False,
)

rccsd = RCC2(system, mixer=DIIS, verbose=True)

conv_tol = 1e-14
t_kwargs = dict(tol=conv_tol)
l_kwargs = dict(tol=conv_tol)

rccsd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)
dm1 = rccsd.compute_one_body_density_matrix()

for i in range(3):
    dip_xi = np.trace(np.dot(dm1, system.dipole_moment[i]))
    print(f"dipole moment x{i}={dip_xi}")

t1, t2 = rccsd.t_1, rccsd.t_2
l1, l2 = rccsd.l_2, rccsd.l_2

