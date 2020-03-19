import numpy as np
from quantum_systems import construct_pyscf_system_rhf

molecule = "li 0.0 0.0 0.0;h 0.0 0.0 3.08"
basis = "cc-pvdz"

from coupled_cluster.mix import DIIS, AlphaMixer
from coupled_cluster.rccd import RCCD
from coupled_cluster.ccd import CCD

system = construct_pyscf_system_rhf(
    molecule,
    basis=basis,
    np=np,
    verbose=False,
    add_spin=False,
    anti_symmetrize=False,
)

rccd = RCCD(system, mixer=DIIS, verbose=False)

conv_tol = 1e-10
t_kwargs = dict(tol=conv_tol)
l_kwargs = dict(tol=conv_tol)

rccd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)
e_rccd = rccd.compute_energy() + system.nuclear_repulsion_energy
print("ECCD: {0}".format(e_rccd))
dm1 = rccd.compute_one_body_density_matrix()
dip_mom_rccd = np.zeros(3)
for i in range(3):
    dip_mom_rccd[i] = np.trace(np.dot(dm1, system.dipole_moment[i]))
print("dipole moment: ", dip_mom_rccd)
