import numpy as np
from quantum_systems import CustomSystem
from pyscf import gto, scf, ao2mo
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles
from coupled_cluster.ccd.oaccd import OACCD


"""
Use PyScf to generate integrals. 
This can be substituted with other integrals, such as those from Quest.
"""
mol = gto.Mole()
mol.unit = "bohr"
mol.build(atom="be 0.0 0.0 0.0", basis="cc-pvdz", symmetry=False)

n_electrons = mol.nelectron
n_ao = mol.nao
n_spin_orbitals = 2 * n_ao

# Do RHF calculation
myhf = scf.RHF(mol)
myhf.conv_tol = 1e-10
ehf = myhf.kernel()

# Change to MO basis
H = myhf.mo_coeff.T.dot(myhf.get_hcore()).dot(myhf.mo_coeff)
# Store full eri tensor and switch to rank 4 representation and physicists notation
eri = ao2mo.kernel(mol, myhf.mo_coeff, compact=False)
eri = np.asarray(eri).reshape(n_ao, n_ao, n_ao, n_ao).transpose(0, 2, 1, 3)

# Make a custom system, change to spin orbital basis and antisymmetrize eri tensor
# Even indices correspond to alpha-spin and odd indices to beta-spin
system = CustomSystem(n_electrons, n_spin_orbitals)
system.set_h(H, add_spin=True)
system.set_u(eri, add_spin=True, anti_symmetrize=True)
system.set_nuclear_repulsion_energy(0)

"""
Make an instance of the CCSD class and call compute ground state.
The verbose argument shows iteration of tau and lambda amplitudes.
Other keyword arguments can be added to control the CCSD computation 
in more detail. The default root finder is set to DIIS extrapolation.
"""
ccsd = CoupledClusterSinglesDoubles(system, verbose=False)
ccsd.compute_ground_state()
print("ECCSD ={0}".format(ccsd.compute_energy()))

"""
Make an instance of the OACCD (this is the same as Rolfs NOCCD but we have used OACCD as the name) 
class and call compute ground state.
The verbose argument shows iteration of tau, lambda and kappa if true.
The default root finder is set to DIIS extrapolation.

num_vecs is the number of DIIS vectors 
tol, termination_tol and tol_factor are internal paramters for the precision of 
the iteration scheme for OACCD/NOCCD. The names are horrible (we know).
"""

oaccd = OACCD(system, verbose=False)
oaccd.compute_ground_state(
    max_iterations=100,
    num_vecs=10,
    tol=1e-10,
    termination_tol=1e-12,
    tol_factor=1e-1,
)
print("EOACCD={0}".format(oaccd.compute_energy())) 