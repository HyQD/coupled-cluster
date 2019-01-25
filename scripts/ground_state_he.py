from quantum_systems import construct_psi4_system
from coupled_cluster.ccd import CoupledClusterDoubles
from tdhf import HartreeFock
import numpy as np

He = """
He 0.0 0.0 0.0
symmetry c1
"""

options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(He, options)

# Compute the Hartree-Fock state
hf = HartreeFock(system,verbose=True)
C = hf.scf(max_iters=30, tolerance=1e-15)

system._h = np.einsum("ap,bq,ab->pq", C.conj(), C, system.h, optimize=True)
system._u = np.einsum(
    "ap,bq,gr,ds,abgd->pqrs",
    C.conj(),
    C.conj(),
    C,
    C,
    system.u,
    optimize=True,
)

ccd = CoupledClusterDoubles(system, verbose=True)
ccd.iterate_t_amplitudes()
print("Ground state energy: {0}".format((ccd.compute_energy()-hf.e_hf).real))

del system
del ccd 

CO = """
C 0.0 0.0 -1.079696382067556
O 0.0 0.0  0.810029743390272
symmetry c1
no_reorient
no_com
units bohr
"""

options = {"basis": "STO-3G", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(CO, options)

# Compute the Hartree-Fock state
hf = HartreeFock(system,verbose=True)
C = hf.scf(max_iters=30, tolerance=1e-15)

system._h = np.einsum("ap,bq,ab->pq", C.conj(), C, system.h, optimize=True)
system._u = np.einsum(
    "ap,bq,gr,ds,abgd->pqrs",
    C.conj(),
    C.conj(),
    C,
    C,
    system.u,
    optimize=True,
)

ccd = CoupledClusterDoubles(system, verbose=True)
ccd.iterate_t_amplitudes()
print("Ground state energy: {0}".format((ccd.compute_energy()-hf.e_hf).real))

del system
del ccd
