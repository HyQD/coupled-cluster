import numpy as np

from coupled_cluster.ccd.oaccd import OACCD
from coupled_cluster.mix import DIIS
from quantum_systems import construct_psi4_system
from tdhf import HartreeFock


He = """
He 0.0 0.0 0.0
symmetry c1
"""
options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(He, options)

hf = HartreeFock(system, verbose=True)
C = hf.scf(tolerance=1e-10)
system.change_basis(C)

oaccd = OACCD(system, mixer=DIIS, verbose=True)
oaccd.compute_ground_state(
    max_iterations=100,
    num_vecs=10,
    tol=1e-10,
    termination_tol=1e-12,
    tol_factor=1e-1,
)

print("Ground state energy: {0}".format(oaccd.compute_energy()))
