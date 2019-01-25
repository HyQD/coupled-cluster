from quantum_systems import construct_psi4_system
from coupled_cluster.ccd import CoupledClusterDoubles

He = """
He 0.0 0.0 0.0
symmetry c1
"""

options = {"basis": "cc-pvtz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(He, options)

ccd = CoupledClusterDoubles(system, verbose=True)
ccd.iterate_t_amplitudes()
print("Ground state energy: {0}".format(ccd.compute_energy()))
