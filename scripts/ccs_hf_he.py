import numpy as np

from quantum_systems import construct_pyscf_system_rhf, ODQD
from coupled_cluster import CCS
from configuration_interaction import CIS


system = construct_pyscf_system_rhf("he")

cis = CIS(system, verbose=True)
cis.compute_ground_state()
print(f"CIS-HF ODHO energy: {cis.compute_energy()}")

ccs = CCS(system, verbose=True)
ccs.compute_ground_state()
print(f"CCS-HF Helium energy: {ccs.compute_energy()}")


system = ODQD(2, 12, 11, 201)
system.setup_system(potential=ODQD.HOPotential(omega=1))

system_c = system.copy_system()
system_c.change_to_hf_basis(verbose=True)

print(system_c.construct_fock_matrix(system_c.h, system_c.u))

cis = CIS(system, verbose=True)
cis.compute_ground_state()
print(f"CIS ODHO energy: {cis.compute_energy()}")

cis = CIS(system_c, verbose=True)
cis.compute_ground_state()
print(f"CIS-HF ODHO energy: {cis.compute_energy()}")

ccs = CCS(system, verbose=True)
ccs.compute_ground_state()
print(f"CCS ODHO energy: {ccs.compute_energy()}")

ccs = CCS(system_c, verbose=True)
ccs.compute_ground_state()
print(f"CCS-HF ODHO energy: {ccs.compute_energy()}")
