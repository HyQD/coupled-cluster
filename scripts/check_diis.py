import numpy as np

from quantum_systems import ODQD
from coupled_cluster.ccd import OACCD
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles
from configuration_interaction import CISD


odqd = ODQD(2, 20, 20, 201)
odqd.setup_system()

cisd = CISD(odqd, verbose=True)
cisd.compute_ground_state()

oaccd = OACCD(odqd, verbose=True)
oaccd.compute_ground_state()

ccsd = CoupledClusterSinglesDoubles(odqd, verbose=True)
ccsd.compute_ground_state()

print(f"cisd energy {cisd.energies[0]}")
print(f"oaccd energy {oaccd.compute_energy()}")
print(f"ccsd energy {ccsd.compute_energy()}")
