import os

import numpy as np
import pandas as pd

from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.ccd import OACCD
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles as CCSD


atoms = [
    # "he",
    # "be",
    # "ne",
    # "mg",
    "ar",
    # "kr",
]

basis = "aug-ccpvtz"

for atom in atoms:
    print(f"Atom: {atom}")
    system = construct_pyscf_system_rhf(atom, basis)
    ccsd = CCSD(system, verbose=False)
    ccsd.compute_ground_state()
    print(f"CCSD energy: {ccsd.compute_energy()}")
    # oaccd = OACCD(system, verbose=False)
    # oaccd.compute_ground_state()
    # print(f"OACCD energy: {oaccd.compute_energy()}")
