import numpy as np
import scipy.optimize

from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster import CCS

system = construct_pyscf_system_rhf("he")

complex_to_real = lambda x: x.view(np.float64)
real_to_complex = lambda x: x.view(np.complex128)

ccs = CCS(system, verbose=True)
y_0 = ccs.get_initial_guess().asarray()

res = scipy.optimize.minimize(
    lambda x: ccs.compute_energy(real_to_complex(x)).real,
    complex_to_real(y_0),
    method="BFGS",
    jac=lambda x: complex_to_real(ccs(real_to_complex(x))),
    options=dict(gtol=1e-6, disp=True),
)

y = res.x.view(np.complex128)

print(f"Final CCS energy: {ccs.compute_energy(y)}")

# Explicit computation of the energy
energy = (
    ccs.compute_one_body_expectation_value(y, system.h)
    + 0.5 * ccs.compute_two_body_expectation_value(y, system.u)
    + system.nuclear_repulsion_energy
)

print(f"Manual CCS energy: {energy}")
