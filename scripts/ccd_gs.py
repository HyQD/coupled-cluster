import numpy as np
import scipy.optimize

from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster import CCD


system = construct_pyscf_system_rhf("he")

complex_to_real = lambda x: np.concatenate((x.real, x.imag))
real_to_complex = lambda x: x[: len(x) // 2] + 1j * x[len(x) // 2 :]

ccd = CCD(system, verbose=True)
y_0 = ccd.get_initial_guess().asarray()

res = scipy.optimize.minimize(
    lambda x: ccd.compute_energy(real_to_complex(x)).real,
    complex_to_real(y_0),
    method="BFGS",
    jac=lambda x: complex_to_real(ccd(real_to_complex(x))),
    options=dict(gtol=1e-6, disp=True),
)