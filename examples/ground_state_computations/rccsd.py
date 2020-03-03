import numpy as np
from quantum_systems import construct_pyscf_system_rhf

molecule = "li 0.0 0.0 0.0;h 0.0 0.0 3.08"
# molecule = "he 0.0 0.0 0.0"
basis = "cc-pvdz"

from coupled_cluster.mix import DIIS, AlphaMixer
from coupled_cluster.rccsd import RCCSD

system = construct_pyscf_system_rhf(
    molecule,
    basis=basis,
    np=np,
    verbose=False,
    add_spin=False,
    anti_symmetrize=False,
)

rccsd = RCCSD(system, mixer=DIIS, verbose=True)

conv_tol = 1e-10
t_kwargs = dict(tol=conv_tol)
l_kwargs = dict(tol=conv_tol)

rccsd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)
print("Ground state correlation energy: {0}".format(rccsd.compute_energy()))
dm1 = rccsd.compute_one_body_density_matrix()
for i in range(3):
    dip_xi = np.trace(np.dot(dm1, system.dipole_moment[i]))
    print(f"dipole moment x{i}={dip_xi}")

t1, t2 = rccsd.t_1, rccsd.t_2
l1, l2 = rccsd.l_2, rccsd.l_2

import matplotlib.pyplot as plt

t1, t2 = t1.ravel(), t2.ravel()
l1, l2 = l1.ravel(), l2.ravel()

plt.figure()
plt.subplot(211)
plt.plot(np.arange(len(t1)), np.abs(t1) ** 2, "o", label=r"$|\tau_1|^2$")
plt.legend()
plt.subplot(212)
plt.plot(np.arange(len(t2)), np.abs(t2) ** 2, "o", label=r"$|\tau_2|^2$")
plt.legend()

plt.figure()
plt.subplot(211)
plt.plot(np.arange(len(l1)), np.abs(l1) ** 2, "o", label=r"$|\lambda_1|^2$")
plt.legend()
plt.subplot(212)
plt.plot(np.arange(len(l2)), np.abs(l2) ** 2, "o", label=r"$|\lambda_2|^2$")
plt.legend()

plt.show()
