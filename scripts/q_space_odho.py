import tqdm
import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import ODQD
from coupled_cluster import OATDCCD, CCD
from hartree_fock import GHF


n = 2
l = 12

odho = ODQD(n, l, 11, 201)
odho.setup_system(potential=ODQD.HOPotential(1))
odho_c = odho.copy_system()
hf = GHF(odho, verbose=True)
hf.compute_ground_state(change_system_basis=True)
print(hf.C)

ccd = CCD(odho, verbose=True)
ccd.compute_ground_state()

foo = hf.C.conj().T @ odho.s @ hf.C
foo[np.abs(foo) < 1e-12] = 0
print(foo)
print(np.diag(foo))

oa = OATDCCD(odho_c, verbose=True)
oa.compute_ground_state(change_system_basis=False)
oa.set_initial_conditions(
    amplitudes=ccd.get_amplitudes(get_t_0=True), C=hf.C, C_tilde=hf.C.T.conj()
)


dt = 1e-2
time_points = np.arange(0, 5 + dt, dt)

energy = np.zeros(len(time_points), dtype=np.complex128)
energy[0] = oa.compute_energy()

for i, amp in tqdm.tqdm(
    enumerate(oa.solve(time_points)), total=len(time_points) - 1
):
    energy[i + 1] = oa.compute_energy()

plt.figure()
plt.plot(time_points[1:], energy[1:].real)
plt.title("Real")

plt.figure()
plt.plot(time_points[1:], energy[1:].imag)
plt.title("Imag")

plt.show()
