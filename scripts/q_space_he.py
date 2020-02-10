import tqdm
import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster import OATDCCD, CCD
from coupled_cluster.sampler import OATDCCSampleAll
from coupled_cluster.integrators import GaussIntegrator


system = construct_pyscf_system_rhf("he")

ccd = CCD(system, verbose=True)
ccd.compute_ground_state()

integrator = GaussIntegrator(s=3, eps=1e-8, np=np)
oa = OATDCCD(system, verbose=False, integrator=integrator)

# Run with CCD-HF groundstate with HF coefficients
oa.compute_ground_state(change_system_basis=False)
oa.set_initial_conditions(amplitudes=ccd.get_amplitudes(get_t_0=True))

# Run with NOCCD groundstate
# oa.compute_ground_state(change_system_basis=True)
# oa.set_initial_conditions()


dt = 1e-2
time_points = np.arange(0, 10 + dt, dt)

sampler = OATDCCSampleAll(oa, len(time_points), np)
sampler.add_sample("time_points", time_points)
sampler.sample(0)

# energy = np.zeros(len(time_points), dtype=np.complex128)
# energy[0] = oa.compute_energy()

for i, amp in tqdm.tqdm(
    enumerate(oa.solve(time_points)), total=len(time_points) - 1
):
    sampler.sample(i + 1)


# sampler.dump(filename="yolo.npy")

samples = sampler.dump(save_samples=False)
print(list(samples))

plt.figure()
plt.plot(samples["time_points"][1:], samples["energy"][1:].real)
plt.title("Real")

plt.figure()
plt.plot(samples["time_points"][1:], samples["energy"][1:].imag)
plt.title("Imag")

plt.show()
