import numpy as np
import matplotlib.pyplot as plt
import time
from quantum_systems import construct_psi4_system
from quantum_systems.time_evolution_operators import LaserField
from tdhf import HartreeFock
from coupled_cluster.ccd import OATDCCD, CoupledClusterDoubles


class laser_pulse:
    def __init__(self, t0=0, td=5, omega=0.1, E=0.03):
        self.t0 = t0
        self.td = td
        self.omega = omega
        self.E = E  # Field strength

    def __call__(self, t):
        T = self.td
        delta_t = t - self.t0
        return (
            -(np.sin(np.pi * delta_t / T) ** 2)
            * np.heaviside(delta_t, 1.0)
            * np.heaviside(T - delta_t, 1.0)
            * np.cos(self.omega * delta_t)
            * self.E
        )


# Define system paramters
He = """
He 0.0 0.0 0.0
symmetry c1
"""


options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-8}
omega = 2.873_564_3
E = 100  # 0.05-5


system = construct_psi4_system(He, options)
hf = HartreeFock(system, verbose=True)
C = hf.scf(tolerance=1e-15)
system.change_basis(C)

cc_kwargs = dict(verbose=True)
oatdccd = OATDCCD(CoupledClusterDoubles, system, np=np, **cc_kwargs)
t_kwargs = dict(theta=0.1, tol=1e-10)
oatdccd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=t_kwargs)
print(
    "Ground state CCD energy: {0}".format(oatdccd.compute_ground_state_energy())
)

polarization = np.zeros(3)
polarization[2] = 1
system.set_polarization_vector(polarization)
system.set_time_evolution_operator(LaserField(laser_pulse(omega=omega, E=E)))

oatdccd.set_initial_conditions()
time_points = np.linspace(0, 3, 3001)
dt = time_points[1] - time_points[0]
print("dt = {0}".format(dt))

td_energies = np.zeros(len(time_points))
td_energies_imag = np.zeros(len(time_points))

td_energies[0] = oatdccd.compute_energy()

for i, amp in enumerate(oatdccd.solve(time_points)):
    t, l, C, C_tilde = amp
    energy = oatdccd.compute_energy()
    td_energies[i+1] = energy.real
    td_energies_imag[i+1] = energy.imag
    if i % 100 == 0:
        print(f"i = {i}")
        eye = C_tilde @ C
        print(np.allclose(eye, np.eye(eye.shape[0])))
        print("norm(t2): %g" % np.linalg.norm(t))
        print("norm(l2): %g" % np.linalg.norm(l))
    # print(eye)
    # print(np.diag(eye))
    # np.testing.assert_allclose(C_tilde @ C, np.eye(C_tilde.shape[0]), atol=1e-10)
    # if i == 1:
    #    break

plt.figure()
plt.plot(time_points, td_energies)
plt.grid()

plt.figure()
plt.plot(time_points, td_energies_imag)
plt.grid()

plt.show()
