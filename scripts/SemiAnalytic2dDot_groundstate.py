import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
from scipy.integrate import simps, trapz
from scipy.linalg import expm

"""
Exact ground state energies from Taut:

wr = 1/2 * w => w = 2*wr 

Exact eps_r

wr = 0.5 => w = 1
eps_r = 1

wr = 1/12 => w = 1/6
eps_r = 3/12 = 1/4

"""


Nr = 1000  # Number of internal grid points
rmax = 15
Omega = (
    1.0 / 6.0
)  # Oscillator frequency, naming convention consistent with Schwengelbeck/Zanghellini.

# Solve ground state equation for the relative coordnate r = r2-r1
r = np.linspace(0, rmax, Nr + 2)
dr = rmax / float(Nr + 1)

wr = 0.5 * Omega
H = np.zeros((Nr + 2, Nr + 2))  # Include boundary points for convenience

for i in range(1, Nr + 1):
    H[i, i] = (
        1.0 / (dr ** 2)
        + 0.5 * wr ** 2 * r[i] ** 2
        + (4 * r[i] - 1) / (8.0 * r[i] ** 2)
    )  # 1.0/(2*r[i]) - 1.0/(8.0*r[i]**2)
    if i + 1 < Nr + 1:
        H[i + 1, i] = -1.0 / (2.0 * dr ** 2)
        H[i, i + 1] = -1.0 / (2.0 * dr ** 2)


epsilon, phi = np.linalg.eigh(H)
phi = phi / np.sqrt(dr)

# Note that with the current setup the ground state is found at in the second column
u_r = phi[:, 2]
plt.plot(r, np.abs(u_r) ** 2)
plt.show()

wR = 2 * Omega
eps_X = 0.5 * wR
eps_Y = 0.5 * wR
eps_R = eps_X + eps_Y

eps_r = epsilon[2]


print("eps_r: %g" % eps_r)
print("eps_R: %g" % eps_R)
print("E0: %g" % (2 * eps_r + 0.5 * eps_R))
