import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
from scipy.integrate import simps, trapz
from scipy.linalg import expm


def ddx_psi(y, dx):
    N = len(y)
    ddx_y = np.zeros(N, dtype=np.complex128)
    ddx_y[1 : N - 1] = (y[0 : N - 2] - 2 * y[1 : N - 1] + y[2:]) / dx ** 2
    return ddx_y


def f(t, y, X, Omega, w, E0=1):
    dx = X[1] - X[0]
    rhs = (
        -0.5 * ddx_psi(y, dx)
        + 0.5 * Omega ** 2 * X ** 2 * y
        + 4 * E0 * X * np.sin(w * t) * y
    )
    return 0.5 * rhs


N = 400
rmax = 10
Omega = (
    0.25
)  # Oscillator frequency, naming convention consistent with Schwengelbeck/Zanghellini.
a = 0.25


# Solve ground state equation for the relative coordnate r = r2-r1
r = np.linspace(-rmax, rmax, N)
dr = r[1] - r[0]
wr = 0.5 * Omega
H = np.zeros((N, N))  # Solving for internal grid points only
for i in range(0, N):
    H[i, i] = (
        1.0 / (dr ** 2)
        + 0.5 * wr ** 2 * r[i] ** 2
        + 0.5 * 1.0 / np.sqrt(r[i] ** 2 + a ** 2)
    )
    if i + 1 < N - 1:
        H[i + 1, i] = -1.0 / (2.0 * dr ** 2)
        H[i, i + 1] = -1.0 / (2.0 * dr ** 2)
epsilon, phi = scipy.linalg.eigh(H)
phi = phi / np.sqrt(dr)

# Setup center of mass intitial condition
R = np.linspace(-rmax, rmax, N)
wR = 2 * Omega
psiR = (wR / np.pi) ** 0.25 * np.exp(-0.5 * wR * R ** 2)
epsR = wR * 0.5

plt.figure(0)
plt.plot(R,np.abs(psiR)**2)


# Time parameters
Psi = psiR.copy()
T = 14
dt = 1e-3
counter = 0
time_steps = int(T / dt)  # Number of time steps
overlap = [simps(np.conj(Psi) * Psi, r)]
t_list = [0]
Energy = []
w = 8 * Omega

print(0.5 * epsR + 2 * epsilon[0])

Energy.append(0.5 * epsR)
Psi = np.complex128(Psi)

while counter < time_steps:

    counter += 1
    tn = counter * dt
    k1 = -dt * 1j * f(tn, Psi, R, wR, w)
    k2 = -dt * 1j * f(tn + 0.5 * dt, Psi + 0.5 * k1, R, wR, w)
    k3 = -dt * 1j * f(tn + 0.5 * dt, Psi + 0.5 * k2, R, wR, w)
    k4 = -dt * 1j * f(tn + dt, Psi + k3, R, wR, w)

    Psi = Psi + 1.0 / 6.0 * (k1 + 2 * (k2 + k3) + k4)
    Ht_psi = f(tn, Psi, R, wR, w)
    Energy.append(simps(np.conj(Psi) * Ht_psi, r).real)

    overlap.append(np.abs(simps(np.conj(Psi) * psiR, r)) ** 2)
    t_list.append(tn)
    
    if(counter%3000 == 0):
        plt.plot(R,np.abs(Psi)**2)
        print(trapz(Psi.conj()*Psi,R),tn)
    
plt.show()

plt.figure(1)
plt.plot(t_list, Energy + 2 * epsilon[0])

plt.figure(2)
plt.plot(t_list, overlap)
plt.show()
