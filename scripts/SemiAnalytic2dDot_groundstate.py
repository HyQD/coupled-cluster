import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
from scipy.integrate import simps, trapz
from scipy.linalg import expm

N      = 801
rmax   = 10
Omega  = 1 #Oscillator frequency, naming convention consistent with Schwengelbeck/Zanghellini.

#Solve ground state equation for the relative coordnate r = r2-r1
r       = np.linspace(0.01, rmax, N)
dr      = r[1] - r[0]
wr      = 0.5*Omega
H       = np.zeros((N, N)) #Solving for internal grid points only

for i in range(0,N):
    H[i, i] = 1.0/(dr**2) + 0.5*wr**2*r[i]**2 + 1.0/(2*r[i]) - 1.0/(8.0*r[i]**2)
    if i + 1 < N - 1:
        H[i + 1, i] = -1.0/(2.0*dr**2)
        H[i, i + 1] = -1.0/(2.0*dr**2)

epsilon, phi = scipy.linalg.eigh(H)
phi = phi/np.sqrt(dr)

plt.plot(r,phi[:,0])
plt.show()

print(epsilon[0])

wR = 2*Omega
eps_X = 0.5*wR
eps_Y = 0.5*wR
eps_R = eps_X+eps_Y

print(2*epsilon[0]+0.5*eps_R)