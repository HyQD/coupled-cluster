import numpy as np
from scipy.optimize import minimize

from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster import CCSD


system = construct_pyscf_system_rhf("he")

F = system.construct_fock_matrix(system.h, system.u).real
W = system.u.real
o = system.o
v = system.v
nocc = system.n
nvirt = system.m

assert np.linalg.norm(F.imag) < 1e-12
assert np.linalg.norm(W.imag) < 1e-12

print("nocc = ", nocc)
print("nvirt = ", nvirt)
nn = nocc * nvirt

epsilon = np.diag(F)
F = np.diag(epsilon)

print("epsilon = ", epsilon)


def vec_to_amp(x):
    T1 = np.reshape(x[0:nn], (nocc, nvirt))
    T2 = np.reshape(x[nn : (nn + nn * nn)], (nocc, nocc, nvirt, nvirt))
    L1 = np.reshape(x[(nn + nn * nn) : (2 * nn + nn * nn)], (nocc, nvirt))
    L2 = np.reshape(
        x[(2 * nn + nn * nn) : (2 * nn + 2 * nn * nn)],
        (nocc, nocc, nvirt, nvirt),
    )
    return (T1, T2, L1, L2)


def amp_to_vec(T1, T2, L1, L2):

    y = np.zeros(2 * nn * (1 + nn))

    y[0:nn] = np.reshape(T1, (nn,))
    y[nn : (nn + nn * nn)] = np.reshape(T2, (nn * nn,))
    y[(nn + nn * nn) : (2 * nn + nn * nn)] = np.reshape(L1, (nn,))
    y[(2 * nn + nn * nn) : (2 * nn + 2 * nn * nn)] = np.reshape(L2, (nn * nn,))

    return y


from coupled_cluster.ccsd.energies import compute_time_dependent_energy

Lagrangian_fun = lambda L1, F, T1, L2, W, T2: compute_time_dependent_energy(
    F, W, T1.T, T2.transpose(2, 3, 0, 1), L1, L2, o, v, np
)

from coupled_cluster.ccsd.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)
from coupled_cluster.ccsd.rhs_l import (
    compute_l_1_amplitudes,
    compute_l_2_amplitudes,
)

Omega1_fun = lambda T1, F, W, T2: compute_t_1_amplitudes(
    F, W, T1.T, T2.transpose(2, 3, 0, 1), o, v, np
).T
Omega2_fun = lambda W, T2, T1, F: compute_t_2_amplitudes(
    F, W, T1.T, T2.transpose(2, 3, 0, 1), o, v, np
).transpose(2, 3, 0, 1)

tOmega1_fun = lambda L1, F, W, T1, L2, T2: compute_l_1_amplitudes(
    F, W, T1.T, T2.transpose(2, 3, 0, 1), L1, L2, o, v, np
)
tOmega2_fun = lambda W, L2, T1, L1, F, T2: compute_l_2_amplitudes(
    F, W, T1.T, T2.transpose(2, 3, 0, 1), L1, L2, o, v, np
)


def f(x):

    (T1, T2, L1, L2) = vec_to_amp(x)
    e = Lagrangian_fun(L1, F, T1, L2, W, T2)
    print("L = ", e)
    return e


def fprime(x):

    (T1, T2, L1, L2) = vec_to_amp(x)

    Omega1 = Omega1_fun(T1, F, W, T2)
    Omega2 = Omega2_fun(W, T2, T1, F)
    tOmega1 = tOmega1_fun(L1, F, W, T1, L2, T2)
    tOmega2 = tOmega2_fun(W, L2, T1, L1, F, T2)

    return amp_to_vec(Omega1, Omega2, tOmega1, tOmega2)


T1 = np.zeros((nocc, nvirt))
T2 = np.zeros((nocc, nocc, nvirt, nvirt))
L1 = np.zeros((nocc, nvirt))
L2 = np.zeros((nocc, nocc, nvirt, nvirt))


x0 = np.zeros(2 * nn * (nn + 1))

res = minimize(
    f, x0, method="BFGS", jac=fprime, options={"gtol": 1e-6, "disp": True}
)

T1 = res.x[0:nn]
T2 = res.x[nn : (nn + nn * nn)]
L1 = res.x[(nn + nn * nn) : (2 * nn + nn * nn)]
L2 = res.x[(2 * nn + nn * nn) : (2 * nn + 2 * nn * nn)]

from coupled_cluster.ccsd.energies import compute_ccsd_ground_state_energy

T1 = T1.reshape(nocc, nvirt)
T2 = T2.reshape(nocc, nocc, nvirt, nvirt)
L1 = L1.reshape(nocc, nvirt)
L2 = L2.reshape(nocc, nocc, nvirt, nvirt)

ccsd_energy_fun = lambda T1, F, T2, W: compute_ccsd_ground_state_energy(
    F, W, T1.T, T2.transpose(2, 3, 0, 1), o, v, np
)

E_CC = ccsd_energy_fun(
    T1, F, T2, W
)  # Warwning! Order of arguments not predictable from code generator a.t.m.

print("Final CCSD correlation energy: %.16g" % E_CC)

ccsd = CCSD(system, verbose=True).compute_ground_state()
