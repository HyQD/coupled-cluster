import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from pyscf import lib
from rccsd_functions import corr_energy, t1_rhs, t2_rhs
from rccsd_lambda_functions import l1_rhs, l2_rhs
from rccsd_density_matrices import one_body_density_matrix

# molecule = "ne 0.0 0.0 0.0"
molecule = "li 0.0 0.0 0.0;h 0.0 0.0 3.08"
# molecule = "ne 0.0 0.0 0.0"
basis = "cc-pvdz"

system = construct_pyscf_system_rhf(
    molecule,
    basis=basis,
    np=np,
    verbose=False,
    add_spin=False,
    anti_symmetrize=False,
)

nocc = system.n // 2
nvirt = system.l // 2 - system.n // 2

occ = slice(0, system.n // 2)
virt = slice(system.n // 2, system.l // 2)

H = system.h.real
W = system.u.real
F = (
    system.h
    + 2 * np.einsum("piqi->pq", W[:, occ, :, occ])
    - np.einsum("piiq->pq", W[:, occ, occ, :])
)

F_zero = F.copy()
np.fill_diagonal(F_zero, 0)


mo_e_o = np.diagonal(F[occ, occ])
mo_e_v = np.diagonal(F[virt, virt])

Dai = (mo_e_o[:, None] - mo_e_v).transpose()
Dabij = np.zeros(
    (nvirt, nvirt, nocc, nocc)
)  # lib.direct_sum('ai,bj->abij',Dai,Dai)


Nabij = np.zeros((nvirt, nvirt, nocc, nocc))
for i in range(0, nocc):
    for j in range(0, nocc):
        for a in range(nocc, nocc + nvirt):
            for b in range(nocc, nocc + nvirt):
                Nabij[a - nocc, b - nocc, i, j] = 1 + (a == b) * (i == j)
                Dabij[a - nocc, b - nocc, i, j] = (
                    F[i, i] + F[j, j] - F[a, a] - F[b, b]
                )


t1 = np.zeros((nvirt, nocc))
l1 = np.zeros((nocc, nvirt))

t2 = np.zeros((nvirt, nvirt, nocc, nocc))
l2 = np.zeros((nocc, nocc, nvirt, nvirt))
t2 = t2_rhs(F_zero, W, t1, t2) / Dabij
l2 = 0.5 * l2_rhs(F_zero, W, l1, l2, t1, t2) / Dabij.transpose(2, 3, 0, 1)

e_mp2 = corr_energy(F, W, t1, t2)
e_ccsd = e_mp2
print(f"e_mp2={e_mp2}")


residual_t1 = t1_rhs(F, W, t1, t2)
norm_residual_t1 = np.linalg.norm(residual_t1)

residual_l1 = l1_rhs(F, W, l1, l2, t1, t2)
norm_residual_l1 = np.linalg.norm(residual_l1)

residual_t2 = t2_rhs(F, W, t1, t2)
norm_residual_t2 = np.linalg.norm(residual_t2)

residual_l2 = l2_rhs(F, W, l1, l2, t1, t2)
norm_residual_l2 = np.linalg.norm(residual_l2)

conv_tol = 1e-6
iters = 1
max_iters = 100

while max(norm_residual_t1, norm_residual_t2) > conv_tol and (
    iters < max_iters
):

    t1 = t1_rhs(F_zero, W, t1, t2) / Dai
    t2 = t2_rhs(F_zero, W, t1, t2) / Dabij

    residual_t1 = t1_rhs(F, W, t1, t2)
    norm_residual_t1 = np.linalg.norm(residual_t1)

    residual_t2 = t2_rhs(F, W, t1, t2)
    norm_residual_t2 = np.linalg.norm(residual_t2)

    e_ccsd = corr_energy(F, W, t1, t2)
    print(f"corr_energy={e_ccsd}")
    print(f"||r_t1||={norm_residual_t1}, ||r_t2||={norm_residual_t2}")
    iters += 1
print(f"n_iters={iters}")


iters = 1
max_iters = 100

while max(norm_residual_l1, norm_residual_l2) > conv_tol and (
    iters < max_iters
):

    l1 = l1_rhs(F_zero, W, l1, l2, t1, t2) / Dai.T
    l2 = 0.5 * l2_rhs(F_zero, W, l1, l2, t1, t2) / Dabij.transpose(2, 3, 0, 1)

    residual_l1 = l1_rhs(F, W, l1, l2, t1, t2)
    norm_residual_l1 = np.linalg.norm(residual_l1)

    residual_l2 = l2_rhs(F, W, l1, l2, t1, t2)
    norm_residual_l2 = np.linalg.norm(residual_l2)

    print(f"||r_l1||={norm_residual_l1}, ||r_l2||={norm_residual_l2}")
    iters += 1
print(f"n_iters={iters}")

rho = one_body_density_matrix(l1, l2, t1, t2)
print(f"tr(rho)={np.trace(rho)}")
for i in range(3):
    dip_xi = np.trace(np.dot(rho, system.dipole_moment[i]))
    print(f"dipole moment x{i}={dip_xi}")


from coupled_cluster.mix import DIIS, AlphaMixer
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles

system = construct_pyscf_system_rhf(
    molecule,
    basis=basis,
    np=np,
    verbose=False,
    add_spin=True,
    anti_symmetrize=True,
)
ccsd = CoupledClusterSinglesDoubles(system, mixer=AlphaMixer, verbose=False)
t_kwargs = dict(tol=conv_tol, theta=0)
l_kwargs = dict(tol=conv_tol, theta=0)

ccsd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)
print("Ground state energy: {0}".format(ccsd.compute_energy()))


dm1 = ccsd.compute_one_body_density_matrix()
for i in range(3):
    dip_xi = np.trace(np.dot(dm1, system.dipole_moment[i]))
    print(f"dipole moment x{i}={dip_xi}")
