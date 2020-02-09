import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from pyscf import lib
from rccd_functions import corr_energy, t2_rhs
from rccd_lambda_functions import l2_rhs
from rccd_density_matrices import one_body_density_matrix

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

# print(Nabij)

t1 = np.zeros((nvirt, nocc))
t2 = np.zeros((nvirt, nvirt, nocc, nocc))
t2 = t2_rhs(F_zero, W, t2) / Dabij
l2 = np.zeros((nocc, nocc, nvirt, nvirt))

"""
The rhs of lambda carries a factor 2 in the terms where f is contracted with lambda
"""
l2 = 0.5 * l2_rhs(F_zero, W, l2, t2) / Dabij.transpose(2, 3, 0, 1)


EMP2 = corr_energy(t2, W)
Eccd = EMP2
residual_t2 = t2_rhs(F, W, t2)
norm_residual_t2 = np.linalg.norm(residual_t2)
residual_l2 = l2_rhs(F, W, l2, t2)
norm_residual_l2 = np.linalg.norm(residual_l2)

conv_tol = 1e-10

max_iters = 100
iters = 1

while (np.abs(norm_residual_t2) > conv_tol) and (iters < max_iters):

    t2 = t2_rhs(F_zero, W, t2) / Dabij
    l2 = 0.5 * l2_rhs(F_zero, W, l2, t2) / Dabij.transpose(2, 3, 0, 1)

    Eccd = corr_energy(t2, W)
    residual_t2 = t2_rhs(F, W, t2)
    norm_residual_t2 = np.linalg.norm(residual_t2)
    residual_l2 = l2_rhs(F, W, l2, t2)
    norm_residual_l2 = np.linalg.norm(residual_l2)
    iters += 1
    print(f"||res_t2||={norm_residual_t2}, ||res_l2||={norm_residual_l2}")

print(f"n_iters={iters}")
print(f"EMP2={EMP2}, iter=init")
print(f"ECCD={Eccd}")

rho = one_body_density_matrix(l2, t2)
print(f"tr(rho)={np.trace(rho)}")

for i in range(3):
    dip_xi = np.trace(np.dot(rho, system.dipole_moment[i]))
    print(f"dipole moment x{i}={dip_xi}")


from coupled_cluster.mix import DIIS, AlphaMixer
from coupled_cluster.ccd import CoupledClusterDoubles

system = construct_pyscf_system_rhf(
    molecule,
    basis=basis,
    np=np,
    verbose=False,
    add_spin=True,
    anti_symmetrize=True,
)
ccd = CoupledClusterDoubles(system, mixer=AlphaMixer, verbose=False)
t_kwargs = dict(tol=conv_tol, theta=0)
l_kwargs = dict(tol=conv_tol, theta=0)

ccd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)
print("Ground state energy: {0}".format(ccd.compute_energy()))
dm1 = ccd.compute_one_body_density_matrix()
dip_mom_z = np.trace(np.dot(dm1, system.dipole_moment[2]))
print(f"dip_mom_z={dip_mom_z}")
