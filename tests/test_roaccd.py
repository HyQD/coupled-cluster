import pytest

from quantum_systems import construct_pyscf_system_rhf

from coupled_cluster.ccd.oaccd import OACCD
from coupled_cluster.rccd.roaccd import ROACCD
from coupled_cluster.mix import DIIS


def test_kappa_derivatives():

    """
    Consistency test that kappa derivatives using only non-zero two-body
    density matrix elements (compute_R_ia, compute_R_tilde_ai)
    gives the same output as using ":"-slices (compute_R_ia_compact, compute_R_tilde_ai_compact).
    """

    import numpy as np
    from coupled_cluster.rccd.p_space_equations import (
        compute_R_ia,
        compute_R_ia_compact,
        compute_R_tilde_ai,
        compute_R_tilde_ai_compact,
    )

    symm = lambda doubles: 0.5 * (
        doubles + np.swapaxes(np.swapaxes(doubles, 0, 1), 2, 3)
    )

    N = 10
    L = 80
    M = L - N

    o = slice(0, N)
    v = slice(N, L)

    h = np.random.random((L, L))
    u = symm(np.random.random((L, L, L, L)))

    rho_qp = np.zeros((L, L))
    rho_qspr = np.zeros((L, L, L, L))

    rho_qspr[o, o, o, o] = np.random.random((N, N, N, N))
    rho_qspr[v, v, o, o] = np.random.random((M, M, N, N))

    rho_qspr[o, v, o, v] = np.random.random((N, M, N, M))
    rho_qspr[v, o, v, o] = (
        rho_qspr[o, v, o, v].swapaxes(1, 0).swapaxes(3, 2).copy()
    )

    rho_qspr[v, o, o, v] = np.random.random((M, N, N, M))
    rho_qspr[o, v, v, o] = (
        rho_qspr[v, o, o, v].swapaxes(1, 0).swapaxes(3, 2).copy()
    )

    rho_qspr[o, o, v, v] = np.random.random((N, N, M, M))
    rho_qspr[v, v, v, v] = np.random.random((M, M, M, M))

    rho_qspr = symm(rho_qspr)

    tmp = compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np)
    tmp2 = compute_R_ia_compact(h, u, rho_qp, rho_qspr, o, v, np)

    tmp3 = compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np)
    tmp4 = compute_R_tilde_ai_compact(h, u, rho_qp, rho_qspr, o, v, np)

    assert np.allclose(tmp, tmp2)
    assert np.allclose(tmp3, tmp4)


def test_roaccd_vs_oaccd():
    import numpy as np

    molecule = "b 0.0 0.0 0.0;h 0.0 0.0 2.4"
    basis = "cc-pvdz"

    system = construct_pyscf_system_rhf(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )

    conv_tol = 1e-10
    roaccd = ROACCD(system, mixer=DIIS, verbose=True)
    roaccd.compute_ground_state(
        max_iterations=100,
        num_vecs=10,
        tol=conv_tol,
        termination_tol=conv_tol,
        tol_factor=1e-1,
    )

    e_roaccd = roaccd.compute_energy()

    system = construct_pyscf_system_rhf(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=True,
        anti_symmetrize=True,
    )

    oaccd = OACCD(system, mixer=DIIS, verbose=True)
    oaccd.compute_ground_state(
        max_iterations=100,
        num_vecs=10,
        tol=conv_tol,
        termination_tol=conv_tol,
        tol_factor=1e-1,
    )

    e_oaccd = oaccd.compute_energy()
    assert abs(e_oaccd - e_roaccd) < conv_tol


if __name__ == "__main__":
    # test_roaccd_vs_oaccd()
    test_kappa_derivatives()
