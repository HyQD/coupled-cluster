import numpy as np
from .cc import CoupledCluster
from .cc_helper import (
    amplitude_scaling_one_body,
    amplitude_scaling_two_body,
    amplitude_scaling_one_body_lambda,
    amplitude_scaling_two_body_lambda,
)


class CoupledClusterSinglesDoubles(CoupledCluster):
    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

        o, v = self.o, self.v
        n, m = self.n, self.m

        self.rhs_1 = np.zeros((m, n), dtype=np.complex128)
        self.rhs_2 = np.zeros((m, m, n, n), dtype=np.complex128)

        self.rhs_1_lambda = np.zeros((n, m), dtype=np.complex128)
        self.rhs_2_lambda = np.zeros((n, n, m, m), dtype=np.complex128)

        self.t_1 = np.zeros((m, n), dtype=np.complex128)
        self.t_2 = np.zeros((m, m, n, n), dtype=np.complex128)

        self.l_1 = np.zeros((n, m), dtype=np.complex128)
        self.l_2 = np.zeros((n, n, m, m), dtype=np.complex128)

        self.xi = np.zeros((m, m, n, n), dtype=np.complex128)
        self.tau = np.zeros((m, m, n, n), dtype=np.complex128)

        self.G_pp = np.zeros((m, m), dtype=np.complex128)
        self.G_hh = np.zeros((n, n), dtype=np.complex128)

        # Intermediates for t amplitudes
        self.F_pp = np.zeros((m, m), dtype=np.complex128)
        self.F_hh = np.zeros((n, n), dtype=np.complex128)
        self.F_hp = np.zeros((n, m), dtype=np.complex128)

        self.W_hhhh = np.zeros((n, n, n, n), dtype=np.complex128)
        self.W_pppp = np.zeros((m, m, m, m), dtype=np.complex128)
        self.W_hpph = np.zeros((n, m, m, n), dtype=np.complex128)

        # Intermediates for lambda amplitudes
        self.F_pp_lambda = np.zeros((m, m), dtype=np.complex128)
        self.F_hh_lambda = np.zeros((n, n), dtype=np.complex128)
        self.F_hp_lambda = self.F_hp

        self.W_hhhh_lambda = np.zeros((n, n, n, n), dtype=np.complex128)
        self.W_pppp_lambda = np.zeros((m, m, m, m), dtype=np.complex128)
        self.W_hpph_lambda = np.zeros((n, m, m, n), dtype=np.complex128)
        self.W_hhhp_lambda = np.zeros((n, n, n, m), dtype=np.complex128)
        self.W_phpp_lambda = np.zeros((m, n, m, m), dtype=np.complex128)
        self.W_hphh_lambda = np.zeros((n, m, n, n), dtype=np.complex128)
        self.W_ppph_lambda = np.zeros((m, m, m, n), dtype=np.complex128)

        self._compute_initial_guess()

        self.rho_qp = np.zeros((self.l, self.l), dtype=np.complex128)

    def _get_t_copy(self):
        return [self.t_1.copy(), self.t_2.copy()]

    def _get_lambda_copy(self):
        return [self.l_1.copy(), self.l_2.copy()]

    def _set_t(self, t):
        t_1, t_2 = t

        np.copyto(self.t_1, t_1)
        np.copyto(self.t_2, t_2)

    def _set_l(self, l):
        l_1, l_2 = l

        np.copyto(self.l_1, l_1)
        np.copyto(self.l_2, l_2)

    def _compute_initial_guess(self):
        o, v = self.o, self.v

        np.copyto(self.rhs_1, self.f[v, o])
        np.copyto(self.rhs_2, self.u[v, v, o, o])

        amplitude_scaling_one_body(self.rhs_1, self.f, self.m, self.n)
        amplitude_scaling_two_body(self.rhs_2, self.f, self.m, self.n)

        np.copyto(self.t_1, self.rhs_1)
        np.copyto(self.t_2, self.rhs_2)

        np.copyto(self.rhs_1_lambda, self.f[o, v])
        np.copyto(self.rhs_2_lambda, self.u[o, o, v, v])

        amplitude_scaling_one_body_lambda(
            self.rhs_1_lambda, self.f, self.m, self.n
        )
        amplitude_scaling_two_body_lambda(
            self.rhs_2_lambda, self.f, self.m, self.n
        )

        np.copyto(self.l_1, self.rhs_1_lambda)
        np.copyto(self.l_2, self.rhs_2_lambda)

    def _compute_time_evolution_probability(self):
        t_1_0, t_2_0 = self._t_0
        l_1_0, l_2_0 = self._l_0

        psi_t_0 = 1
        psi_t_0 += np.einsum("ia, ai ->", self.l_1, t_1_0)
        psi_t_0 -= np.einsum("ia, ai ->", self.l_1, self.t_1)
        psi_t_0 += 0.25 * np.einsum("ijab, abij ->", self.l_2, t_2_0)
        psi_t_0 -= 0.5 * np.einsum(
            "ijab, aj, bi ->", self.l_2, t_1_0, t_1_0, optimize=True
        )
        psi_t_0 -= np.einsum(
            "ijab, ai, bj ->", self.l_2, self.t_1, t_1_0, optimize=True
        )
        psi_t_0 -= 0.5 * np.einsum(
            "ijab, aj, bi ->", self.l_2, self.t_1, self.t_1, optimize=True
        )
        psi_t_0 -= 0.25 * np.einsum("ijab, abij ->", self.l_2, self.t_2)

        psi_0_t = 1
        psi_0_t += np.einsum("ia, ai ->", l_1_0, self.t_1)
        psi_0_t -= np.einsum("ia, ai ->", l_1_0, t_1_0)
        psi_0_t += 0.25 * np.einsum("ijab, abij ->", l_2_0, self.t_2)
        psi_0_t -= 0.5 * np.einsum(
            "ijab, aj, bi ->", l_2_0, t_1_0, t_1_0, optimize=True
        )
        psi_0_t -= np.einsum("ijab, ai, bj ->", l_2_0, self.t_1, t_1_0)
        psi_0_t -= 0.5 * np.einsum(
            "ijab, aj, bi ->", l_2_0, self.t_1, self.t_1, optimize=True
        )
        psi_0_t -= 0.25 * np.einsum("ijab, abij ->", l_2_0, t_2_0)

        return psi_t_0 * psi_0_t

    def _compute_one_body_density_matrix(self):
        o, v = self.o, self.v

        self.rho_qp.fill(0)

        self.rho_qp[v, v] += np.dot(self.t_1, self.l_1)
        self.rho_qp[v, v] += 0.5 * np.tensordot(
            self.t_2, self.l_2, axes=((1, 2, 3), (3, 0, 1))
        )

        self.rho_qp[o, v] += self.l_1

        self.rho_qp[o, o] += np.eye(self.n)
        self.rho_qp[o, o] -= np.dot(self.l_1, self.t_1)
        self.rho_qp[o, o] += 0.5 * np.tensordot(
            self.l_2, self.t_2, axes=((1, 2, 3), (2, 0, 1))
        )

        self.rho_qp[v, o] += self.t_1
        self.rho_qp[v, o] += np.tensordot(
            self.l_1,
            self.t_2 - np.einsum("bi, aj -> abij", self.t_1, self.t_1),
            axes=((0, 1), (3, 1)),
        )
        self.rho_qp[v, o] += 0.5 * np.einsum(
            "bi, kjcb, ackj -> ai", self.t_1, self.l_2, self.t_2, optimize=True
        )
        self.rho_qp[v, o] -= 0.5 * np.einsum(
            "aj, kjcb, cbki -> ai", self.t_1, self.l_2, self.t_2, optimize=True
        )

        return self.rho_qp

    def _compute_energy(self):
        return self._compute_ccsd_energy()

    def _compute_ccsd_energy(self):
        o, v = self.o, self.v

        energy = np.einsum("ia, ai ->", self.f[o, v], self.t_1)
        energy += 0.25 * np.einsum(
            "ijab, abij ->", self.u[o, o, v, v], self.t_2
        )
        energy += 0.5 * np.einsum(
            "ijab, ai, bj ->", self.u[o, o, v, v], self.t_1, self.t_1
        )

        return energy + self.compute_reference_energy()

    def _compute_amplitudes(self, theta, iterative=True):
        self._compute_effective_amplitudes()
        self._compute_intermediates(iterative=iterative)
        self._compute_ccsd_amplitude_s(iterative=iterative)
        self._compute_ccsd_amplitude_d()

        if not iterative:
            return [self.rhs_1.copy(), self.rhs_2.copy()]

        amplitude_scaling_one_body(self.rhs_1, self.f, self.m, self.n)
        amplitude_scaling_two_body(self.rhs_2, self.f, self.m, self.n)

        np.add((1 - theta) * self.rhs_1, theta * self.t_1, out=self.t_1)
        np.add((1 - theta) * self.rhs_2, theta * self.t_2, out=self.t_2)

    def _compute_lambda_amplitudes(self, theta, iterative=True):
        self._compute_effective_three_body_intermediates()
        self._compute_lambda_intermediates()
        self._compute_ccsd_lambda_amplitudes_s()
        self._compute_ccsd_lambda_amplitudes_d()

        if not iterative:
            return [self.rhs_1_lambda.copy(), self.rhs_2_lambda.copy()]

        amplitude_scaling_one_body_lambda(
            self.rhs_1_lambda, self.f, self.m, self.n
        )
        amplitude_scaling_two_body_lambda(
            self.rhs_2_lambda, self.f, self.m, self.n
        )

        np.add((1 - theta) * self.rhs_1_lambda, theta * self.l_1, out=self.l_1)
        np.add((1 - theta) * self.rhs_2_lambda, theta * self.l_2, out=self.l_2)

    def _compute_effective_amplitudes(self):
        o, v = self.o, self.v

        term = np.einsum("ai, bj -> abij", self.t_1, self.t_1)
        term -= term.swapaxes(0, 1)
        term -= term.swapaxes(2, 3)

        self.xi.fill(0)
        self.xi += self.t_2
        self.xi += 0.25 * term

        self.tau.fill(0)
        self.tau += self.t_2
        self.tau += 0.5 * term

    def _compute_effective_three_body_intermediates(self):
        o, v = self.o, self.v

        self.G_pp.fill(0)
        self.G_pp -= 0.5 * np.tensordot(
            self.l_2, self.t_2, axes=((0, 1, 3), (2, 3, 1))
        )

        self.G_hh.fill(0)
        self.G_hh += 0.5 * np.tensordot(
            self.t_2, self.l_2, axes=((0, 1, 3), (2, 3, 1))
        )

    def _compute_ccsd_amplitude_s(self, iterative):
        o, v = self.o, self.v

        f = self.off_diag_f

        if not iterative:
            f = self.f

        self.rhs_1.fill(0)

        self.rhs_1 += f[v, o]
        self.rhs_1 += np.dot(self.F_pp, self.t_1)
        self.rhs_1 -= np.dot(self.t_1, self.F_hh)
        self.rhs_1 += np.einsum(
            "me, aeim -> ai", self.F_hp, self.t_2, optimize=True
        )
        self.rhs_1 += np.einsum(
            "em, amie -> ai", self.t_1, self.u[v, o, o, v], optimize=True
        )
        self.rhs_1 -= 0.5 * np.einsum(
            "aemn, mnie -> ai", self.t_2, self.u[o, o, o, v], optimize=True
        )
        self.rhs_1 += 0.5 * np.einsum(
            "amef, efim -> ai", self.u[v, o, v, v], self.t_2, optimize=True
        )

    def _compute_ccsd_amplitude_d(self):
        o, v = self.o, self.v

        self.rhs_2.fill(0)

        self.rhs_2 += self.u[v, v, o, o]
        term = -0.5 * np.dot(self.t_1, self.F_hp)
        term += self.F_pp
        term = np.einsum("aeij, be -> abij", self.t_2, term, optimize=True)
        term -= term.swapaxes(0, 1)
        self.rhs_2 += term

        term = 0.5 * np.dot(self.F_hp, self.t_1)
        term += self.F_hh
        term = np.einsum("abim, mj -> abij", self.t_2, term, optimize=True)
        term -= term.swapaxes(2, 3)
        self.rhs_2 -= term

        self.rhs_2 += 0.5 * np.tensordot(
            self.tau, self.W_hhhh, axes=((2, 3), (0, 1))
        )

        self.rhs_2 += 0.5 * np.tensordot(
            self.W_pppp, self.tau, axes=((2, 3), (0, 1))
        )

        term = np.einsum(
            "ei, mbej -> mbij", self.t_1, self.u[o, v, v, o], optimize=True
        )
        term = np.einsum("am, mbij -> abij", self.t_1, term, optimize=True)
        term *= -1.0
        term += np.tensordot(
            self.t_2, self.W_hpph, axes=((1, 3), (2, 0))
        ).transpose(0, 2, 1, 3)
        term -= term.swapaxes(0, 1)
        term -= term.swapaxes(2, 3)
        self.rhs_2 += term

        term = np.einsum(
            "abej, ei -> abij", self.u[v, v, v, o], self.t_1, optimize=True
        )
        term -= term.swapaxes(2, 3)
        self.rhs_2 += term

        term = np.tensordot(self.t_1, self.u[o, v, o, o], axes=((1), (0)))
        term -= term.swapaxes(0, 1)
        self.rhs_2 -= term

    def _compute_ccsd_lambda_amplitudes_s(self):
        self.rhs_1_lambda.fill(0)

        self.rhs_1_lambda += self.F_hp_lambda
        self.rhs_1_lambda += np.dot(self.l_1, self.F_pp_lambda)
        self.rhs_1_lambda -= np.dot(self.F_hh_lambda, self.l_1)
        self.rhs_1_lambda += np.tensordot(
            self.l_1, self.W_hpph_lambda, axes=((0, 1), (3, 1))
        )
        self.rhs_1_lambda += 0.5 * np.tensordot(
            self.l_2, self.W_ppph_lambda, axes=((1, 2, 3), (3, 0, 1))
        )
        self.rhs_1_lambda -= 0.5 * np.tensordot(
            self.W_hphh_lambda, self.l_2, axes=((1, 2, 3), (3, 0, 1))
        )
        self.rhs_1_lambda -= np.tensordot(
            self.G_pp, self.W_phpp_lambda, axes=((0, 1), (0, 2))
        )
        self.rhs_1_lambda -= np.tensordot(
            self.G_hh, self.W_hhhp_lambda, axes=((0, 1), (0, 2))
        )

    def _compute_ccsd_lambda_amplitudes_d(self):
        o, v = self.o, self.v

        self.rhs_2_lambda.fill(0)

        self.rhs_2_lambda += self.u[o, o, v, v]

        term = np.tensordot(self.l_2, self.F_pp_lambda, axes=((3), (0)))
        term -= term.swapaxes(2, 3)
        self.rhs_2_lambda += term

        term = np.einsum(
            "imab, jm -> ijab", self.l_2, self.F_hh_lambda, optimize=True
        )
        term -= term.swapaxes(0, 1)
        self.rhs_2_lambda -= term

        self.rhs_2_lambda += 0.5 * np.tensordot(
            self.W_hhhh_lambda, self.l_2, axes=((2, 3), (0, 1))
        )

        self.rhs_2_lambda += 0.5 * np.tensordot(
            self.l_2, self.W_pppp_lambda, axes=((2, 3), (0, 1))
        )

        term = np.tensordot(self.l_1, self.W_phpp_lambda, axes=((1), (0)))
        term -= term.swapaxes(0, 1)
        self.rhs_2_lambda += term

        term = np.einsum(
            "ma, ijmb -> ijab", self.l_1, self.W_hhhp_lambda, optimize=True
        )
        term -= term.swapaxes(2, 3)
        self.rhs_2_lambda -= term

        term = np.einsum(
            "imae, jebm -> ijab", self.l_2, self.W_hpph_lambda, optimize=True
        )
        term -= term.swapaxes(0, 1)
        term -= term.swapaxes(2, 3)
        self.rhs_2_lambda += term

        term = np.einsum("ia, jb -> ijab", self.l_1, self.F_hp_lambda)
        term -= term.swapaxes(0, 1)
        term -= term.swapaxes(2, 3)
        self.rhs_2_lambda += term

        term = np.tensordot(self.u[o, o, v, v], self.G_pp, axes=((3), (1)))
        term -= term.swapaxes(2, 3)
        self.rhs_2_lambda += term

        term = np.einsum(
            "imab, mj -> ijab", self.u[o, o, v, v], self.G_hh, optimize=True
        )
        term -= term.swapaxes(0, 1)
        self.rhs_2_lambda -= term

    def _compute_intermediates(self, iterative):
        o, v = self.o, self.v

        f = self.off_diag_f

        if not iterative:
            f = self.f

        # One-body intermediate F_{ae}
        self.F_pp.fill(0)
        self.F_pp += f[v, v]
        self.F_pp -= 0.5 * np.dot(self.t_1, f[o, v])
        self.F_pp += np.tensordot(
            self.u[v, o, v, v], self.t_1, axes=((1, 3), (1, 0))
        )
        self.F_pp -= 0.5 * np.tensordot(
            self.xi, self.u[o, o, v, v], axes=((1, 2, 3), (3, 0, 1))
        )

        # One-body intermediate F_{mi}
        self.F_hh.fill(0)
        self.F_hh += f[o, o]
        self.F_hh += 0.5 * np.dot(f[o, v], self.t_1)
        self.F_hh += np.einsum(
            "en, mnie -> mi", self.t_1, self.u[o, o, o, v], optimize=True
        )
        self.F_hh += 0.5 * np.tensordot(
            self.u[o, o, v, v], self.xi, axes=((1, 2, 3), (3, 0, 1))
        )

        # One-body intermediate F_{me}
        self.F_hp.fill(0)
        self.F_hp += f[o, v]
        self.F_hp += np.einsum(
            "fn, mnef -> me", self.t_1, self.u[o, o, v, v], optimize=True
        )

        # Two-body intermediate W_{mnij}
        self.W_hhhh.fill(0)
        self.W_hhhh += self.u[o, o, o, o]
        term = np.einsum(
            "ej, mnie -> mnij", self.t_1, self.u[o, o, o, v], optimize=True
        )
        term -= term.swapaxes(2, 3)
        self.W_hhhh += term
        self.W_hhhh += 0.25 * np.tensordot(
            self.u[o, o, v, v], self.tau, axes=((2, 3), (0, 1))
        )

        # Two-body intermediate W_{abef}
        self.W_pppp.fill(0)
        self.W_pppp += self.u[v, v, v, v]
        term = np.einsum(
            "bm, amef -> abef", self.t_1, self.u[v, o, v, v], optimize=True
        )
        term -= term.swapaxes(0, 1)
        self.W_pppp -= term
        self.W_pppp += 0.25 * np.tensordot(
            self.tau, self.u[o, o, v, v], axes=((2, 3), (0, 1))
        )

        # Two-body intermediate W_{mbej}
        self.W_hpph.fill(0)
        self.W_hpph += self.u[o, v, v, o]
        self.W_hpph += np.einsum(
            "fj, mbef -> mbej", self.t_1, self.u[o, v, v, v], optimize=True
        )
        self.W_hpph -= np.einsum(
            "bn, mnej -> mbej", self.t_1, self.u[o, o, v, o], optimize=True
        )
        self.W_hpph -= 0.5 * np.tensordot(
            self.t_2, self.u[o, o, v, v], axes=((0, 3), (3, 1))
        ).transpose(2, 0, 3, 1)
        term = np.einsum(
            "fj, mnef -> mnej", self.t_1, self.u[o, o, v, v], optimize=True
        )
        term = np.einsum("bn, mnej -> mbej", self.t_1, term, optimize=True)
        self.W_hpph -= term

    def _compute_lambda_intermediates(self):
        o, v = self.o, self.v

        self.F_pp_lambda.fill(0)
        self.F_pp_lambda += self.F_pp
        self.F_pp_lambda -= 0.5 * np.dot(self.t_1, self.F_hp)

        self.F_hh_lambda.fill(0)
        self.F_hh_lambda += self.F_hh
        self.F_hh_lambda += 0.5 * np.dot(self.F_hp, self.t_1)

        if self.F_hp_lambda is not self.F_hp:
            self.F_hp_lambda = self.F_hp

        self.W_hhhh_lambda.fill(0)
        self.W_hhhh_lambda += self.W_hhhh
        self.W_hhhh_lambda += 0.25 * np.tensordot(
            self.u[o, o, v, v], self.tau, axes=((2, 3), (0, 1))
        )

        self.W_pppp_lambda.fill(0)
        self.W_pppp_lambda += self.W_pppp
        self.W_pppp_lambda += 0.25 * np.tensordot(
            self.tau, self.u[o, o, v, v], axes=((2, 3), (0, 1))
        )

        self.W_hpph_lambda.fill(0)
        self.W_hpph_lambda += self.W_hpph
        self.W_hpph_lambda -= 0.5 * np.einsum(
            "fbjn, mnef -> mbej", self.t_2, self.u[o, o, v, v]
        )

        self.W_hhhp_lambda.fill(0)
        self.W_hhhp_lambda += self.u[o, o, o, v]
        self.W_hhhp_lambda += np.einsum(
            "fi, mnfe -> mnie", self.t_1, self.u[o, o, v, v]
        )

        self.W_phpp_lambda.fill(0)
        self.W_phpp_lambda += self.u[v, o, v, v]
        self.W_phpp_lambda -= np.tensordot(
            self.t_1, self.u[o, o, v, v], axes=((1), (0))
        )

        self.W_hphh_lambda.fill(0)
        self.W_hphh_lambda += self.u[o, v, o, o]
        self.W_hphh_lambda -= np.tensordot(
            self.F_hp_lambda, self.t_2, axes=((1), (1))
        )
        self.W_hphh_lambda -= np.einsum(
            "bn, mnij -> mbij", self.t_1, self.W_hhhh_lambda
        )
        self.W_hphh_lambda += 0.5 * np.tensordot(
            self.u[o, v, v, v], self.tau, axes=((2, 3), (0, 1))
        )
        term = np.einsum("mnie, bejn -> mbij", self.u[o, o, o, v], self.t_2)
        term -= term.swapaxes(2, 3)
        self.W_hphh_lambda += term
        term = -np.einsum("bfnj, mnef -> mbej", self.t_2, self.u[o, o, v, v])
        term += self.u[o, v, v, o]
        term = np.einsum("ei, mbej -> mbij", self.t_1, term)
        term -= term.swapaxes(2, 3)
        self.W_hphh_lambda += term

        self.W_ppph_lambda.fill(0)
        self.W_ppph_lambda += self.u[v, v, v, o]
        self.W_ppph_lambda -= np.einsum(
            "me, abmi -> abei", self.F_hp_lambda, self.t_2
        )
        self.W_ppph_lambda += np.tensordot(
            self.W_pppp_lambda, self.t_1, axes=((3), (0))
        )
        self.W_ppph_lambda += 0.5 * np.tensordot(
            self.tau, self.u[o, o, v, o], axes=((2, 3), (0, 1))
        )
        term = np.einsum("mbef, afmi -> abei", self.u[o, v, v, v], self.t_2)
        term -= term.swapaxes(0, 1)
        self.W_ppph_lambda -= term
        term = -np.einsum("bfni, mnef -> mbei", self.t_2, self.u[o, o, v, v])
        term += self.u[o, v, v, o]
        term = np.tensordot(self.t_1, term, axes=((1), (0)))
        term -= term.swapaxes(0, 1)
        self.W_ppph_lambda -= term
