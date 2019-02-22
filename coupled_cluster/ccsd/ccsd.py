from coupled_cluster.cc import CoupledCluster

from coupled_cluster.ccsd.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)


class CoupledClusterSinglesDoubles(CoupledCluster):
    def __init__(self, system, include_singles=True, **kwargs):
        super().__init__(system, **kwargs)

        np = self.np
        # Add option to run ccd instead of ccsd. This is mostly used for
        # testing.
        self.include_singles = include_singles

        n, m = self.n, self.m

        self.rhs_t_1 = np.zeros((m, n), dtype=np.complex128)
        self.rhs_t_2 = np.zeros((m, m, n, n), dtype=np.complex128)

        self.rhs_l_1 = np.zeros((n, m), dtype=np.complex128)
        self.rhs_l_2 = np.zeros((n, n, m, m), dtype=np.complex128)

        self.t_1 = np.zeros_like(self.rhs_t_1)
        self.t_2 = np.zeros_like(self.rhs_t_2)

        self.l_1 = np.zeros_like(self.rhs_l_1)
        self.l_2 = np.zeros_like(self.rhs_l_2)

        self.d_1_t = np.diag(self.f)[self.o] - np.diag(self.f)[self.v].reshape(
            -1, 1
        )
        self.d_2_t = (
            np.diag(self.f)[self.o]
            + np.diag(self.f)[self.o].reshape(-1, 1)
            - np.diag(self.f)[self.v].reshape(-1, 1, 1)
            - np.diag(self.f)[self.v].reshape(-1, 1, 1, 1)
        )
        # Copying the transposed matrices for the lambda amplitudes (especially
        # d_2) greatly increases the speed of the division later on.
        self.d_1_l = self.d_1_t.T.copy()
        self.d_2_l = self.d_2_t.transpose(2, 3, 0, 1).copy()

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

        self.compute_initial_guess()

        self.rho_qp = np.zeros((self.l, self.l), dtype=np.complex128)
        self.changed_t = False

    def compute_initial_guess(self):
        np = self.np
        o, v = self.o, self.v

        if self.include_singles:
            np.copyto(self.rhs_t_1, self.f[v, o])
            np.divide(self.rhs_t_1, self.d_1_t, out=self.t_1)

            np.copyto(self.rhs_l_1, self.f[o, v])
            np.divide(self.rhs_l_1, self.d_1_l, out=self.l_1)

        np.copyto(self.rhs_t_2, self.u[v, v, o, o])
        np.divide(self.rhs_t_2, self.d_2_t, out=self.t_2)

        np.copyto(self.rhs_l_2, self.u[o, o, v, v])
        np.divide(self.rhs_l_2, self.d_2_l, out=self.l_2)

    def _get_t_copy(self):
        return [self.t_1.copy(), self.t_2.copy()]

    def _get_l_copy(self):
        return [self.l_1.copy(), self.l_2.copy()]

    def compute_energy(self):
        np = self.np
        o, v = self.o, self.v

        energy = np.einsum("ia, ai ->", self.f[o, v], self.t_1)
        energy += 0.25 * np.einsum(
            "ijab, abij ->", self.u[o, o, v, v], self.t_2
        )
        energy += 0.5 * np.einsum(
            "ijab, ai, bj ->", self.u[o, o, v, v], self.t_1, self.t_1
        )

        return energy + self.compute_reference_energy()

    def compute_t_amplitudes(self, theta, iterative=True):
        np = self.np
        f = self.off_diag_f if iterative else self.f

        self.rhs_t_1.fill(0)
        self.rhs_t_2.fill(0)

        # if self.include_singles:
        #    compute_t_1_amplitudes(
        #        f,
        #        self.u,
        #        self.t_1,
        #        self.t_2,
        #        self.o,
        #        self.v,
        #        out=self.rhs_t_1,
        #        np=np,
        #    )

        # compute_t_2_amplitudes(
        #    f,
        #    self.u,
        #    self.t_1,
        #    self.t_2,
        #    self.o,
        #    self.v,
        #    out=self.rhs_t_2,
        #    np=np,
        # )

        self._compute_effective_amplitudes()
        self._compute_intermediates(iterative=iterative)

        if self.include_singles:
            self._compute_ccsd_amplitude_s(iterative=iterative)

        self._compute_ccsd_amplitude_d()

        if not iterative:
            return [self.rhs_t_1.copy(), self.rhs_t_2.copy()]

        if self.include_singles:
            np.divide(self.rhs_t_1, self.d_1_t, out=self.rhs_t_1)
            np.add((1 - theta) * self.rhs_t_1, theta * self.t_1, out=self.t_1)

        np.divide(self.rhs_t_2, self.d_2_t, out=self.rhs_t_2)
        np.add((1 - theta) * self.rhs_t_2, theta * self.t_2, out=self.t_2)

        # Notify a change in t for recalculation of intermediates
        self.changed_t = True

    def compute_l_amplitudes(self, theta, iterative=True):
        np = self.np

        if self.changed_t:
            # Make sure that we use updated intermediates for lambda
            self._compute_effective_amplitudes()
            self._compute_intermediates(iterative=iterative)
            self.changed_t = False

        self._compute_effective_three_body_intermediates()
        self._compute_lambda_intermediates()

        if self.include_singles:
            self._compute_ccsd_lambda_amplitudes_s()

        self._compute_ccsd_lambda_amplitudes_d()

        if not iterative:
            return [self.rhs_l_1.copy(), self.rhs_l_2.copy()]

        if self.include_singles:
            np.divide(self.rhs_l_1, self.d_1_l, out=self.rhs_l_1)
            np.add((1 - theta) * self.rhs_l_1, theta * self.l_1, out=self.l_1)

        np.divide(self.rhs_l_2, self.d_2_l, out=self.rhs_l_2)
        np.add((1 - theta) * self.rhs_l_2, theta * self.l_2, out=self.l_2)

    def compute_one_body_density_matrix(self):
        np = self.np
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

    def compute_two_body_density_matrix(self):
        pass

    def _compute_time_evolution_probability(self):
        np = self.np
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

    def _compute_effective_amplitudes(self):
        np = self.np
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
        np = self.np
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
        np = self.np
        o, v = self.o, self.v

        f = self.off_diag_f

        if not iterative:
            f = self.f

        self.rhs_t_1.fill(0)

        self.rhs_t_1 += f[v, o]
        self.rhs_t_1 += np.dot(self.F_pp, self.t_1)
        self.rhs_t_1 -= np.dot(self.t_1, self.F_hh)
        self.rhs_t_1 += np.einsum(
            "me, aeim -> ai", self.F_hp, self.t_2, optimize=True
        )
        self.rhs_t_1 += np.einsum(
            "em, amie -> ai", self.t_1, self.u[v, o, o, v], optimize=True
        )
        self.rhs_t_1 -= 0.5 * np.einsum(
            "aemn, mnie -> ai", self.t_2, self.u[o, o, o, v], optimize=True
        )
        self.rhs_t_1 += 0.5 * np.einsum(
            "amef, efim -> ai", self.u[v, o, v, v], self.t_2, optimize=True
        )

        # TODO: Remove this test
        out = compute_t_1_amplitudes(f, self.u, self.t_1, self.t_2, o, v, np=np)
        np.testing.assert_allclose(self.rhs_t_1, out)

    def _compute_ccsd_amplitude_d(self):
        np = self.np
        o, v = self.o, self.v

        self.rhs_t_2.fill(0)

        ### TODO: Remove this test
        import coupled_cluster.ccd.rhs_t as ccd_t
        import coupled_cluster.ccsd.rhs_t as ccsd_t

        f = self.off_diag_f
        out = np.zeros_like(self.rhs_t_2)
        ###

        self.rhs_t_2 += self.u[v, v, o, o]

        ### TODO: Remove this test
        ccd_t.add_d1_t(self.u, o, v, out, np)
        np.testing.assert_allclose(self.rhs_t_2, out)
        ###

        term = -0.5 * np.dot(self.t_1, self.F_hp)
        term += self.F_pp
        term = np.einsum("aeij, be -> abij", self.t_2, term, optimize=True)
        term -= term.swapaxes(0, 1)
        self.rhs_t_2 += term

        ### TODO: Remove this test
        ccd_t.add_d2a_t(f, self.t_2, o, v, out, np)
        ccd_t.add_d3d_t(self.u, self.t_2, o, v, out, np)
        ccsd_t.add_d5b_t(f, self.t_1, self.t_2, o, v, out, np)
        ccsd_t.add_d5g_t(self.u, self.t_1, self.t_2, o, v, out, np)
        ccsd_t.add_d7e_t(self.u, self.t_1, self.t_2, o, v, out, np)

        np.testing.assert_allclose(out, self.rhs_t_2)
        ###

        term = 0.5 * np.dot(self.F_hp, self.t_1)
        term += self.F_hh
        term = np.einsum("abim, mj -> abij", self.t_2, term, optimize=True)
        term -= term.swapaxes(2, 3)
        self.rhs_t_2 -= term

        ### TODO: Remove this test
        ccd_t.add_d2b_t(f, self.t_2, o, v, out, np)
        ccd_t.add_d3c_t(self.u, self.t_2, o, v, out, np)
        ccsd_t.add_d5a_t(f, self.t_1, self.t_2, o, v, out, np)
        ccsd_t.add_d5h_t(self.u, self.t_1, self.t_2, o, v, out, np)
        ccsd_t.add_d7d_t(self.u, self.t_1, self.t_2, o, v, out, np)

        np.testing.assert_allclose(self.rhs_t_2, out)
        ###

        self.rhs_t_2 += 0.5 * np.tensordot(
            self.tau, self.W_hhhh, axes=((2, 3), (0, 1))
        )

        ### TODO: Remove this test
        ccd_t.add_d2d_t(self.u, self.t_2, o, v, out, np)
        ccd_t.add_d3a_t(0.5 * self.u, self.t_2, o, v, out, np)
        ccsd_t.add_d5f_t(self.u, self.t_1, self.t_2, o, v, out, np)
        ccsd_t.add_d7a_t(0.5 * self.u, self.t_1, self.t_2, o, v, out, np)
        ccsd_t.add_d6b_t(self.u, self.t_1, o, v, out, np)
        ccsd_t.add_d8b_t(self.u, self.t_1, o, v, out, np)
        ccsd_t.add_d7b_t(0.5 * self.u, self.t_1, self.t_2, o, v, out, np)
        ccsd_t.add_d9_t(0.5 * self.u, self.t_1, o, v, out, np)

        np.testing.assert_allclose(self.rhs_t_2, out)
        ###

        self.rhs_t_2 += 0.5 * np.tensordot(
            self.W_pppp, self.tau, axes=((2, 3), (0, 1))
        )

        ### TODO: Remove this test
        ccd_t.add_d2c_t(self.u, self.t_2, o, v, out, np)
        ccd_t.add_d3a_t(0.5 * self.u, self.t_2, o, v, out, np)
        ccsd_t.add_d5e_t(self.u, self.t_1, self.t_2, o, v, out, np)
        ccsd_t.add_d7b_t(0.5 * self.u, self.t_1, self.t_2, o, v, out, np)
        ccsd_t.add_d6a_t(self.u, self.t_1, o, v, out, np)
        ccsd_t.add_d8a_t(-self.u, self.t_1, o, v, out, np)
        ccsd_t.add_d7a_t(0.5 * self.u, self.t_1, self.t_2, o, v, out, np)
        ccsd_t.add_d9_t(0.5 * self.u, self.t_1, o, v, out, np)

        np.testing.assert_allclose(self.rhs_t_2, out)
        ###

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
        self.rhs_t_2 += term

        term = np.einsum(
            "abej, ei -> abij", self.u[v, v, v, o], self.t_1, optimize=True
        )
        term -= term.swapaxes(2, 3)
        self.rhs_t_2 += term

        term = np.tensordot(self.t_1, self.u[o, v, o, o], axes=((1), (0)))
        term -= term.swapaxes(0, 1)
        self.rhs_t_2 -= term

        # TODO: Remove this test
        out = compute_t_2_amplitudes(f, self.u, self.t_1, self.t_2, o, v, np=np)
        np.testing.assert_allclose(self.rhs_t_2, out)

    def _compute_ccsd_lambda_amplitudes_s(self):
        np = self.np
        self.rhs_l_1.fill(0)

        self.rhs_l_1 += self.F_hp_lambda
        self.rhs_l_1 += np.dot(self.l_1, self.F_pp_lambda)
        self.rhs_l_1 -= np.dot(self.F_hh_lambda, self.l_1)
        self.rhs_l_1 += np.tensordot(
            self.l_1, self.W_hpph_lambda, axes=((0, 1), (3, 1))
        )
        self.rhs_l_1 += 0.5 * np.tensordot(
            self.l_2, self.W_ppph_lambda, axes=((1, 2, 3), (3, 0, 1))
        )
        self.rhs_l_1 -= 0.5 * np.tensordot(
            self.W_hphh_lambda, self.l_2, axes=((1, 2, 3), (3, 0, 1))
        )
        self.rhs_l_1 -= np.tensordot(
            self.G_pp, self.W_phpp_lambda, axes=((0, 1), (0, 2))
        )
        self.rhs_l_1 -= np.tensordot(
            self.G_hh, self.W_hhhp_lambda, axes=((0, 1), (0, 2))
        )

    def _compute_ccsd_lambda_amplitudes_d(self):
        np = self.np
        o, v = self.o, self.v

        self.rhs_l_2.fill(0)

        # d1
        self.rhs_l_2 += self.u[o, o, v, v]

        # d2a and d3f
        self.rhs_l_2 += 0.5 * np.tensordot(
            self.W_hhhh_lambda, self.l_2, axes=((2, 3), (0, 1))
        )

        # d2b and d3b
        self.rhs_l_2 += 0.5 * np.tensordot(
            self.l_2, self.W_pppp_lambda, axes=((2, 3), (0, 1))
        )

        # d2c and d3a
        term = np.tensordot(self.l_2, self.F_pp_lambda, axes=((3), (0)))
        term -= term.swapaxes(2, 3)
        self.rhs_l_2 += term

        # d2d and d3c
        term = np.einsum(
            "imab, jm -> ijab", self.l_2, self.F_hh_lambda, optimize=True
        )
        term -= term.swapaxes(0, 1)
        self.rhs_l_2 -= term

        # d2e and d3d
        term = np.einsum(
            "imae, jebm -> ijab", self.l_2, self.W_hpph_lambda, optimize=True
        )
        term -= term.swapaxes(0, 1)
        term -= term.swapaxes(2, 3)
        self.rhs_l_2 += term

        # d3e
        term = np.einsum(
            "imab, mj -> ijab", self.u[o, o, v, v], self.G_hh, optimize=True
        )
        term -= term.swapaxes(0, 1)
        self.rhs_l_2 -= term

        # d3g
        term = np.tensordot(self.u[o, o, v, v], self.G_pp, axes=((3), (1)))
        term -= term.swapaxes(2, 3)
        self.rhs_l_2 += term

        term = np.tensordot(self.l_1, self.W_phpp_lambda, axes=((1), (0)))
        term -= term.swapaxes(0, 1)
        self.rhs_l_2 += term

        term = np.einsum(
            "ma, ijmb -> ijab", self.l_1, self.W_hhhp_lambda, optimize=True
        )
        term -= term.swapaxes(2, 3)
        self.rhs_l_2 -= term

        term = np.einsum("ia, jb -> ijab", self.l_1, self.F_hp_lambda)
        term -= term.swapaxes(0, 1)
        term -= term.swapaxes(2, 3)
        self.rhs_l_2 += term

    def _compute_intermediates(self, iterative):
        np = self.np
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
        np = self.np
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
