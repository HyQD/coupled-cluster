import numpy as np
from .cc import CoupledCluster
from .cc_helper import amplitude_scaling_two_body


# TODO: The energy is too low, which means there is an error somewhere...
class CoupledClusterDoubles(CoupledCluster):
    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

        n, m = self.n, self.m

        self.rhs_2 = np.zeros((m, m, n, n), dtype=np.complex128)
        self.t_2 = np.zeros((m, m, n, n), dtype=np.complex128)

        self.W_pppp = np.zeros((m, m, m, m), dtype=np.complex128)
        self.W_phhp = np.zeros((m, n, n, m), dtype=np.complex128)
        self.W_pp = np.zeros((m, m), dtype=np.complex128)
        self.W_hh = np.zeros((n, n), dtype=np.complex128)

        self._compute_initial_guess()

    def _get_t_copy(self):
        return [self.t_2.copy()]

    def _set_t(self, t):
        t_2 = t[0]

        np.copyto(self.t_2, t_2)

    def _compute_initial_guess(self):
        o, v = self.o, self.v

        np.copyto(self.rhs_2, self.u[v, v, o, o])

        amplitude_scaling_two_body(self.rhs_2, self.f, self.m, self.n)
        np.copyto(self.t_2, self.rhs_2)

    def _compute_energy(self):
        return self._compute_ccd_energy()

    def _compute_ccd_energy(self):
        o, v = self.o, self.v

        energy = 0.25 * np.einsum("abij, abij ->", self.u[v, v, o, o], self.t_2)
        energy += self.compute_reference_energy()

        # e_ref = np.einsum("ii ->", self.f[o, o])
        # e_ref -= 0.5 * np.einsum("ijij ->", self.u[o, o, o, o])
        # e_ref += 0.25 * np.einsum("abij, abij ->", self.u[v, v, o, o], self.t_2)
        # np.testing.assert_allclose(
        #    energy,
        #    e_ref
        # )

        return energy

    def _compute_amplitudes(self, theta, iterative=True):
        self._compute_intermediates()
        self._compute_ccd_amplitude_d(iterative=iterative)

        if not iterative:
            return [self.rhs_2.copy()]

        amplitude_scaling_two_body(self.rhs_2, self.f, self.m, self.n)
        np.add((1 - theta) * self.rhs_2, theta * self.t_2, out=self.t_2)

    def _compute_ccd_amplitude_d(self, iterative):
        o, v = self.o, self.v

        f = self.off_diag_f

        if not iterative:
            f = self.f

        self.rhs_2.fill(0)
        self.rhs_2 += self.u[v, v, o, o]

        term = np.tensordot(f[v, v], self.t_2, axes=((1), (1)))
        term -= term.swapaxes(0, 1)

        # term_test = np.einsum("bc, acij -> abij", f[v, v], self.t_2)
        # term_test -= term_test.swapaxes(0, 1)
        # np.testing.assert_allclose(
        #    term,
        #    -term_test
        # )

        self.rhs_2 -= term

        term = np.tensordot(self.t_2, f[o, o], axes=((3), (0)))
        term -= term.swapaxes(2, 3)

        # term_test = np.einsum("kj, abik -> abij", f[o, o], self.t_2)
        # term_test -= term_test.swapaxes(2, 3)
        # np.testing.assert_allclose(
        #    term,
        #    term_test
        # )

        self.rhs_2 -= term

        self.rhs_2 += np.tensordot(self.W_pppp, self.t_2, axes=((2, 3), (0, 1)))
        # term = np.tensordot(self.W_pppp, self.t_2, axes=((2, 3), (0, 1)))
        # term_test = np.einsum(
        #    "cdij, abcd -> abij", self.t_2, self.W_pppp
        # )
        # np.testing.assert_allclose(
        #    term,
        #    term_test
        # )
        # self.rhs_2 += term

        term = np.tensordot(self.t_2, self.W_hh, axes=((3), (0)))
        term -= term.swapaxes(2, 3)

        # term_test = np.einsum(
        #    "abin, nj -> abij", self.t_2, self.W_hh
        # )
        # term_test -= term_test.swapaxes(2, 3)
        # np.testing.assert_allclose(
        #    term,
        #    term_test
        # )

        self.rhs_2 += term

        term = np.tensordot(self.W_pp, self.t_2, axes=((1), (1)))
        term -= term.swapaxes(0, 1)

        # term_test = np.einsum("bdij, ad -> abij", self.t_2, self.W_pp)
        # term_test -= term_test.swapaxes(0, 1)
        # np.testing.assert_allclose(
        #    term,
        #    term_test
        # )

        self.rhs_2 -= term

        term = np.einsum("acim, bmjc -> abij", self.t_2, self.W_phhp)
        term -= term.swapaxes(0, 1)
        term -= term.swapaxes(2, 3)

        # term_test = np.einsum("acim, bmjc -> abij", self.t_2, self.W_phhp)
        # term_test -= term_test.swapaxes(0, 1)
        # term_test -= term_test.swapaxes(2, 3)
        # np.testing.assert_allclose(
        #    term,
        #    term_test
        # )

        self.rhs_2 += term

        term = np.einsum("mnjn -> mj", self.u[o, o, o, o])
        term = 0.5 * np.tensordot(self.t_2, term, axes=((3), (0)))

        # term_test = 0.5 * np.einsum(
        #    "abim, mnjn -> abij", self.t_2, self.u[o, o, o, o]
        # )
        # np.testing.assert_allclose(
        #    term,
        #    term_test
        # )

        self.rhs_2 += term

    def _compute_intermediates(self):
        o, v = self.o, self.v

        self.W_pppp.fill(0)
        self.W_pppp += 0.25 * np.tensordot(
            self.t_2, self.u[o, o, v, v], axes=((2, 3), (0, 1))
        )
        self.W_pppp += 0.5 * self.u[v, v, v, v]
        # np.testing.assert_allclose(
        #    self.W_pppp,
        #    0.5 * self.u[v, v, v, v]
        #    + 0.25 * np.einsum(
        #        "abmn, mncd -> abcd", self.t_2, self.u[o, o, v, v]
        #    )
        # )

        self.W_phhp.fill(0)
        self.W_phhp += self.u[v, o, o, v]
        self.W_phhp += 0.5 * np.einsum(
            "bdjn, mncd -> bmjc", self.t_2, self.u[o, o, v, v]
        )

        self.W_pp.fill(0)
        self.W_pp += 0.5 * np.tensordot(
            self.t_2, self.u[o, o, v, v], axes=((1, 2, 3), (2, 0, 1))
        )
        # np.testing.assert_allclose(
        #    self.W_pp,
        #    0.5 * np.einsum(
        #        "acnm, nmcd -> ad", self.t_2, self.u[o, o, v, v]
        #    )
        # )

        self.W_hh.fill(0)
        self.W_hh += 0.5 * np.tensordot(
            self.u[o, o, v, v], self.t_2, axes=((0, 2, 3), (3, 0, 1))
        )
        # np.testing.assert_allclose(
        #    self.W_hh,
        #    0.5 * np.einsum(
        #        "cdjm, mncd -> nj", self.t_2, self.u[o, o, v, v]
        #    )
        # )
