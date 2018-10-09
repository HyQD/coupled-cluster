import numpy as np
from .cc import CoupledCluster
from .cc_helper import amplitude_scaling_two_body


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

        return energy + self.compute_reference_energy()

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

        term = np.tensordot(f, self.t_2, axes=((1), (1)))
        term -= term.swapaxes(0, 1)
        self.rhs_2 -= term

        term = np.tensordot(self.t_2, f, axes=((3), (0)))
        term -= term.swapaxes(2, 3)
        self.rhs_2 -= term

        self.rhs_2 += np.tensordot(self.W_pppp, self.t_2, axes=((2, 3), (0, 1)))

        term = np.tensordot(self.t_2, self.W_hh, axes=((3), (0)))
        term -= term.swapaxes(2, 3)
        self.rhs_2 += term

        term = np.tensordot(self.W_pp, self.t_2, axes=((1), (1)))
        term -= term.swapaxes(0, 1)
        self.rhs_2 -= term

        term = np.einsum("acim, bmjc -> abij", self.t_2, self.W_phhp)
        term -= term.swapaxes(0, 1)
        term -= term.swapaxes(2, 3)
        self.rhs_2 += term

        term = np.einsum("mnjn -> mj", self.u[o, o, o, o])
        self.rhs_2 += 0.5 * np.tensordot(self.t_2, term, axes=((3), (0)))

    def _compute_intermediates(self):
        o, v = self.o, self.v

        self.W_pppp.fill(0)
        self.W_pppp += 0.25 * np.tensordot(
            self.t_2, self.u[o, o, v, v], axes=((2, 3), (0, 1))
        )
        self.W_pppp += 0.5 * self.u[v, v, v, v]

        self.W_phhp.fill(0)
        self.W_phhp += self.u[v, o, o, v]
        self.W_phhp += 0.5 * np.einsum(
            "bdjn, mncd -> bmjc", self.t_2, self.u[o, o, v, v]
        )

        self.W_pp.fill(0)
        self.W_pp += 0.5 * np.tensordot(
            self.t_2, self.u[o, o, v, v], axes=((1, 2, 3), (2, 0, 1))
        )
