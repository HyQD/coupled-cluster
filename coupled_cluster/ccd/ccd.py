import numpy as np
from coupled_cluster.cc import CoupledCluster
from coupled_cluster.ccd.rhs_t import compute_t_2_amplitudes
from coupled_cluster.ccd.rhs_l import compute_l_2_amplitudes


class CoupledClusterDoubles(CoupledCluster):
    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

        n, m = self.n, self.m

        self.rhs_2_t = np.zeros((m, m, n, n), dtype=np.complex128)
        self.rhs_2_l = np.zeros((n, n, m, m), dtype=np.complex128)

        self.t_2 = np.zeros_like(self.rhs_2_t)
        self.l_2 = np.zeros_like(self.rhs_2_l)

        self.d_2_t = (
            np.diag(self.f)[self.o]
            + np.diag(self.f)[self.o].reshape(-1, 1)
            - np.diag(self.f)[self.v].reshape(-1, 1, 1)
            - np.diag(self.f)[self.v].reshape(-1, 1, 1, 1)
        )
        self.d_2_l = self.d_2_t.transpose(2, 3, 0, 1).copy()

        self._compute_initial_guess()

    def _get_t_copy(self):
        return [self.t_2.copy()]

    def _set_t(self, t):
        t_2 = t[0]

        np.copyto(self.t_2, t_2)

    def _compute_initial_guess(self):
        o, v = self.o, self.v

        np.copyto(self.rhs_2_t, self.u[v, v, o, o])
        np.divide(self.rhs_2_t, self.d_2_t, out=self.t_2)

        np.copyto(self.rhs_2_l, self.u[o, o, v, v])
        np.divide(self.rhs_2_l, self.d_2_l, out=self.l_2)

    def _compute_energy(self):
        o, v = self.o, self.v

        energy = 0.25 * np.einsum("abij, abij ->", self.u[v, v, o, o], self.t_2)
        energy += self.compute_reference_energy()

        return energy

    def _compute_t_amplitudes(self, theta, iterative=True):
        f = self.off_diag_f if iterative else self.f

        self.rhs_2_t.fill(0)
        compute_t_2_amplitudes(
            f, self.u, self.t_2, self.o, self.v, out=self.rhs_2_t, np=np
        )

        if not iterative:
            return [self.rhs_2_t.copy()]

        np.divide(self.rhs_2_t, self.d_2_t, out=self.rhs_2_t)
        np.add((1 - theta) * self.rhs_2_t, theta * self.t_2, out=self.t_2)

    def _compute_l_amplitudes(self, theta, iterative=True):
        self.rhs_2_l.fill(0)
        compute_l_2_amplitude(
            self.u, self.t_2, self.l_2, self.o, self.v, out=self.rhs_2_l, np=np
        )

        if not iterative:
            return [self.rhs_2_l.copy()]

        np.divide(self.rhs_2_l, self.d_2_l, out=self.rhs_2_l)
        np.add((1 - theta) * self.rhs_2_l, theta * self.l_2, out=self.l_2)
