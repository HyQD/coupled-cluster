import numpy as np
from coupled_cluster.cc import CoupledCluster
from coupled_cluster.ccd.rhs import compute_t_2_amplitude


class CoupledClusterDoubles(CoupledCluster):
    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

        n, m = self.n, self.m

        self.rhs_2 = np.zeros((m, m, n, n), dtype=np.complex128)
        self.t_2 = np.zeros_like(self.rhs_2)
        self.d_2 = (
            np.diag(self.f)[self.o]
            + np.diag(self.f)[self.o].reshape(-1, 1)
            - np.diag(self.f)[self.v].reshape(-1, 1, 1)
            - np.diag(self.f)[self.v].reshape(-1, 1, 1, 1)
        )

        self._compute_initial_guess()

    def _get_t_copy(self):
        return [self.t_2.copy()]

    def _set_t(self, t):
        t_2 = t[0]

        np.copyto(self.t_2, t_2)

    def _compute_initial_guess(self):
        o, v = self.o, self.v

        np.copyto(self.rhs_2, self.u[v, v, o, o])
        np.divide(self.rhs_2, self.d_2, out=self.t_2)

    def _compute_energy(self):
        o, v = self.o, self.v

        energy = 0.25 * np.einsum("abij, abij ->", self.u[v, v, o, o], self.t_2)
        energy += self.compute_reference_energy()

        return energy

    def _compute_amplitudes(self, theta, iterative=True):
        f = self.off_diag_f if iterative else self.f

        self.rhs_2.fill(0)
        compute_t_2_amplitude(
            f, self.u, self.t_2, self.o, self.v, out=self.rhs_2, np=np
        )

        if not iterative:
            return [self.rhs_2.copy()]

        np.divide(self.rhs_2, self.d_2, out=self.rhs_2)
        np.add((1 - theta) * self.rhs_2, theta * self.t_2, out=self.t_2)
