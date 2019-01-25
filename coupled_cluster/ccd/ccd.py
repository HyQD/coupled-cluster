import numpy as np
from coupled_cluster.cc import CoupledCluster
from coupled_cluster.ccd.energies import compute_ccd_ground_state_energy
from coupled_cluster.ccd.rhs_t import compute_t_2_amplitudes
from coupled_cluster.ccd.rhs_l import compute_l_2_amplitudes
from coupled_cluster.ccd.density_matrices import compute_one_body_density_matrix


class CoupledClusterDoubles(CoupledCluster):
    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

        n, m = self.n, self.m

        self.rhs_t_2 = np.zeros((m, m, n, n), dtype=np.complex128)
        self.rhs_l_2 = np.zeros((n, n, m, m), dtype=np.complex128)

        self.t_2 = np.zeros_like(self.rhs_t_2)
        self.l_2 = np.zeros_like(self.rhs_l_2)

        self.d_t_2 = (
            np.diag(self.f)[self.o]
            + np.diag(self.f)[self.o].reshape(-1, 1)
            - np.diag(self.f)[self.v].reshape(-1, 1, 1)
            - np.diag(self.f)[self.v].reshape(-1, 1, 1, 1)
        )
        self.d_l_2 = self.d_t_2.transpose(2, 3, 0, 1).copy()

        self._compute_initial_guess()

    def _get_t_copy(self):
        return [self.t_2.copy()]

    def _get_l_copy(self):
        return [self.l_2.copy()]

    def _set_t(self, t):
        t_2 = t[0]

        np.copyto(self.t_2, t_2)

    def _set_l(self, l):
        l_2 = l[0]

        np.copyto(self.l_2, l_2)

    def _compute_initial_guess(self):
        o, v = self.o, self.v

        np.copyto(self.rhs_t_2, self.u[v, v, o, o])
        np.divide(self.rhs_t_2, self.d_t_2, out=self.t_2)

        np.copyto(self.rhs_l_2, self.u[o, o, v, v])
        np.divide(self.rhs_l_2, self.d_l_2, out=self.l_2)

    def _compute_energy(self):
        return compute_ccd_ground_state_energy(
            self.f, self.u, self.t_2, self.o, self.v, np=np
        )

    def _compute_t_amplitudes(self, theta, iterative=True):
        f = self.off_diag_f if iterative else self.f

        self.rhs_t_2.fill(0)
        compute_t_2_amplitudes(
            f, self.u, self.t_2, self.o, self.v, out=self.rhs_t_2, np=np
        )

        if not iterative:
            return [self.rhs_t_2.copy()]

        np.divide(self.rhs_t_2, self.d_t_2, out=self.rhs_t_2)
        np.add((1 - theta) * self.rhs_t_2, theta * self.t_2, out=self.t_2)

    def _compute_l_amplitudes(self, theta, iterative=True):
        f = self.off_diag_f if iterative else self.f

        self.rhs_l_2.fill(0)
        compute_l_2_amplitudes(
            f,
            self.u,
            self.t_2,
            self.l_2,
            self.o,
            self.v,
            out=self.rhs_l_2,
            np=np,
        )

        if not iterative:
            return [self.rhs_l_2.copy()]

        np.divide(self.rhs_l_2, self.d_l_2, out=self.rhs_l_2)
        np.add((1 - theta) * self.rhs_l_2, theta * self.l_2, out=self.l_2)

    def _compute_one_body_density_matrix(self):
        return compute_one_body_density_matrix(
            self.t_2, self.l_2, self.o, self.v, np=np
        )
