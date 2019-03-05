from coupled_cluster.cc import CoupledCluster
from coupled_cluster.ccd.energies import compute_ccd_ground_state_energy
from coupled_cluster.ccd.rhs_t import compute_t_2_amplitudes
from coupled_cluster.ccd.rhs_l import compute_l_2_amplitudes
from coupled_cluster.ccd.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)


class CoupledClusterDoubles(CoupledCluster):
    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

        np = self.np
        n, m = self.n, self.m

        self.rhs_t_2 = np.zeros((m, m, n, n), dtype=self.u.dtype)
        self.rhs_l_2 = np.zeros((n, n, m, m), dtype=self.u.dtype)

        self.t_2 = np.zeros_like(self.rhs_t_2)
        self.l_2 = np.zeros_like(self.rhs_l_2)

        self.d_t_2 = (
            np.diag(self.f)[self.o]
            + np.diag(self.f)[self.o].reshape(-1, 1)
            - np.diag(self.f)[self.v].reshape(-1, 1, 1)
            - np.diag(self.f)[self.v].reshape(-1, 1, 1, 1)
        )
        self.d_l_2 = self.d_t_2.transpose(2, 3, 0, 1).copy()

        self.compute_initial_guess()

    def compute_initial_guess(self):
        np = self.np
        o, v = self.o, self.v

        np.copyto(self.rhs_t_2, self.u[v, v, o, o])
        np.divide(self.rhs_t_2, self.d_t_2, out=self.t_2)

        np.copyto(self.rhs_l_2, self.u[o, o, v, v])
        np.divide(self.rhs_l_2, self.d_l_2, out=self.l_2)

    def _get_t_copy(self):
        return [self.t_2.copy()]

    def _get_l_copy(self):
        return [self.l_2.copy()]

    def compute_energy(self):
        return compute_ccd_ground_state_energy(
            self.f, self.u, self.t_2, self.o, self.v, np=self.np
        )

    def compute_t_amplitudes(self, theta):
        np = self.np

        self.rhs_t_2.fill(0)
        compute_t_2_amplitudes(
            self.off_diag_f,
            self.u,
            self.t_2,
            self.o,
            self.v,
            out=self.rhs_t_2,
            np=np,
        )

        np.divide(self.rhs_t_2, self.d_t_2, out=self.rhs_t_2)
        np.add((1 - theta) * self.rhs_t_2, theta * self.t_2, out=self.t_2)

    def compute_l_amplitudes(self, theta):
        np = self.np

        self.rhs_l_2.fill(0)
        compute_l_2_amplitudes(
            self.off_diag_f,
            self.u,
            self.t_2,
            self.l_2,
            self.o,
            self.v,
            out=self.rhs_l_2,
            np=np,
        )

        np.divide(self.rhs_l_2, self.d_l_2, out=self.rhs_l_2)
        np.add((1 - theta) * self.rhs_l_2, theta * self.l_2, out=self.l_2)

    def compute_one_body_density_matrix(self):
        return compute_one_body_density_matrix(
            self.t_2, self.l_2, self.o, self.v, np=self.np
        )

    def compute_two_body_density_matrix(self):
        return compute_two_body_density_matrix(
            self.t_2, self.l_2, self.o, self.v, np=self.np
        )
