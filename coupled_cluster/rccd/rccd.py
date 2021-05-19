from coupled_cluster.cc import CoupledCluster
from coupled_cluster.rccd.rhs_t import compute_t_2_amplitudes
from coupled_cluster.rccd.rhs_l import compute_l_2_amplitudes
from coupled_cluster.rccd.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)
from coupled_cluster.cc_helper import construct_d_t_2_matrix

from opt_einsum import contract


class RCCD(CoupledCluster):
    """Coupled Cluster Doubles

    Class for Coupled Cluster solver, including
    double excitations.

    Parameters
    ----------
    system : QuantumSystem
        QuantumSystem class describing the system
    """

    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

        np = self.np
        n, m = self.n, self.m

        self.rhs_t_2 = np.zeros((m, m, n, n), dtype=self.u.dtype)
        self.rhs_l_2 = np.zeros((n, n, m, m), dtype=self.u.dtype)

        self.t_2 = np.zeros_like(self.rhs_t_2)
        self.l_2 = np.zeros_like(self.rhs_l_2)

        self.d_t_2 = construct_d_t_2_matrix(self.f, self.o, self.v, np)
        self.d_l_2 = self.d_t_2.transpose(2, 3, 0, 1).copy()

        self.l_2_mixer = None
        self.t_2_mixer = None

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

    def compute_l_residuals(self):
        return [self.np.linalg.norm(self.rhs_l_2)]

    def compute_t_residuals(self):
        return [self.np.linalg.norm(self.rhs_t_2)]

    def setup_l_mixer(self, **kwargs):
        if self.l_2_mixer is None:
            self.l_2_mixer = self.mixer(**kwargs)

        self.l_2_mixer.clear_vectors()

    def setup_t_mixer(self, **kwargs):
        if self.t_2_mixer is None:
            self.t_2_mixer = self.mixer(**kwargs)

        self.t_2_mixer.clear_vectors()

    def compute_energy(self):
        e_ref = self.system.compute_reference_energy()
        ccd_corr = 2 * contract(
            "abij,ijab->", self.t_2, self.u[self.o, self.o, self.v, self.v]
        )
        ccd_corr -= contract(
            "abij,ijba->", self.t_2, self.u[self.o, self.o, self.v, self.v]
        )
        return e_ref + ccd_corr

    def compute_t_amplitudes(self):
        np = self.np

        self.rhs_t_2.fill(0)
        self.rhs_t_2 = compute_t_2_amplitudes(
            self.f, self.u, self.t_2, self.o, self.v, out=self.rhs_t_2, np=np
        )

        trial_vector = self.t_2
        direction_vector = np.divide(self.rhs_t_2, self.d_t_2)
        error_vector = self.rhs_t_2.copy()

        self.t_2 = self.t_2_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

    def compute_l_amplitudes(self):
        np = self.np

        self.rhs_l_2.fill(0)
        self.rhs_l_2 = compute_l_2_amplitudes(
            self.f,
            self.u,
            self.t_2,
            self.l_2,
            self.o,
            self.v,
            out=self.rhs_l_2,
            np=np,
        )

        trial_vector = self.l_2
        direction_vector = np.divide(self.rhs_l_2, self.d_l_2)
        error_vector = self.rhs_l_2.copy()

        self.l_2 = self.l_2_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

    def compute_one_body_density_matrix(self):
        return compute_one_body_density_matrix(
            self.t_2, self.l_2, self.o, self.v, np=self.np
        )

    def compute_two_body_density_matrix(self):
        return compute_two_body_density_matrix(
            self.t_2, self.l_2, self.o, self.v, np=self.np
        )
