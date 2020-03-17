from coupled_cluster.cc import CoupledCluster
from coupled_cluster.ccs.energies import compute_ccs_ground_state_energy
from coupled_cluster.ccs.rhs_t import compute_t_1_amplitudes
from coupled_cluster.ccs.rhs_l import compute_l_1_amplitudes
from coupled_cluster.ccs.density_matrices import compute_one_body_density_matrix
from coupled_cluster.cc_helper import construct_d_t_1_matrix


class CCS(CoupledCluster):
    """Coupled cluster singles

    Class for coupled cluster solver with singles excitations.

    Parameters
    ----------
    system: QuantumSystem
        QuantumSystem class describing the system
    """

    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

        np = self.np
        n, m = self.n, self.m

        self.rhs_t_1 = np.zeros((m, n), dtype=self.u.dtype)  # ai
        self.rhs_l_1 = np.zeros((n, m), dtype=self.u.dtype)  # ia

        self.t_1 = np.zeros_like(self.rhs_t_1)
        self.l_1 = np.zeros_like(self.rhs_l_1)

        self.d_t_1 = construct_d_t_1_matrix(self.f, self.o, self.v, np)
        self.d_l_1 = self.d_t_1.transpose(1, 0).copy()

        # Mixer
        self.t_mixer = None
        self.l_mixer = None

        # Go!
        self.compute_initial_guess()

    def compute_initial_guess(self):
        np = self.np
        o, v = self.o, self.v

        np.copyto(self.rhs_t_1, self.f[v, o])
        np.divide(self.rhs_t_1, self.d_t_1, out=self.t_1)

        np.copyto(self.rhs_l_1, self.f[o, v])
        np.divide(self.rhs_l_1, self.d_l_1, out=self.l_1)

    def _get_t_copy(self):
        return [self.t_1.copy()]

    def _get_l_copy(self):
        return [self.l_1.copy()]

    def compute_l_residuals(self):
        return [self.np.linalg.norm(self.rhs_l_1)]

    def compute_t_residuals(self):
        return [self.np.linalg.norm(self.rhs_t_1)]

    def setup_l_mixer(self, **kwargs):
        if self.l_mixer is None:
            self.l_mixer = self.mixer(**kwargs)

        self.l_mixer.clear_vectors()

    def setup_t_mixer(self, **kwargs):
        if self.t_mixer is None:
            self.t_mixer = self.mixer(**kwargs)

        self.t_mixer.clear_vectors()

    def compute_energy(self):
        """Compute ground state CCS energy.

        Returns
        -------
        float
            CCS ground state energy
        """
        return compute_ccs_ground_state_energy(
            self.f, self.u, self.t_1, self.o, self.v, np=self.np
        )

    def compute_t_amplitudes(self):
        np = self.np

        self.rhs_t_1.fill(0)
        compute_t_1_amplitudes(
            self.f, self.u, self.t_1, self.o, self.v, out=self.rhs_t_1, np=np
        )

        trial_vector = self.t_1
        direction_vector = np.divide(self.rhs_t_1, self.d_t_1)
        error_vector = self.rhs_t_1.copy()

        self.t_1 = self.t_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

    def compute_l_amplitudes(self):
        np = self.np

        self.rhs_l_1.fill(0)
        compute_l_1_amplitudes(
            self.f,
            self.u,
            self.t_1,
            self.l_1,
            self.o,
            self.v,
            out=self.rhs_l_1,
            np=np,
        )

        trial_vector = self.l_1
        direction_vector = np.divide(self.rhs_l_1, self.d_l_1)
        error_vector = self.rhs_l_1.copy()

        self.l_1 = self.l_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

    def compute_one_body_density_matrix(self):
        return compute_one_body_density_matrix(
            self.t_1, self.l_1, self.o, self.v, np=self.np
        )

    def compute_two_body_density_matrix(self):
        return 0
