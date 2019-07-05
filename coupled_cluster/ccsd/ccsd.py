from coupled_cluster.cc import CoupledCluster

from coupled_cluster.ccsd.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)

from coupled_cluster.ccsd.rhs_l import (
    compute_l_1_amplitudes,
    compute_l_2_amplitudes,
)

from coupled_cluster.cc_helper import (
    construct_d_t_1_matrix,
    construct_d_t_2_matrix,
)

from coupled_cluster.ccsd.density_matrices import (
    compute_one_body_density_matrix,
)


class CoupledClusterSinglesDoubles(CoupledCluster):
    """Coupled Cluster Singels Doubles

    Coupled Cluster solver with single-, and double
    excitations.

    Parameters
    ----------
    system : QuantumSystems
        QuantumSystems class instance describing the system to be solved
    include_singles : bool
        Include singles
    """

    def __init__(self, system, include_singles=True, **kwargs):
        super().__init__(system, **kwargs)

        np = self.np
        n, m = self.n, self.m

        self.include_singles = include_singles

        # Singles
        self.rhs_t_1 = np.zeros((m, n), dtype=self.u.dtype)  # ai
        self.rhs_l_1 = np.zeros((n, m), dtype=self.u.dtype)  # ia

        self.t_1 = np.zeros_like(self.rhs_t_1)
        self.l_1 = np.zeros_like(self.rhs_l_1)

        self.d_t_1 = construct_d_t_1_matrix(self.f, self.o, self.v, np)
        self.d_l_1 = self.d_t_1.transpose(1, 0).copy()

        # Doubles
        self.rhs_t_2 = np.zeros((m, m, n, n), dtype=self.u.dtype)  # abij
        self.rhs_l_2 = np.zeros((n, n, m, m), dtype=self.u.dtype)  # ijab

        self.t_2 = np.zeros_like(self.rhs_t_2)
        self.l_2 = np.zeros_like(self.rhs_l_2)

        self.d_t_2 = construct_d_t_2_matrix(self.f, self.o, self.v, np)
        self.d_l_2 = self.d_t_2.transpose(2, 3, 0, 1).copy()

        # Mixer
        self.t_mixer = None
        self.l_mixer = None

        # Integrator
        self.t_integrator = None
        self.l_integrator = None

        # Go!
        self.compute_initial_guess()

    def compute_initial_guess(self):
        np = self.np
        o, v = self.o, self.v

        # Singles
        if self.include_singles:
            np.copyto(self.rhs_t_1, self.f[v, o])
            np.divide(self.rhs_t_1, self.d_t_1, out=self.t_1)

            np.copyto(self.rhs_l_1, self.f[o, v])
            np.divide(self.rhs_l_1, self.d_l_1, out=self.l_1)

        # Doubles
        np.copyto(self.rhs_t_2, self.u[v, v, o, o])
        np.divide(self.rhs_t_2, self.d_t_2, out=self.t_2)

        np.copyto(self.rhs_l_2, self.u[o, o, v, v])
        np.divide(self.rhs_l_2, self.d_l_2, out=self.l_2)

    def _get_t_copy(self):
        return [self.t_1.copy(), self.t_2.copy()]

    def _get_l_copy(self):
        return [self.l_1.copy(), self.l_2.copy()]

    def compute_l_residuals(self):
        return [
            self.np.linalg.norm(self.rhs_l_1),
            self.np.linalg.norm(self.rhs_l_2),
        ]

    def compute_t_residuals(self):
        return [
            self.np.linalg.norm(self.rhs_t_1),
            self.np.linalg.norm(self.rhs_t_2),
        ]

    def setup_l_mixer(self, **kwargs):
        if self.l_mixer is None:
            self.l_mixer = self.mixer(**kwargs)

        self.l_mixer.clear_vectors()

    def setup_t_mixer(self, **kwargs):
        if self.t_mixer is None:
            self.t_mixer = self.mixer(**kwargs)

        self.t_mixer.clear_vectors()

    def compute_energy(self):
        """Compute Energy

        Returns
        -------
        float
            Energy of current state
        """
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

    def compute_t_amplitudes(self):
        np = self.np

        trial_vector = np.array([], dtype=self.u.dtype)
        direction_vector = np.array([], dtype=self.u.dtype)
        error_vector = np.array([], dtype=self.u.dtype)

        # Singles
        if self.include_singles:
            self.rhs_t_1.fill(0)
            compute_t_1_amplitudes(
                self.f,
                self.u,
                self.t_1,
                self.t_2,
                self.o,
                self.v,
                out=self.rhs_t_1,
                np=np,
            )

            trial_vector = self.t_1.ravel()
            direction_vector = (self.rhs_t_1 / self.d_t_1).ravel()
            error_vector = -self.rhs_t_1.ravel()

        # Doubles
        self.rhs_t_2.fill(0)
        compute_t_2_amplitudes(
            self.f,
            self.u,
            self.t_1,
            self.t_2,
            self.o,
            self.v,
            out=self.rhs_t_2,
            np=np,
        )

        trial_vector = np.concatenate((trial_vector, self.t_2.ravel()), axis=0)
        direction_vector = np.concatenate(
            (direction_vector, (self.rhs_t_2 / self.d_t_2).ravel()), axis=0
        )
        error_vector = np.concatenate(
            (error_vector, -self.rhs_t_2.ravel()), axis=0
        )

        newvectors = self.t_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

        n_t1 = 0
        if self.include_singles:
            n_t1 = self.m * self.n
            self.t_1 = np.reshape(newvectors[:n_t1], self.t_1.shape)

        self.t_2 = np.reshape(newvectors[n_t1:], self.t_2.shape)

    def compute_l_amplitudes(self):
        np = self.np

        trial_vector = np.array([], dtype=self.u.dtype)  # Empty array
        direction_vector = np.array([], dtype=self.u.dtype)  # Empty array
        error_vector = np.array([], dtype=self.u.dtype)  # Empty array

        # Singles
        if self.include_singles:
            self.rhs_l_1.fill(0)
            compute_l_1_amplitudes(
                self.f,
                self.u,
                self.t_1,
                self.t_2,
                self.l_1,
                self.l_2,
                self.o,
                self.v,
                out=self.rhs_l_1,
                np=np,
            )

            trial_vector = self.l_1.ravel()
            direction_vector = (self.rhs_l_1 / self.d_l_1).ravel()
            error_vector = -self.rhs_l_1.ravel()

        # Doubles
        self.rhs_l_2.fill(0)
        compute_l_2_amplitudes(
            self.f,
            self.u,
            self.t_1,
            self.t_2,
            self.l_1,
            self.l_2,
            self.o,
            self.v,
            out=self.rhs_l_2,
            np=np,
        )

        trial_vector = np.concatenate((trial_vector, self.l_2.ravel()), axis=0)
        direction_vector = np.concatenate(
            (direction_vector, (self.rhs_l_2 / self.d_l_2).ravel()), axis=0
        )
        error_vector = np.concatenate(
            (error_vector, -self.rhs_l_2.ravel()), axis=0
        )

        newvectors = self.l_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

        n_l1 = 0
        if self.include_singles:
            n_l1 = self.m * self.n
            self.l_1 = np.reshape(newvectors[:n_l1], self.l_1.shape)

        self.l_2 = np.reshape(newvectors[n_l1:], self.l_2.shape)

    def compute_one_body_density_matrix(self):
        """Computes one-body density matrix

        Returns
        -------
        np.array
            One-body density matrix
        """

        return compute_one_body_density_matrix(
            self.t_1, self.t_2, self.l_1, self.l_2, self.o, self.v, np=self.np
        )

    def compute_two_body_density_matrix(self):

        pass

    def rhs_t_amplitudes(self):
        if self.include_singles:
            yield compute_t_1_amplitudes
        yield compute_t_2_amplitudes

    def rhs_l_amplitudes(self):
        if self.include_singles:
            yield compute_l_1_amplitudes
        yield compute_l_2_amplitudes

    def t_rhs_der(self, prev_amp, time):
        """Return approximate derivative of rhs
        """

        np = self.np

        if self.include_singles:
            return np.concatenate(
                (self.d_t_1.ravel(), self.d_t_2.ravel()), axis=0
            )
        else:
            return self.d_t_2.ravel()

    def l_rhs_der(self, prev_amp, time):
        """Return approximate derivative of rhs
        """

        np = self.np

        if self.include_singles:
            return np.concatenate(
                (self.d_l_1.ravel(), self.d_l_2.ravel()), axis=0
            )
        else:
            return self.d_l_2.ravel()

    def t_shape(self, u):
        """Takes a flat vector u and returns it as tensors with the shapes of t1 and t2
        """

        np = self.np

        n_u1 = 0
        if self.include_singles:
            n_u1 = self.m * self.n
            u_1 = np.reshape(u[:n_u1], self.t_1.shape)

        u_2 = np.reshape(u[n_u1:], self.t_2.shape)

        if self.include_singles:
            return u_1, u_2
        else:
            return u_2

    def l_shape(self, u):
        """Takes a flat vector u and returns it as tensors with the shapes of l1 and l2
        """

        np = self.np

        n_u1 = 0
        if self.include_singles:
            n_u1 = self.m * self.n
            u_1 = np.reshape(u[:n_u1], self.l_1.shape)

        u_2 = np.reshape(u[n_u1:], self.l_2.shape)

        if self.include_singles:
            return u_1, u_2
        else:
            return u_2

    def u_flat(self, u_1, u_2):
        """Takes two tenors shaped like returns a single flat vector
        """

        np = self.np

        if self.include_singles:
            return np.concatenate((u_1.ravel(), u_2.ravel()), axis=0)
        else:
            return self.u_2.ravel()

    def get_t_flat(self):
        """Return t1 and t2 as a flat vector
        """

        np = self.np

        if self.include_singles:
            return np.concatenate((self.t_1.ravel(), self.t_2.ravel()), axis=0)
        else:
            return self.t_2.ravel()

    def get_l_flat(self):
        """Return l1 and l2 as a flat vector
        """

        np = self.np

        if self.include_singles:
            return np.concatenate((self.l_1.ravel(), self.l_2.ravel()), axis=0)
        else:
            return self.l_2.ravel()

    def update_t_and_rhs(self, t_vector, rhs_vector):
        """update self.t and self.t_rhs with input vectors
        """

        self.t_1, self.t_2 = self.t_shape(t_vector)
        self.rhs_t_1, self.rhs_t_2 = self.t_shape(rhs_vector)

    def update_l_and_rhs(self, l_vector, rhs_vector):
        """update self.t and self.t_rhs with input vectors
        """

        self.l_1, self.l_2 = self.l_shape(l_vector)
        self.rhs_l_1, self.rhs_l_2 = self.l_shape(rhs_vector)

    def get_zero_vec(self):
        """Return a zero vector with length n_t1 + n_t2
        """

        np = self.np

        if self.include_singles:
            return np.zeros(self.t_1.size + self.t_2.size)
        else:
            return np.zeros(self.t_2.size)

    def get_t_amps(self):
        """Return t amplitudes
        """
        return self.t_1, self.t_2

    def get_l_amps(self):
        """Return t amplitudes
        """
        return self.l_1, self.l_2
