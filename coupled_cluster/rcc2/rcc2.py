from coupled_cluster.cc import CoupledCluster

from coupled_cluster.rcc2.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)

from coupled_cluster.rcc2.rhs_l import (
    compute_l_1_amplitudes,
    compute_l_2_amplitudes,
)

from coupled_cluster.cc_helper import (
    construct_d_t_1_matrix,
    construct_d_t_2_matrix,
)

from coupled_cluster.rcc2.density_matrices import (
    compute_one_body_density_matrix,
)

from coupled_cluster.rcc2.energies import (
    compute_ground_state_energy_correction,
)


class RCC2(CoupledCluster):
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

        n, m = self.o.stop - self.o.start, self.m
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

        e_corr = compute_ground_state_energy_correction(
            self.f, self.u, self.t_1, self.t_2, self.o, self.v, self.np
        )
        e_ref = self.system.compute_reference_energy()

        return e_corr + e_ref

    def compute_reference_energy(self, h, u, o, v, np):
        """
        Computes reference energy
        Parameters
        ----------
        f: np.array
            One-particle operator (Fock matrix)
        u: np.array
            Two-particle operator
        o: Slice
            Occupied orbitals
        v: Slice
            Virtual orbitals
        np: Module
            Matrix library
        Returns
        -------
        np.float
            Reference energy
        """
        return (
            2 * self.np.trace(h[o, o])  # Changed from h to f
            + 2 * self.np.trace(self.np.trace(u[o, o, o, o], axis1=1, axis2=3))
            - self.np.trace(self.np.trace(u[o, o, o, o], axis1=1, axis2=2))
        )

    def compute_t_amplitudes(self):
        np = self.np

        trial_vector = np.array([], dtype=self.u.dtype)
        direction_vector = np.array([], dtype=self.u.dtype)
        error_vector = np.array([], dtype=self.u.dtype)

        (
            self.h_transform,
            self.f_transform,
            self.u_transform,
        ) = self.t1_transform_integrals(self.t_1, self.system.h, self.system.u)

        # Singles
        if self.include_singles:
            self.rhs_t_1.fill(0)
            self.rhs_t_1 = compute_t_1_amplitudes(
                self.f,
                self.f_transform,
                self.u_transform,
                self.t_1,
                self.t_2,
                self.o,
                self.v,
                out=self.rhs_t_1,
                np=np,
            )
            trial_vector = self.t_1.ravel()
            direction_vector = (self.rhs_t_1 / self.d_t_1).ravel()
            error_vector = self.rhs_t_1.ravel().copy()

        # Doubles
        self.rhs_t_2.fill(0)
        self.rhs_t_2 = compute_t_2_amplitudes(
            self.f,
            self.f_transform,
            self.u_transform,
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
            (error_vector, self.rhs_t_2.ravel().copy()), axis=0
        )

        new_vectors = self.t_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

        if self.include_singles:
            n_t1 = self.m * (self.o.stop - self.o.start)
            self.t_1 = np.reshape(new_vectors[:n_t1], self.t_1.shape)

        self.t_2 = np.reshape(new_vectors[n_t1:], self.t_2.shape)

    def compute_l_amplitudes(self):
        np = self.np

        trial_vector = np.array([], dtype=self.u.dtype)  # Empty array
        direction_vector = np.array([], dtype=self.u.dtype)  # Empty array
        error_vector = np.array([], dtype=self.u.dtype)  # Empty array

        """
        The transposes of l1 and l2 are just a temporary solution until 
        we switch it in the generated right hand sides.
        """

        # Singles
        if self.include_singles:
            self.rhs_l_1.fill(0)
            self.rhs_l_1 = compute_l_1_amplitudes(
                self.f,
                self.f_transform,
                self.u_transform,
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
            error_vector = self.rhs_l_1.ravel().copy()

        # Doubles
        self.rhs_l_2.fill(0)
        self.rhs_l_2 = compute_l_2_amplitudes(
            self.f,
            self.f_transform,
            self.u_transform,
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
            (error_vector, self.rhs_l_2.ravel().copy()), axis=0
        )

        new_vectors = self.l_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

        if self.include_singles:
            n_l1 = self.m * (self.o.stop - self.o.start)
            self.l_1 = np.reshape(new_vectors[:n_l1], self.l_1.shape)

        self.l_2 = np.reshape(new_vectors[n_l1:], self.l_2.shape)

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

    def t1_transform_integrals(self, t_1, h, u):

        np = self.np

        tot = self.m + self.n

        t1_t = self.np.zeros((tot, tot), dtype=t_1.dtype)
        t1_t[self.n : self.n + t_1.shape[0], 0 : t_1.shape[1]] = t_1

        x_transform = np.eye(tot) - t1_t
        y_transform = np.eye(tot) + t1_t.T
        C_tilde = x_transform
        C = y_transform.T

        h_transform = self.system.transform_one_body_elements(h, C, C_tilde)
        u_transform = self.system.transform_two_body_elements(u, C, C_tilde)

        f_transform = self.system.construct_fock_matrix(
            h_transform, u_transform
        )

        return h_transform, f_transform, u_transform

    def t1_transform_integrals_one_body(self, dipole):

        np = self.np

        tot = self.m + self.n

        t_1 = self.t_1
        t1_t = self.np.zeros((tot, tot), dtype=t_1.dtype)
        t1_t[self.n : self.n + t_1.shape[0], 0 : t_1.shape[1]] = t_1

        x_transform = np.eye(tot) - t1_t
        y_transform = np.eye(tot) + t1_t.T
        C_tilde = x_transform
        C = y_transform.T

        d_transform = self.system.transform_one_body_elements(
            dipole, C, C_tilde
        )

        return d_transform
