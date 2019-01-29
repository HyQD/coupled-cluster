import abc
from coupled_cluster.cc_helper import OACCVector
from coupled_cluster.tdcc import TimeDependentCoupledCluster
from coupled_cluster.integrator import RungeKutta4


class OATDCC(TimeDependentCoupledCluster, metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of an orbital-adaptive
    time-dependent coupled cluster solver class.

    Note that this solver _only_ supports a basis of orthonomal orbitals. If the
    original atomic orbitals are not orthonormal, this can solved done by
    transforming the ground state orbitals to the Hartree-Fock basis.
    """

    def set_initial_conditions(self, amplitudes=None, C=None, C_tilde=None):
        if amplitudes is None:
            # Create copy of ground state amplitudes for time-integration
            amplitudes = self.cc.get_amplitudes()

        if C is None:
            C = self.np.eye(self.system.l)

        if C_tilde is None:
            C_tilde = self.np.eye(self.system.l)

        self._amplitudes = OACCVector(t, l, C, C_tilde)

    @abc.abstractmethod
    def compute_time_dependent_overlap(self):
        """The time-dependent overlap for orbital-adaptive coupled cluster
        changes due to the time-dependent orbitals in the wavefunctions.
        """
        pass

    @abc.abstractmethod
    def compute_p_space_equations(self):
        pass

    def compute_q_space_ket_equations(self):
        return 0

    def compute_q_space_bra_equations(self):
        return 0

    def __call__(self, prev_amp, current_time):
        t_old, l_old, C, C_tilde = prev_amp

        # Evolve system in time
        self.h = self.system.h_t(current_time)
        self.u = self.system.u_t(current_time)

        # Change basis to C and C_tilde
        # TODO: Consider offloading these transformations to separate functions
        self.h = C_tilde @ self.h @ C
        self.u = np.einsum(
            "pa, qb, cr, ds, abcd -> pqrs",
            C_tilde,
            C_tilde,
            C,
            C,
            self.u,
            optimize=True,
        )
        self.f = self.system.construct_fock_matrix(self.h, self.u)

        # OATDCC procedure:
        # Do amplitude step
        t_new = [
            -1j * rhs_t_func(self.f, self.u, *t_old, o, v, np=self.np)
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        l_new = [
            1j * rhs_l_func(self.f, self.u, *t_old, *l_old, o, v, np=self.np)
            for rhs_l_func in self.rhs_l_amplitudes()
        ]

        # Compute density matrices
        self.rho_qp = self.compute_one_body_density_matrix()
        self.rho_qspr = self.compute_two_body_density_matrix()

        # Solve P-space equations for eta
        eta = self.compute_p_space_equations()

        # Solve Q-space for C and C_tilde
        C_new = -1j * self.compute_q_space_ket_equations()
        C_tilde_new = 1j * self.compute_q_space_bra_equations()

        # Return amplitudes and C and C_tilde
        return OACCVector(t=t_new, l=l_new, C=C_new, C_tilde=C_tilde_new)
