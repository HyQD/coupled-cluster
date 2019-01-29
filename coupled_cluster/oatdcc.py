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

    def __call__(self, prev_amp, current_time):
        t, l, C, C_tilde = prev_amp

        # Evolve system in time
        self.h = self.system.h_t(current_time)
        self.u = self.system.u_t(current_time)

        # Change basis to C and C_tilde
        # TODO: Consider offloading these transformations to separate functions
        self.h = self.C_tilde @ self.h @ self.C
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
        # Compute density matrices
        # Solve P-space equations for eta
        # Solve Q-space for C and C_tilde
        # Transform basis

        # Return amplitudes and C and C_tilde
        # TODO: Create new container for these

        pass
