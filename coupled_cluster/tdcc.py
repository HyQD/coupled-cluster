import abc
import collections
import warnings
from coupled_cluster.cc_helper import AmplitudeContainer
from coupled_cluster.integrators import RungeKutta4


class TimeDependentCoupledCluster(metaclass=abc.ABCMeta):
    """Time Dependent Coupled Cluster Parent Class

    Abstract base class defining the skeleton of a time-dependent Coupled
    Cluster solver class.

    Parameters
    ----------
    cc : CoupledCluster
        Class instance defining the ground state solver
    system : QuantumSystem
        Class instance defining the system to be solved
    np : module
        Matrix/linear algebra library to be uses, like numpy or cupy
    integrator : Integrator
        Integrator class instance (RK4, GaussIntegrator)
    """

    def __init__(self, cc, system, np=None, integrator=None, **cc_kwargs):
        if np is None:
            import numpy as np

        self.np = np

        if not "np" in cc_kwargs:
            cc_kwargs["np"] = self.np

        # Initialize ground state solver
        self.cc = cc(system, **cc_kwargs)
        self.system = system

        self.h = self.system.h
        self.u = self.system.u
        self.f = self.system.construct_fock_matrix(self.h, self.u)
        self.o = self.system.o
        self.v = self.system.v

        if integrator is None:
            integrator = RungeKutta4(np=self.np)

        self.integrator = integrator.set_rhs(self)
        self._amplitudes = None

        # Inherit functions from ground state solver
        self.compute_ground_state_energy = self.cc.compute_energy
        self.compute_ground_state_reference_energy = (
            self.cc.compute_reference_energy
        )
        self.compute_ground_state_particle_density = (
            self.cc.compute_particle_density
        )
        self.compute_ground_state_one_body_density_matrix = (
            self.cc.compute_one_body_density_matrix
        )

    def compute_ground_state(self, *args, **kwargs):
        """Calls on method from CoupledCluster class to compute
        ground state of system.
        """

        # Compute ground state amplitudes
        self.cc.compute_ground_state(*args, **kwargs)

    def set_initial_conditions(self, amplitudes=None):
        """Set initial condition of system.

        Necessary to call this function befor computing
        time development. Can be passed without arguments,
        will revert to amplitudes of ground state solver.

        Parameters
        ----------
        amplitudes : AmplitudeContainer
            Amplitudes for the system
        """

        if amplitudes is None:
            # Create copy of ground state amplitudes for time-integration
            amplitudes = self.cc.get_amplitudes(get_t_0=True)

        self._amplitudes = amplitudes

    @property
    def amplitudes(self):
        return self._amplitudes

    def solve(self, time_points, timestep_tol=1e-8):
        n = len(time_points)

        for i in range(n - 1):
            dt = time_points[i + 1] - time_points[i]
            amp_vec = self.integrator.step(
                self._amplitudes.asarray(), time_points[i], dt
            )

            self._amplitudes = type(self._amplitudes).from_array(
                self._amplitudes, amp_vec
            )

            if abs(self.last_timestep - (time_points[i] + dt)) > timestep_tol:
                self.update_hamiltonian(time_points[i] + dt, self._amplitudes)
                self.last_timestep = time_points[i] + dt

            yield self._amplitudes

    @abc.abstractmethod
    def rhs_t_0_amplitude(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def rhs_t_amplitudes(self):
        """Function that needs to be implemented as a generator. The generator
        should return the t-amplitudes right hand sides, in order of increasing
        excitation. For example, for ccsd, this function should contain:

            yield compute_t_1_amplitudes
            yield compute_t_2_amplitudes
        """
        pass

    @abc.abstractmethod
    def rhs_l_amplitudes(self):
        """Function that needs to be implemented as a generator. The generator
        should return the l-amplitudes right hand sides, in order of increasing
        excitation. For example, for ccsd, this function should contain:

            yield compute_l_1_amplitudes
            yield compute_l_2_amplitudes
        """
        pass

    @abc.abstractmethod
    def compute_energy(self):
        pass

    @abc.abstractmethod
    def compute_one_body_density_matrix(self):
        pass

    @abc.abstractmethod
    def compute_two_body_density_matrix(self):
        pass

    @abc.abstractmethod
    def compute_time_dependent_overlap(self):
        pass

    def compute_right_phase(self):
        r"""Function computing the inner product of the (potentially
        time-dependent) reference state and the right coupled-cluster wave
        function.
        That is,

        .. math:: \langle \Phi \rvert \Psi(t) \rangle = \exp(\tau_0),

        where :math:`\tau_0` is the zeroth cluster amplitude.

        Returns
        -------
        complex128
            The right-phase describing the weight of the reference determinant.
        """
        t_0 = self._amplitudes.t[0][0]

        return self.np.exp(t_0)

    def compute_left_phase(self):
        r"""Function computing the inner product of the (potentially
        time-dependent) reference state and the left coupled-cluster wave
        function.
        That is,

        .. math:: \langle \tilde{\Psi}(t) \rvert \Phi \rangle
            = \exp(-\tau_0)[1 - \langle \Phi \rvert \hat{\Lambda}(t) \hat{T}(t)
            \lvert \Phi \rangle],

        where :math:`\tau_0` is the zeroth cluster amplitude.

        Returns
        -------
        complex128
            The left-phase describing the weight of the reference determinant.
        """
        t_0 = self._amplitudes.t[0][0]

        return self.np.exp(-t_0) * self.left_reference_overlap()

    def compute_reference_weight(self):
        r"""Function computing the weight of the reference state in the
        time-evolved coupled-cluster wave function. This is given by

        .. math:: W(t) = \vert \langle \tilde{\Psi}(t) \rvert \Phi \rangle^{*}
        + \langle \Phi \rvert \Psi(t) \rangle \vert^2,

        where the inner-products are the left- and right-phase expressions.

        Returns
        -------
        complex128
            The weight of the reference state in the time-evolved wave function.
        """

        return (
            self.np.abs(
                self.compute_right_phase() + self.compute_left_phase().conj()
            )
            ** 2
        )

    @abc.abstractmethod
    def left_reference_overlap(self):
        pass

    def compute_particle_density(self):
        """Computes current one-body density

        Returns
        -------
        np.array
            One-body density of system at current time step
        """
        np = self.np

        rho_qp = self.compute_one_body_density_matrix()

        if np.abs(np.trace(rho_qp) - self.system.n) > 1e-8:
            warn = "Trace of rho_qp = {0} != {1} = number of particles"
            warn = warn.format(np.trace(rho_qp), self.system.n)
            warnings.warn(warn)

        return self.system.compute_particle_density(rho_qp)

    def update_hamiltonian(self, current_time, amplitudes):
        if self.system.has_one_body_time_evolution_operator:
            self.h = self.system.h_t(current_time)

        if self.system.has_two_body_time_evolution_operator:
            self.u = self.system.u_t(current_time)

        self.f = self.system.construct_fock_matrix(self.h, self.u)

    def __call__(self, prev_amp, current_time):
        o, v = self.system.o, self.system.v

        prev_amp = AmplitudeContainer.from_array(self._amplitudes, prev_amp)
        t_old, l_old = prev_amp

        self.update_hamiltonian(current_time, prev_amp)

        # Remove phase from t-amplitude list
        t_old = t_old[1:]

        t_new = [
            -1j * rhs_t_func(self.f, self.u, *t_old, o, v, np=self.np)
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        # Compute derivative of phase
        t_0_new = -1j * self.rhs_t_0_amplitude(
            self.f, self.u, *t_old, self.o, self.v, np=self.np
        )
        t_new = [t_0_new, *t_new]

        l_new = [
            1j * rhs_l_func(self.f, self.u, *t_old, *l_old, o, v, np=self.np)
            for rhs_l_func in self.rhs_l_amplitudes()
        ]

        self.last_timestep = current_time

        return AmplitudeContainer(t=t_new, l=l_new, np=self.np).asarray()
