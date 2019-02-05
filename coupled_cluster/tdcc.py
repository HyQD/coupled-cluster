import abc
import collections
from coupled_cluster.cc_helper import AmplitudeContainer
from coupled_cluster.integrators import RungeKutta4


class TimeDependentCoupledCluster(metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of a time-dependent Coupled
    Cluster solver class.
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

    def compute_ground_state(
        self, t_args=[], t_kwargs={}, l_args=[], l_kwargs={}
    ):
        # Compute ground state amplitudes
        self.cc.iterate_t_amplitudes(*t_args, **t_kwargs)
        self.cc.iterate_l_amplitudes(*l_args, **l_kwargs)

    def set_initial_conditions(self, amplitudes=None):
        if amplitudes is None:
            # Create copy of ground state amplitudes for time-integration
            amplitudes = self.cc.get_amplitudes()

        self._amplitudes = amplitudes

    @property
    def amplitudes(self):
        return self._amplitudes

    def solve(self, time_points):
        n = len(time_points)

        for i in range(n - 1):
            dt = time_points[i + 1] - time_points[i]
            amp_vec = self.integrator.step(
                self._amplitudes.asarray(), time_points[i], dt
            )

            self._amplitudes = type(self._amplitudes).from_array(
                self._amplitudes, amp_vec
            )

            yield self._amplitudes

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

    def __call__(self, prev_amp, current_time):
        o, v = self.system.o, self.system.v

        self.h = self.system.h_t(current_time)
        self.u = self.system.u_t(current_time)
        self.f = self.system.construct_fock_matrix(self.h, self.u)

        prev_amp = AmplitudeContainer.from_array(self._amplitudes, prev_amp)
        t_old, l_old = prev_amp

        t_new = [
            -1j * rhs_t_func(self.f, self.u, *t_old, o, v, np=self.np)
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        l_new = [
            1j * rhs_l_func(self.f, self.u, *t_old, *l_old, o, v, np=self.np)
            for rhs_l_func in self.rhs_l_amplitudes()
        ]

        return AmplitudeContainer(t=t_new, l=l_new, np=self.np).asarray()
