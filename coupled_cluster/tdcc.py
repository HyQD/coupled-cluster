import abc
import collections
from coupled_cluster.cc_helper import AmplitudeContainer
from coupled_cluster.integrators import RungeKutta4


class TimeDependentCoupledCluster(metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of a time-dependent Coupled
    Cluster solver class.
    """

    def __init__(
        self, cc, system, np=None, integrator=RungeKutta4, **cc_kwargs
    ):
        if np is None:
            import numpy as np

        self.np = np

        if not "np" in cc_kwargs:
            cc_kwargs["np"] = self.np

        # Initialize ground state solver
        self.cc = cc(system, **cc_kwargs)
        self.system = system
        self.integrator = integrator(self, np=self.np)

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

    def compute_initial_state(
        self, t_args=[], t_kwargs={}, l_args=[], l_kwargs={}
    ):
        # Compute ground state amplitudes
        self.cc.compute_t_amplitudes(*t_args, **t_kwargs)
        self.cc.compute_l_amplitudes(*l_args, **l_kwargs)

        # Create copy of ground state amplitudes
        self.u = self.cc.get_amplitudes()

    def compute_time_dependent_energy(self, amplitudes):
        return self.energy_func(
            self.f,
            self.u,
            *amplitudes.unpack(),
            self.system.o,
            self.system.v,
            np=self.np
        )

    def __call__(self, prev_amp, current_time):
        o, v = self.system.o, self.system.v

        self.h = self.system.h_t(current_time)
        self.u = self.system.u_t(current_time)
        self.f = self.system.construct_fock_matrix(self.h, self.u)

        t_old, l_old = prev_amp

        t_new = [
            -1j * rhs_t_func(self.f, self.u, *t_old, o, v, np=self.np)
            for rhs_t_func in self.rhs_t_func
        ]

        l_new = [
            1j * rhs_l_func(self.f, self.u, *t_old, *l_old, o, v, np=self.np)
            for rhs_l_func in self.rhs_l_func
        ]

        return AmplitudeContainer(t=t_new, l=l_new)

    def step(self, u, t, dt):
        return self.integrator.step(u, t, dt)
