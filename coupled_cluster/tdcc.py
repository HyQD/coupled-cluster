import collections
from coupled_cluster.cc_helper import AmplitudeContainer
from coupled_cluster.integrators import RungeKutta4


class TimeDependentCoupledCluster:
    def __init__(
        self,
        rhs_t_func,
        rhs_l_func,
        energy_func,
        system,
        np=None,
        integrator=RungeKutta4,
    ):
        if np is None:
            import numpy as np

        self.np = np
        self.integrator = integrator(self, np=self.np)
        self.system = system
        self.h = self.system.h
        self.u = self.system.u
        self.f = self.system.construct_fock_matrix(self.h, self.u)

        if not isinstance(rhs_t_func, collections.Iterable):
            rhs_t_func = [rhs_t_func]

        if not isinstance(rhs_l_func, collections.Iterable):
            rhs_l_func = [rhs_l_func]

        self.rhs_t_func = rhs_t_func
        self.rhs_l_func = rhs_l_func
        self.energy_func = energy_func

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
