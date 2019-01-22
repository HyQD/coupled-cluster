import collections
from coupled_cluster.cc_helper import AmplitudeContainer


class TimeDependentCoupledCluster:
    def __init__(self, rhs_t_func, rhs_l_func, energy_func, system, np=None):
        if np is None:
            import numpy as np

        self.np = np
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

    def rhs(self, prev_amp, current_time):
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

    def rk4_step(self, u, t, dt):
        f = self.rhs

        K1 = dt * f(u, t)
        K2 = dt * f(u + 0.5 * K1, t + 0.5 * dt)
        K3 = dt * f(u + 0.5 * K2, t + 0.5 * dt)
        K4 = dt * f(u + K3, t + dt)
        u_new = u + (1 / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)

        return u_new
