import collections
from coupled_cluster.cc_helper import AmplitudeContainer


class TimeDependentCoupledCluster:
    def __init__(self, rhs_t_func, rhs_l_func, system, np=None):
        if np is None:
            import numpy as np

        self.np = np
        self.system = system

        if not isinstance(rhs_t_func, collections.Iterable):
            rhs_t_func = [rhs_t_func]

        if not isinstance(rhs_l_func, collections.Iterable):
            rhs_l_func = [rhs_l_func]

        self.rhs_t_func = rhs_t_func
        self.rhs_l_func = rhs_l_func

    def rhs(self, prev_amp, current_time):
        o, v = self.system.o, self.system.v

        h = self.system.h_t(current_time)
        u = self.system.u_t(current_time)
        f = self.system.construct_fock_matrix(h, u)

        l_old, t_old = prev_amp

        t_new = [
            -1j * rhs_t_func(f, u, *t_old, o, v, np=self.np)
            for rhs_t_func in self.rhs_t_func
        ]

        l_new = [
            1j * rhs_l_func(f, u, *t_old, *l_old, o, v, np=self.np)
            for rhs_l_func in self.rhs_l_func
        ]

        return AmplitudeContainer(l=l_new, t=t_new)

    def rk4_step(self, u, t, dt):
        f = self.rhs

        K1 = dt * f(u, t)
        K2 = dt * f(u + 0.5 * K1, t + 0.5 * dt)
        K3 = dt * f(u + 0.5 * K2, t + 0.5 * dt)
        K4 = dt * f(u + K3, t + dt)
        u_new = u + (1 / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)

        return u_new
