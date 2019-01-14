import collections


class TimeDependentCoupledCluster:
    def __init__(self, system, rhs_t, rhs_l, np=None):
        self.system = self.system

        if np is None:
            import numpy as np

        self.np = np

        if not isinstance(rhs_t, collections.Iterable):
            self.rhs_t = [rhs_t]

        if not isinstance(rhs_l, collections.Iterable):
            self.rhs_l = [rhs_l]

    def rhs(self, l, t, current_time):
        o, v = self.system.o, self.system.v

        h = self.system.h_t(current_time)
        u = self.system.u_t(current_time)
        f = self.system.construct_fock_matrix(h, u)

        t_new = [
            -1j * rhs_t(f, u, *t, o, v, np=self.np) for rhs_t in self.rhs_t
        ]

        l_new = [
            1j * rhs_l(f, u, *t, *l, o, v, np=self.np) for rhs_l in self.rhs_l
        ]

        return l_new, t_new
