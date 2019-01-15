import collections


class TimeDependentCoupledCluster:
    def __init__(self, rhs_t_func, rhs_l_func, np=None):
        if np is None:
            import numpy as np

        self.np = np

        if not isinstance(rhs_t_func, collections.Iterable):
            rhs_t_func = [rhs_t_func]

        if not isinstance(rhs_l_func, collections.Iterable):
            rhs_l_func = [rhs_l_func]

        self.rhs_t_func = rhs_t_func
        self.rhs_l_func = rhs_l_func

    def rhs(self, f, u, o, v, l, t, current_time):
        # o, v = self.system.o, self.system.v

        # h = self.system.h_t(current_time)
        # u = self.system.u_t(current_time)
        # f = self.system.construct_fock_matrix(h, u)

        t_new = [
            -1j * rhs_t_func(f, u, *t, o, v, np=self.np)
            for rhs_t_func in self.rhs_t_func
        ]

        l_new = [
            1j * rhs_l_func(f, u, *t, *l, o, v, np=self.np)
            for rhs_l_func in self.rhs_l_func
        ]

        return l_new, t_new
