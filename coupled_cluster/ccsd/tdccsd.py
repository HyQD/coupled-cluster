from coupled_cluster.tdcc import TimeDependentCoupledCluster
from coupled_cluster.ccsd.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)
from coupled_cluster.ccsd.rhs_l import (
    compute_l_1_amplitudes,
    compute_l_2_amplitudes,
)


class TDCCSD(TimeDependentCoupledCluster):
    def rhs_t_amplitudes(self):
        yield compute_t_1_amplitudes
        yield compute_t_2_amplitudes

    def rhs_l_amplitudes(self):
        yield compute_l_1_amplitudes
        yield compute_l_2_amplitudes

    def compute_energy(self):
        t, l = self._amplitudes

        return compute_time_dependent_energy(
            self.f, self.u, *t, *l, self.system.o, self.system.v, np=self.np
        )
