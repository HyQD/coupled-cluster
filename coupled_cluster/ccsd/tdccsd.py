from coupled_cluster.tdcc import TimeDependentCoupledCluster
from coupled_cluster.ccsd.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)
from coupled_cluster.ccsd.rhs_l import (
    compute_l_1_amplitudes,
    compute_l_2_amplitudes,
)
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles
from coupled_cluster.ccsd.energies import compute_time_dependent_energy
from coupled_cluster.ccsd.density_matrices import compute_one_body_density_matrix


class TDCCSD(TimeDependentCoupledCluster):
    def __init__(self, *args, **kwargs):
        super().__init__(CoupledClusterSinglesDoubles, *args, **kwargs)

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

    def compute_one_body_density_matrix(self):
        t_1, t_2, l_1, l_2 = self._amplitudes.unpack()
        return compute_one_body_density_matrix(
            t_1, t_2, l_1, l_2, self.o, self.v, np=self.np
        )

    # TODO: Implement these?
    def compute_two_body_density_matrix(self):
        pass

    def compute_time_dependent_overlap(self):
        pass
