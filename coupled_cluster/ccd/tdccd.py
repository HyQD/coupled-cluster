from coupled_cluster.tdcc import TimeDependentCoupledCluster
from coupled_cluster.ccd.rhs_t import compute_t_2_amplitudes
from coupled_cluster.ccd.rhs_l import compute_l_2_amplitudes
from coupled_cluster.ccd.energies import compute_time_dependent_energy
from coupled_cluster.ccd.density_matrices import compute_one_body_density_matrix
from coupled_cluster.ccd.time_dependent_overlap import (
    compute_time_dependent_overlap,
)


class TDCCD(TimeDependentCoupledCluster):
    def rhs_t_amplitudes(self):
        yield compute_t_2_amplitudes

    def rhs_l_amplitudes(self):
        yield compute_l_2_amplitudes

    def compute_energy(self):
        t_2, l_2 = self._amplitudes

        return compute_time_dependent_energy(
            self.f, self.u, t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_one_body_density_matrix(self):
        t_2, l_2 = self._amplitudes

        return compute_one_body_density_matrix(
            t_2, l_2, self.system.o, self.system.v
        )

    def compute_time_dependent_overlap(self):
        t_2, l_2 = self._amplitudes

        return compute_time_dependent_overlap(
            self.cc.t_2, self.cc.l_2, t_2, l_2, np=self.np
        )
