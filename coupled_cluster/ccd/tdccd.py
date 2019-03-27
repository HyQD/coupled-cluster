from coupled_cluster.tdcc import TimeDependentCoupledCluster
from coupled_cluster.ccd.rhs_t import compute_t_2_amplitudes
from coupled_cluster.ccd.rhs_l import compute_l_2_amplitudes
from coupled_cluster.ccd.energies import (
    compute_time_dependent_energy,
    compute_ccd_ground_state_energy,
)
from coupled_cluster.ccd.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)
from coupled_cluster.ccd.time_dependent_overlap import (
    compute_time_dependent_overlap,
)
from coupled_cluster.ccd import CoupledClusterDoubles


class TDCCD(TimeDependentCoupledCluster):
    def __init__(self, *args, **kwargs):
        super().__init__(CoupledClusterDoubles, *args, **kwargs)

    def rhs_t_0_amplitude(self, *args, **kwargs):
        return compute_ccd_ground_state_energy(*args, **kwargs)

    def rhs_t_amplitudes(self):
        yield compute_t_2_amplitudes

    def rhs_l_amplitudes(self):
        yield compute_l_2_amplitudes

    def compute_energy(self):
        t_2, l_2 = self._amplitudes.unpack()

        return compute_time_dependent_energy(
            self.f, self.u, t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_one_body_density_matrix(self):
        t_2, l_2 = self._amplitudes.unpack()

        return compute_one_body_density_matrix(
            t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_two_body_density_matrix(self):
        t_2, l_2 = self._amplitudes.unpack()

        return compute_two_body_density_matrix(
            t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_time_dependent_overlap(self):
        t_2, l_2 = self._amplitudes.unpack()

        return compute_time_dependent_overlap(
            self.cc.t_2, self.cc.l_2, t_2, l_2, np=self.np
        )
