from coupled_cluster.tdcc import TimeDependentCoupledCluster
from coupled_cluster.ccsd.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)
from coupled_cluster.ccsd.rhs_l import (
    compute_l_1_amplitudes,
    compute_l_2_amplitudes,
)
from coupled_cluster.ccsd import CCSD
from coupled_cluster.ccsd.energies import (
    compute_time_dependent_energy,
    compute_ccsd_ground_state_energy,
)
from coupled_cluster.ccsd.density_matrices import (
    compute_one_body_density_matrix,
)
from coupled_cluster.ccsd.time_dependent_overlap import (
    compute_time_dependent_overlap,
)


class TDCCSD(TimeDependentCoupledCluster):
    def __init__(self, *args, **kwargs):
        super().__init__(CCSD, *args, **kwargs)

    def rhs_t_0_amplitude(self, *args, **kwargs):
        return self.np.array(
            [compute_ccsd_ground_state_energy(*args, **kwargs)]
        )

    def rhs_t_amplitudes(self):
        yield compute_t_1_amplitudes
        yield compute_t_2_amplitudes

    def rhs_l_amplitudes(self):
        yield compute_l_1_amplitudes
        yield compute_l_2_amplitudes

    def left_reference_overlap(self):
        np = self.np

        t_0, t_1, t_2, l_1, l_2 = self._amplitudes.unpack()

        temp = np.einsum("ai, bj -> abij", t_1, t_1)
        temp -= temp.swapaxes(2, 3)
        temp -= temp.swapaxes(0, 1)

        return (
            1
            - 0.25 * np.tensordot(l_2, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
            - np.trace(l_1 @ t_1)
            + 0.125 * np.tensordot(l_2, temp, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
        )

    def compute_energy(self):
        t_0, t_1, t_2, l_1, l_2 = self._amplitudes.unpack()

        return compute_time_dependent_energy(
            self.f,
            self.u,
            t_1,
            t_2,
            l_1,
            l_2,
            self.system.o,
            self.system.v,
            np=self.np,
        )

    def compute_one_body_density_matrix(self):
        t_0, t_1, t_2, l_1, l_2 = self._amplitudes.unpack()
        return compute_one_body_density_matrix(
            t_1, t_2, l_1, l_2, self.o, self.v, np=self.np
        )

    # TODO: Implement this?
    def compute_two_body_density_matrix(self):
        pass

    def compute_time_dependent_overlap(self, use_old=False):
        t_0, t_1, t_2, l_1, l_2 = self._amplitudes.unpack()

        return compute_time_dependent_overlap(
            self.cc.t_1,
            self.cc.t_2,
            self.cc.l_1,
            self.cc.l_2,
            t_0,
            t_1,
            t_2,
            l_1,
            l_2,
            np=self.np,
            use_old=use_old,
        )
