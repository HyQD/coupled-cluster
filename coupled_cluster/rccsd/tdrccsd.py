from coupled_cluster.tdcc import TimeDependentCoupledCluster
from coupled_cluster.rccsd.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)
from coupled_cluster.rccsd.rhs_l import (
    compute_l_1_amplitudes,
    compute_l_2_amplitudes,
)
from coupled_cluster.rccsd import RCCSD
from coupled_cluster.rccsd.energies import (
    compute_time_dependent_energy,
    compute_ground_state_energy_correction,
)
from coupled_cluster.rccsd.density_matrices import (
    compute_one_body_density_matrix,
)
from coupled_cluster.rccsd.time_dependent_overlap import (
    compute_time_dependent_overlap,
)


class TDRCCSD(TimeDependentCoupledCluster):
    def __init__(self, *args, **kwargs):
        super().__init__(RCCSD, *args, **kwargs)

    def rhs_t_0_amplitude(self, *args, **kwargs):
        return self.np.array(
            [
                self.system.compute_reference_energy()
                + compute_ground_state_energy_correction(*args, **kwargs)
            ]
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

        val = 1
        val -= 0.5 * np.einsum("ijab,abij->", l_2, t_2)
        val += 0.5 * np.einsum("ai,bj,ijab->", t_1, t_1, l_2, optimize=True)
        val -= np.einsum("ia,ai->", l_1, t_1)

        return val

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
