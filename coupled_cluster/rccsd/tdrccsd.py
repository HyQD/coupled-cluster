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
    compute_rccsd_correlation_energy,
)
from coupled_cluster.rccsd.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)
from coupled_cluster.rccsd.time_dependent_overlap import (
    compute_time_dependent_overlap,
)

from opt_einsum import contract


class TDRCCSD(TimeDependentCoupledCluster):
    truncation = "CCSD"

    def rhs_t_0_amplitude(self, *args, **kwargs):
        return self.np.array(
            [
                self.system.compute_reference_energy()
                + compute_rccsd_correlation_energy(*args, **kwargs)
            ]
        )

    def rhs_t_amplitudes(self):
        yield compute_t_1_amplitudes
        yield compute_t_2_amplitudes

    def rhs_l_amplitudes(self):
        yield compute_l_1_amplitudes
        yield compute_l_2_amplitudes

    def compute_left_reference_overlap(self, current_time, y):
        np = self.np

        t_0, t_1, t_2, l_1, l_2 = self._amp_template.from_array(y).unpack()

        val = 1
        val -= 0.5 * contract("ijab,abij->", l_2, t_2)
        val += 0.5 * contract("ai,bj,ijab->", t_1, t_1, l_2, optimize=True)
        val -= contract("ia,ai->", l_1, t_1)

        return val

    def compute_energy(self, current_time, y):
        t_0, t_1, t_2, l_1, l_2 = self._amp_template.from_array(y).unpack()

        self.update_hamiltonian(current_time, y)

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

    def compute_one_body_density_matrix(self, current_time, y):
        t_0, t_1, t_2, l_1, l_2 = self._amp_template.from_array(y).unpack()
        return compute_one_body_density_matrix(
            t_1, t_2, l_1, l_2, self.o, self.v, np=self.np
        )

    # TODO: Implement this?
    def compute_two_body_density_matrix(self, current_time, y):
        t_0, t_1, t_2, l_1, l_2 = self._amp_template.from_array(y).unpack()
        return compute_two_body_density_matrix(
            t_1, t_2, l_1, l_2, self.o, self.v, np=self.np
        )

    def compute_overlap(self, current_time, y_a, y_b, use_old=False):
        t0a, t1a, t2a, l1a, l2a = self._amp_template.from_array(y_a).unpack()
        t0b, t1b, t2b, l1b, l2b = self._amp_template.from_array(y_b).unpack()

        return compute_time_dependent_overlap(
            t1a,
            t2a,
            l1a,
            l2a,
            t0b,
            t1b,
            t2b,
            l1b,
            l2b,
            np=self.np,
            use_old=use_old,
        )
