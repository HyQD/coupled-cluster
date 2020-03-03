from coupled_cluster.oatdcc import OATDCC
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
    compute_orbital_adaptive_time_dependent_overlap,
)
from coupled_cluster.ccd.p_space_equations import compute_eta
from coupled_cluster.ccd import OACCD


class OATDCCD(OATDCC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def rhs_t_0_amplitude(self, *args, **kwargs):
        return self.np.array([compute_ccd_ground_state_energy(*args, **kwargs)])

    def rhs_t_amplitudes(self):
        yield compute_t_2_amplitudes

    def rhs_l_amplitudes(self):
        yield compute_l_2_amplitudes

    def left_reference_overlap(self):
        t_0, t_2, l_2, _, _ = self._amplitudes.unpack()

        return 1 - 0.25 * self.np.tensordot(
            l_2, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))
        )

    def compute_energy(self):
        t_0, t_2, l_2, _, _ = self._amplitudes.unpack()
        return compute_time_dependent_energy(
            self.f_prime,
            self.u_prime,
            t_2,
            l_2,
            self.o_prime,
            self.v_prime,
            np=self.np,
        )

    def one_body_density_matrix(self, t, l):
        t_2 = t[0]
        l_2 = l[0]

        return compute_one_body_density_matrix(
            t_2, l_2, self.o_prime, self.v_prime, np=self.np
        )

    def two_body_density_matrix(self, t, l):
        t_2 = t[0]
        l_2 = l[0]

        # Avoid re-allocating memory for two-body density matrix
        if not hasattr(self, "rho_qspr"):
            self.rho_qspr = None
        else:
            self.rho_qspr.fill(0)

        return compute_two_body_density_matrix(
            t_2, l_2, self.o_prime, self.v_prime, np=self.np, out=self.rho_qspr
        )

    def compute_one_body_density_matrix(self):
        t_0, t_2, l_2, _, _ = self._amplitudes.unpack()

        return compute_one_body_density_matrix(
            t_2, l_2, self.o_prime, self.v_prime, np=self.np
        )

    def compute_two_body_density_matrix(self):
        t_0, t_2, l_2, _, _ = self._amplitudes.unpack()

        return compute_two_body_density_matrix(
            t_2, l_2, self.o_prime, self.v_prime, np=self.np
        )

    def compute_time_dependent_overlap(self, cc):
        """
        Computes time dependent overlap with respect to a given cc-state
        """
        t_0, t_2, l_2, _, _ = self._amplitudes.unpack()

        return compute_orbital_adaptive_time_dependent_overlap(
            cc.t_2, cc.l_2, t_2, l_2, np=self.np
        )

    def compute_p_space_equations(self):
        eta = compute_eta(
            self.h_prime,
            self.u_prime,
            self.rho_qp,
            self.rho_qspr,
            self.o_prime,
            self.v_prime,
            np=self.np,
        )

        return eta
