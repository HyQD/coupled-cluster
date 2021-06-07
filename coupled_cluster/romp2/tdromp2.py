from coupled_cluster.romp2.rhs_t import (
    compute_t_2_amplitudes,
    compute_l_2_amplitudes,
)


from coupled_cluster.romp2.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)

from coupled_cluster.romp2.p_space_equations import compute_eta

from coupled_cluster.cc_helper import OACCVector
from coupled_cluster.cc_helper import AmplitudeContainer

from coupled_cluster.oatdcc import OATDCC

from opt_einsum import contract


class TDROMP2(OATDCC):
    """Time-dependent orbital-optimized second-order Møller-Plesset perturbation theory (TDOMP2)

    Parameters
    ----------
    system : QuantumSystem
        QuantumSystem class instance description of system

    References
    ----------
    .. [1] H. Pathak, T. Sato, K. Ishikawa
          "Time-dependent optimized coupled-cluster method for multielectron dynamics. III.
          A second-order many-body perturbation approximation", J. Chem. Phys. 153, 034110, 2020.

    """

    truncation = "CCD"

    def rhs_t_0_amplitude(self, *args, **kwargs):
        return self.np.array([0 + 0j])

    def rhs_t_amplitudes(self):
        yield compute_t_2_amplitudes

    def rhs_l_amplitudes(self):
        yield compute_l_2_amplitudes

    def compute_left_reference_overlap(self, current_time, y):
        t_0, t_2, l_2, _, _ = self._amp_template.from_array(y).unpack()

        return 1 - 0.25 * self.np.tensordot(
            l_2, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))
        )

    def compute_energy(self, current_time, y):

        t_0, t_2, l_2, C, C_tilde = self._amp_template.from_array(y).unpack()
        self.update_hamiltonian(current_time=current_time, y=y)

        rho_qp = self.compute_one_body_density_matrix(current_time, y)
        rho_qspr = self.compute_two_body_density_matrix(current_time, y)

        return (
            contract("pq,qp->", self.h_prime, rho_qp, optimize=True)
            + 0.5
            * contract("pqrs,rspq->", self.u_prime, rho_qspr, optimize=True)
            + self.system.nuclear_repulsion_energy
        )

    def one_body_density_matrix(self, t, l):
        t_2 = t[0]
        l_2 = l[0]

        return compute_one_body_density_matrix(
            t_2, l_2, self.o, self.v, np=self.np
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
            t_2, l_2, self.o, self.v, np=self.np, out=self.rho_qspr
        )

    def compute_one_body_density_matrix(self, current_time, y):
        t_0, t_2, l_2, _, _ = self._amp_template.from_array(y).unpack()

        return compute_one_body_density_matrix(
            t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_two_body_density_matrix(self, current_time, y):
        t_0, t_2, l_2, _, _ = self._amp_template.from_array(y).unpack()

        return compute_two_body_density_matrix(
            t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_overlap(self, current_time, y_a, y_b):
        """
        Computes time dependent overlap with respect to a given cc-state
        """
        t0a, t2a, l2a, _, _ = self._amp_template.from_array(y_a).unpack()
        t0b, t2b, l2b, _, _ = self._amp_template.from_array(y_b).unpack()

        return compute_orbital_adaptive_overlap(t2a, l2a, t2b, l2b, np=self.np)

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
