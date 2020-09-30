from coupled_cluster.tdocc import TDOCC
from coupled_cluster.omp2.rhs_t import compute_t_2_amplitudes
from coupled_cluster.omp2.energies import (
    compute_time_dependent_energy,
    compute_ccd_ground_state_energy,
)
from coupled_cluster.omp2.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)

from coupled_cluster.cc_helper import OACCVector


class TDOMP2(TDOCC):
    def __call__(self, current_time, prev_amp):
        np = self.np
        o_prime, v_prime = self.o_prime, self.v_prime

        prev_amp = self._amp_template.from_array(prev_amp)
        t_old, l_old, C, C_tilde = prev_amp

        self.update_hamiltonian(current_time, C=C, C_tilde=C_tilde)

        # Remove t_0 phase as this is not used in any of the equations
        t_old = t_old[1:]

        # OATDCC procedure:
        # Do amplitude step
        t_new = [
            -1j
            * rhs_t_func(
                self.f_prime, self.u_prime, *t_old, o_prime, v_prime, np=self.np
            )
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        # Compute derivative of phase
        t_0_new = -1j * self.rhs_t_0_amplitude(
            self.f_prime, self.u_prime, *t_old, o_prime, v_prime, np=self.np
        )

        t_new = [t_0_new, *t_new]

        # Compute density matrices
        o, v = self.o, self.v

        self.rho_qp = self.one_body_density_matrix(t_old, l_old)
        self.rho_qspr = self.two_body_density_matrix(t_old, l_old)

        opdm = 0.5 * (self.rho_qp + self.rho_qp.T.conj())
        tpdm = 0.5 * (self.rho_qspr + self.rho_qspr.T.conj())

        # Eq. (23) in: https://aip.scitation.org/doi/10.1063/1.5020633

        R_ai = np.einsum("aj,ji->ai", self.h_prime[v, o], opdm[o, o])
        R_ai += 0.5 * np.einsum(
            "arqs,qsir->ai", self.u_prime[v, :, :, :], tpdm[:, :, o, :]
        )
        R_ai -= np.einsum("bi,ab->ai", self.h_prime[v, o], opdm[v, v])
        R_ai -= 0.5 * np.einsum(
            "arqs, qsir->ai", tpdm[v, :, :, :], self.u_prime[:, :, o, :]
        )

        # Solve P-space equations for X^b_j
        delta_ij = self.np.eye(o.stop)
        delta_ba = self.np.eye(v.stop - o.stop)

        A_aibj = self.np.einsum("ab, ji -> aibj", delta_ba, opdm[o, o])
        A_aibj -= self.np.einsum("ji, ab -> aibj", delta_ij, opdm[v, v])

        X_bj = -1j * self.np.linalg.tensorsolve(A_aibj, R_ai)
        X = np.zeros((self.system.l, self.system.l), dtype=C.dtype)

        X[v, o] = X_bj
        X[o, v] = -X_bj.T.conj()

        C_new = np.dot(C, X)
        C_tilde_new = C_new.T.conj()

        self.last_timestep = current_time

        # Return amplitudes and C and C_tilde
        return OACCVector(
            t=t_new, l=l_old, C=C_new, C_tilde=C_tilde_new, np=self.np
        ).asarray()

    def rhs_t_0_amplitude(self, *args, **kwargs):
        return self.np.array([compute_ccd_ground_state_energy(*args, **kwargs)])

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

        return self.np.einsum(
            "pq,pq->", self.h_prime, rho_qp, optimize=True
        ) + 0.25 * self.np.einsum(
            "pqrs,pqrs->", self.u_prime, rho_qspr, optimize=True
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

    def compute_one_body_density_matrix(self, current_time, y):
        t_0, t_2, l_2, _, _ = self._amp_template.from_array(y).unpack()

        return compute_one_body_density_matrix(
            t_2, l_2, self.o_prime, self.v_prime, np=self.np
        )

    def compute_two_body_density_matrix(self, current_time, y):
        t_0, t_2, l_2, _, _ = self._amp_template.from_array(y).unpack()

        return compute_two_body_density_matrix(
            t_2, l_2, self.o_prime, self.v_prime, np=self.np
        )

    def compute_overlap(self, current_time, y_a, y_b):
        """
        Computes time dependent overlap with respect to a given cc-state
        """
        t0a, t2a, l2a, _, _ = self._amp_template.from_array(y_a).unpack()
        t0b, t2b, l2b, _, _ = self._amp_template.from_array(y_b).unpack()

        return compute_orbital_adaptive_overlap(t2a, l2a, t2b, l2b, np=self.np)
