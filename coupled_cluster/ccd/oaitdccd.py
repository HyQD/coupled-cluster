from . import OATDCCD
from ..cc_helper import OACCVector


class OAITDCCD(OATDCCD):
    def __call__(self, prev_amp, current_time):
        print(1)
        np = self.np
        o, v = self.o, self.v

        prev_amp = OACCVector.from_array(self._amplitudes, prev_amp)
        t_old, l_old, C, C_tilde = prev_amp

        self.update_hamiltonian(current_time, prev_amp)

        # Remove t_0 phase as this is not used in any of the equations
        t_old = t_old[1:]

        # OATDCC procedure:
        # Do amplitude step
        t_new = [
            -rhs_t_func(self.f, self.u, *t_old, o, v, np=self.np)
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        # Compute derivative of phase
        t_0_new = -self.rhs_t_0_amplitude(
            self.f, self.u, *t_old, self.o, self.v, np=self.np
        )

        t_new = [t_0_new, *t_new]

        l_new = [
            -rhs_l_func(self.f, self.u, *t_old, *l_old, o, v, np=self.np)
            for rhs_l_func in self.rhs_l_amplitudes()
        ]

        # Compute density matrices
        self.rho_qp = self.one_body_density_matrix(t_old, l_old)
        self.rho_qspr = self.two_body_density_matrix(t_old, l_old)

        # Solve P-space equations for eta
        # divide by imaginary number
        eta = -1j * self.compute_p_space_equations()
        # TODO: move 1j out of compute_p_space_equations

        # Compute the inverse of rho_qp needed in Q-space eqs.
        """
        If rho_qp is singular we can regularize it as,

        rho_qp_reg = rho_qp + eps*expm( -(1.0/eps) * rho_qp) Eq [3.14]
        Multidimensional Quantum Dynamics, Meyer

        with eps = 1e-8 (or some small number). It seems like it is standard in
        the MCTDHF literature to always work with the regularized rho_qp. Note
        here that expm refers to the matrix exponential which I can not find in
        numpy only in scipy.
        """
        # rho_pq_inv = self.np.linalg.inv(self.rho_qp)

        # Solve Q-space for C and C_tilde
        # C_new = np.dot(C, eta)
        # C_tilde_new = -np.dot(eta, C_tilde)

        C_new = -compute_q_space_ket_equations(
            C,
            C_tilde,
            eta,
            self.h_orig,
            self.h,
            self.u_orig,
            self.u,
            rho_pq_inv,
            self.rho_qspr,
            np=np,
        )
        C_tilde_new = -compute_q_space_bra_equations(
            C,
            C_tilde,
            eta,
            self.h_orig,
            self.h,
            self.u_orig,
            self.u,
            rho_pq_inv,
            self.rho_qspr,
            np=np,
        )

        self.last_timestep = current_time

        # Return amplitudes and C and C_tilde
        return OACCVector(
            t=t_new, l=l_new, C=C_new, C_tilde=C_tilde_new, np=self.np
        ).asarray()
