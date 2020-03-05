from . import OATDCCD
from ..oatdcc import (
    compute_q_space_ket_equations,
    compute_q_space_bra_equations,
)
from ..cc_helper import OACCVector


class OAITDCCD(OATDCCD):
    def compute_ground_state(self, *args, **kwargs):
        if "change_system_basis" not in kwargs:
            kwargs["change_system_basis"] = True

        # self.cc.compute_ground_state(*args, **kwargs)
        self.h_orig = self.system.h
        self.u_orig = self.system.u

        self.h = self.system.h
        self.u = self.system.u
        self.f = self.system.construct_fock_matrix(self.h, self.u)

    def solve(self, time_points, timestep_tol=1e-8):

        n = len(time_points)

        for i in range(n - 1):
            dt = time_points[i + 1] - time_points[i]
            amp_vec = self.integrator.step(
                self._amplitudes.asarray(), time_points[i], dt
            )

            self._amplitudes = type(self._amplitudes).from_array(
                self._amplitudes, amp_vec
            )

            # from coupledcluster.tools import biortonormalize

            # t,l,C,Ctilde = self._amplitudes

            # print("AAA", self.np.linalg.norm(Ctilde @ C - self.np.eye(C.shape[0])))
            # C, Ctilde = biortonormalize(C, Ctilde)
            # print("BBB", self.np.linalg.norm(Ctilde @ C - self.np.eye(C.shape[0])))

            # self._amplitudes = OACCVector(t, l, C ,Ctilde, np=self.np)


            if abs(self.last_timestep - (time_points[i] + dt)) > timestep_tol:
                self.update_hamiltonian(time_points[i] + dt, self._amplitudes)
                self.last_timestep = time_points[i] + dt

            yield self._amplitudes


    def __call__(self, prev_amp, current_time):
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
        # as in regular td, but divided by imaginary number
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
        C_new = -1j * (np.dot(C, eta))
        C_tilde_new = 1j * (-np.dot(eta, C_tilde))

        """
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
        """

        self.last_timestep = current_time

        # Return amplitudes and C and C_tilde
        return OACCVector(
            t=t_new, l=l_new, C=C_new, C_tilde=C_tilde_new, np=self.np
        ).asarray()
