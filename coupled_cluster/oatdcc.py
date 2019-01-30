import abc
from coupled_cluster.cc_helper import OACCVector
from coupled_cluster.tdcc import TimeDependentCoupledCluster
from coupled_cluster.integrator import RungeKutta4


class OATDCC(TimeDependentCoupledCluster, metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of an orbital-adaptive
    time-dependent coupled cluster solver class.

    Note that this solver _only_ supports a basis of orthonomal orbitals. If the
    original atomic orbitals are not orthonormal, this can solved done by
    transforming the ground state orbitals to the Hartree-Fock basis.
    """

    def set_initial_conditions(self, amplitudes=None, C=None, C_tilde=None):
        if amplitudes is None:
            # Create copy of ground state amplitudes for time-integration
            amplitudes = self.cc.get_amplitudes()

        if C is None:
            C = self.np.eye(self.system.l)

        if C_tilde is None:
            C_tilde = self.np.eye(self.system.l)

        self._amplitudes = OACCVector(t, l, C, C_tilde)

    @abc.abstractmethod
    def compute_time_dependent_overlap(self):
        """The time-dependent overlap for orbital-adaptive coupled cluster
        changes due to the time-dependent orbitals in the wavefunctions.
        """
        pass

    @abc.abstractmethod
    def compute_p_space_equations(self):
        pass

    def __call__(self, prev_amp, current_time):
        t_old, l_old, C, C_tilde = prev_amp

        # Evolve system in time
        self.h_orig = self.system.h_t(current_time)
        self.u_orig = self.system.u_t(current_time)

        # Change basis to C and C_tilde
        # TODO: Consider offloading these transformations to separate functions
        self.h = C_tilde @ self.h @ C
        self.u = np.einsum(
            "pa, qb, cr, ds, abcd -> pqrs",
            C_tilde,
            C_tilde,
            C,
            C,
            self.u,
            optimize=True,
        )
        self.f = self.system.construct_fock_matrix(self.h, self.u)

        # OATDCC procedure:
        # Do amplitude step
        t_new = [
            -1j * rhs_t_func(self.f, self.u, *t_old, o, v, np=self.np)
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        l_new = [
            1j * rhs_l_func(self.f, self.u, *t_old, *l_old, o, v, np=self.np)
            for rhs_l_func in self.rhs_l_amplitudes()
        ]

        # Compute density matrices
        self.rho_qp = self.compute_one_body_density_matrix()
        self.rho_qspr = self.compute_two_body_density_matrix()

        # Solve P-space equations for eta
        eta = self.compute_p_space_equations()

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
        rho_pq_inv = self.np.linalg.inv(self.rho_qp)

        # Solve Q-space for C and C_tilde
        C_new = -1j * compute_q_space_ket_equations(
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
        C_tilde_new = 1j * compute_q_space_bra_equations(
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

        # Return amplitudes and C and C_tilde
        return OACCVector(t=t_new, l=l_new, C=C_new, C_tilde=C_tilde_new)


def compute_q_space_ket_equations(
    C, C_tilde, eta, h, h_tilde, u, u_tilde, rho_inv_pq, rho_qspr, np
):
    rhs = 1j * np.dot(C, eta)
    rhs += np.dot(h, C)
    rhs -= np.dot(C, h_tilde)

    u_quart = np.einsum(
        "qb, cr, ds, pbcd -> pqrs", C_tilde, C, C, u, optimize=True
    )
    u_quart -= np.tensordot(C, u_tilde, axes=((1), (0)))

    temp_ap = np.tensordot(u_quart, rho_qspr, axes=((1, 2, 3), (3, 0, 1)))
    rhs += np.dot(temp_ap, rho_inv_pq)

    return rhs


def compute_q_space_bra_equations(
    C, C_tilde, eta, h, h_tilde, u, u_tilde, rho_inv_pq, rho_qspr, np
):
    rhs = 1j * np.dot(eta, C_tilde)
    rhs += np.dot(C_tilde, h)
    rhs -= np.dot(h_tilde, C_tilde)

    u_quart = np.einsum(
        "pa, qb, ds, abrd -> pqrs", C_tilde, C_tilde, C, u, optimize=True
    )
    u_quart -= np.tensordot(u_tilde, C_tilde, axes=((2), (0))).transpose(
        0, 1, 3, 2
    )

    temp_qb = np.tensordot(rho_qspr, u_quart, axes=((1, 2, 3), (3, 0, 1)))
    rhs += np.dot(rho_inv_pq, temp_qb)

    return rhs
