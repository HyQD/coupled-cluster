import abc
import warnings
from coupled_cluster.cc_helper import OACCVector, compute_particle_density
from coupled_cluster.tdcc import TimeDependentCoupledCluster
from coupled_cluster.integrators import RungeKutta4


class OATDCC(TimeDependentCoupledCluster, metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of an orbital-adaptive
    time-dependent coupled cluster solver class.

    Note that this solver _only_ supports a basis of orthonomal orbitals. If the
    original atomic orbitals are not orthonormal, this can solved done by
    transforming the ground state orbitals to the Hartree-Fock basis.
    """

    def set_initial_conditions(self, cc=None, amplitudes=None, C=None,
            C_tilde=None, *args, **kwargs):
        """Set initial condition of system.

        Necessary to call this function befor computing time development. Must
        be passed with either ground state solver or amplitudes, will revert to
        amplitudes of ground state solver.

        args and kwargs sent to ground state solver if amplitudes is None.

        Parameters
        ----------
        cc : CoupledCluster (optional)
            Ground state solver
        amplitudes : AmplitudeContainer (optional)
            Amplitudes for the system
        """
        if amplitudes is None:
            if cc is None:
                raise TypeError('must specify either amplitudes or ' 
                        + 'initialized ground state solver')
            # remnant from old 
            if "change_system_basis" not in kwargs:
                kwargs["change_system_basis"] = True
            # Compute ground state amplitudes for time-integration
            cc.compute_ground_state(*args, **kwargs)
            amplitudes = cc.get_amplitudes(get_t_0=True)

        if C is None:
            C = self.np.eye(self.system.l)

        if C_tilde is None:
            C_tilde = self.np.eye(self.system.l)
        
        assert C.shape == C_tilde.T.shape
        self.h = self.system.h
        self.u = self.system.u

        self.h_prime = self.system.transform_one_body_elements(
            self.h, C, C_tilde
        )
        self.u_prime = self.system.transform_two_body_elements(
            self.u, C, C_tilde
        )
        self.f_prime = self.system.construct_fock_matrix(   
            self.h_prime, self.u_prime
        )
        
        self.n_prime = self.system.n
        self.l_prime = C.shape[1]
        self.m_prime = self.l_prime - self.n_prime

        self.o_prime = slice(0, self.n_prime)
        self.v_prime = slice(self.n_prime, self.l_prime)

        self._amplitudes = OACCVector(*amplitudes, C, C_tilde, np=self.np)

    @abc.abstractmethod
    def one_body_density_matrix(self, t, l):
        pass

    @abc.abstractmethod
    def two_body_density_matrix(self, t, l):
        pass

    @abc.abstractmethod
    def compute_time_dependent_overlap(self):
        """The time-dependent overlap for orbital-adaptive coupled cluster
        changes due to the time-dependent orbitals in the wavefunctions.
        """
        pass

    def compute_particle_density(self):
        np = self.np

        rho_qp = self.compute_one_body_density_matrix()

        if np.abs(np.trace(rho_qp) - self.system.n) > 1e-8:
            warn = "Trace of rho_qp = {0} != {1} = number of particles"
            warn = warn.format(np.trace(rho_qp), self.system.n)
            warnings.warn(warn)

        t, l, C, C_tilde = self._amplitudes

        # pa, a -> p
        # C_tilde, spf^{*} -> spf_tilde
        bra_spf = np.tensordot(C_tilde, self.system.bra_spf, axes=((1), (0)))
        # ap, a -> p
        # C, spf -> spf
        ket_spf = np.tensordot(C, self.system.spf, axes=((0), (0)))

        rho = compute_particle_density(rho_qp, bra_spf, ket_spf, np=np)

        return rho

    @abc.abstractmethod
    def compute_p_space_equations(self):
        pass

    def update_hamiltonian(self, current_time, amplitudes):
        C = amplitudes.C
        C_tilde = amplitudes.C_tilde

        # Evolve system in time
        if self.system.has_one_body_time_evolution_operator:
            self.h = self.system.h_t(current_time)

        if self.system.has_two_body_time_evolution_operator:
            self.u = self.system.u_t(current_time)

        # Change basis to C and C_tilde
        self.h_prime = self.system.transform_one_body_elements(
            self.h, C, C_tilde
        )
        self.u_prime = self.system.transform_two_body_elements(
            self.u, C, C_tilde
        )

        self.f_prime = self.system.construct_fock_matrix(self.h_prime, self.u_prime)

    def __call__(self, prev_amp, current_time):
        np = self.np
        o_prime, v_prime = self.o_prime, self.v_prime

        prev_amp = OACCVector.from_array(self._amplitudes, prev_amp)
        t_old, l_old, C, C_tilde = prev_amp

        self.update_hamiltonian(current_time, prev_amp)

        # Remove t_0 phase as this is not used in any of the equations
        t_old = t_old[1:]

        # OATDCC procedure:
        # Do amplitude step
        t_new = [
            -1j * rhs_t_func(self.f_prime, self.u_prime, *t_old, o_prime,
                v_prime, np=self.np)
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        # Compute derivative of phase
        t_0_new = -1j * self.rhs_t_0_amplitude(
            self.f_prime, self.u_prime, *t_old, o_prime, v_prime, np=self.np
        )

        t_new = [t_0_new, *t_new]

        l_new = [
            1j * rhs_l_func(
                self.f_prime, self.u_prime, *t_old, *l_old, o_prime, v_prime, 
                np=self.np
            )
            for rhs_l_func in self.rhs_l_amplitudes()
        ]

        # Compute density matrices
        self.rho_qp = self.one_body_density_matrix(t_old, l_old)
        self.rho_qspr = self.two_body_density_matrix(t_old, l_old)

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

        """
        C_new = np.dot(C, eta)
        C_tilde_new = -np.dot(eta, C_tilde)
        """

        C_new = -1j * compute_q_space_ket_equations(
            C,
            C_tilde,
            eta,
            self.h,
            self.h_prime,
            self.u,
            self.u_prime,
            rho_pq_inv,
            self.rho_qspr,
            np=np,
        )
        C_tilde_new = 1j * compute_q_space_bra_equations(
            C,
            C_tilde,
            eta,
            self.h,
            self.h_prime,
            self.u,
            self.u_prime,
            rho_pq_inv,
            self.rho_qspr,
            np=np,
        )

        self.last_timestep = current_time

        # Return amplitudes and C and C_tilde
        return OACCVector(
            t=t_new, l=l_new, C=C_new, C_tilde=C_tilde_new, np=self.np
        ).asarray()


def compute_q_space_ket_equations(
    C, C_tilde, eta, h, h_tilde, u, u_tilde, rho_inv_pq, rho_qspr, np
):
    rhs = 1j * np.dot(C, eta)

    rhs += np.dot(h, C)
    rhs -= np.dot(C, h_tilde)

    u_quart = np.einsum("rb,gq,ds,abgd->arqs", C_tilde[:], C, C, u, optimize=True)
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
        "pa,rg,ds,agbd->prbs", C_tilde, C_tilde, C, u, optimize=True
    )
    u_quart -= np.tensordot(u_tilde, C_tilde, axes=((2), (0))).transpose(
        0, 1, 3, 2
    )

    temp_qb = np.tensordot(rho_qspr, u_quart, axes=((1, 2, 3), (3, 0, 1)))
    rhs += np.dot(rho_inv_pq, temp_qb)

    return rhs
