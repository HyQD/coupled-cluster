import abc
import warnings
from coupled_cluster.cc_helper import OACCVector
from coupled_cluster.tdcc import TimeDependentCoupledCluster
from quantum_systems import GeneralOrbitalSystem


class OATDCC(TimeDependentCoupledCluster, metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of an orbital-adaptive
    time-dependent coupled cluster solver class.

    Note that this solver _only_ supports a basis of orthonomal orbitals. If the
    original atomic orbitals are not orthonormal, this can solved done by
    transforming the ground state orbitals to the Hartree-Fock basis.

    Note also that this solver (should) support both spin dependent and spin-independent
    basis sets.

    Parameters
    ==========
    system : GeneralOrbitalSystem
    """

    def __init__(self, system, C=None, C_tilde=None, eps_reg=1e-6):
        self.np = system.np

        self.system = system

        # references to the matrix elements in the original basis
        self.h = self.system.h
        self.u = self.system.u

        assert type(self.system) is GeneralOrbitalSystem
        assert not self.system._basis_set.anti_symmetrized_u

        if C is None:
            C = self.np.eye(system.l)

        if C_tilde is None:
            C_tilde = C.T.conj()

        assert C.shape == C_tilde.T.shape
        assert C.shape[0] >= C.shape[1]

        # Q space is non-zero if C is rectangular
        self.has_non_zero_Q_space = C.shape[0] != C.shape[1]
        self.eps_reg = eps_reg

        # OA system sizes
        n_prime = self.system.n
        l_prime = C.shape[1]
        m_prime = l_prime - n_prime

        self.o_prime = slice(0, n_prime)
        self.v_prime = slice(n_prime, l_prime)

        if not self.np.isclose(self.np.sum(C_tilde @ C), l_prime):
            import warnings

            warnings.warn("C_tilde @ C does not add up to basis size")

        self._amp_template = self.construct_oaccvector_template(
            self.truncation, n_prime, m_prime, C.shape, self.np
        )

        self.initialize_arrays(self.system.l, l_prime)
        self.update_oa_hamiltonian(C=C, C_tilde=C_tilde)

        self.last_timestep = None

    @staticmethod
    def construct_oaccvector_template(
        truncation, n_prime, m_prime, C_shape, np
    ):
        _amp = super(OATDCC, OATDCC).construct_amplitude_template(
            truncation, n_prime, m_prime, np=np
        )
        C = np.eye(C_shape[0])[:, : C_shape[1]]
        _amp_template = OACCVector(*_amp, C, C.T.conj(), np=np)
        return _amp_template

    @abc.abstractmethod
    def one_body_density_matrix(self, t, l):
        pass

    @abc.abstractmethod
    def two_body_density_matrix(self, t, l):
        pass

    @abc.abstractmethod
    def compute_overlap(self, current_time, y_a, y_b):
        """The time-dependent overlap for orbital-adaptive coupled cluster
        changes due to the time-dependent orbitals in the wavefunctions.
        """
        pass

    def compute_particle_density(self, current_time, y):
        np = self.np

        rho_qp = self.compute_one_body_density_matrix(current_time, y)

        if np.abs(np.trace(rho_qp) - self.system.n) > 1e-8:
            warn = "Trace of rho_qp = {0} != {1} = number of particles"
            warn = warn.format(np.trace(rho_qp), self.system.n)
            warnings.warn(warn)

        t, l, C, C_tilde = self._amp_template.from_array(y)

        return self.system.compute_particle_density(
            rho_qp, C=C, C_tilde=C_tilde
        )

    @abc.abstractmethod
    def compute_p_space_equations(self):
        pass

    @staticmethod
    def construct_mean_field_operator(u, C, C_tilde, np, **kwargs):
        """Constructs the mean field operator W from the (non-antisymmetric) coloumb
        integrals by converting two of the indices to the new basis given by C &
        C_tilde:
            W^{ar}_{bs} r= \sum_{ab} u^{ag}_{bd} \tilde C^r_g C^d_s.
        Here a, g, b and d are indices in the original basis, while r and s are indicies
        in the new basis
        """
        return np.einsum(
            "rg,ds,agbd->arbs", C_tilde, C, u, optimize=True, **kwargs
        )

    @staticmethod
    def contract_W_partially_ket(W_arbs, C, C_tilde, np, **kwargs):
        """Constructs an intermediate coloumb integral matrices with one unconverted
        index in the ket. For use in both the construction of the fully converted
        integrals and in the Q-space ket equations

        Parameters:
        W_arbs : numpy array
            shape (l, l_prime, l, l_prime)
        C : numpy array
            shape (l, l_prime)
        C_tilde : numpy array
            shape (l_prime, l)
        out_values : numpy array
            shapes (l, l_prime, l_prime, l_prime)
        **kwargs : supports out (see np.einsum)
        """
        return np.einsum("bq,arbs->arqs", C, W_arbs, **kwargs)

    @staticmethod
    def contract_W_partially_bra(W_arbs, C, C_tilde, np, **kwargs):
        """Constructs an intermediate coloumb integral matrices with one unconverted
        index in the bra. For use in the Q-space bra equations

        Parameters:
        W_arbs : numpy array
            shape (l, l_prime, l, l_prime)
        C : numpy array
            shape (l, l_prime)
        C_tilde : numpy array
            shape (l_prime, l)
        out_values : numpy array
            shapes (l_prime, l_prime, l, l_prime)
        **kwargs : supports out (see np.einsum)
        """
        return np.einsum("pa,arbs->prbs", C_tilde, W_arbs, **kwargs)

    def initialize_arrays(self, l, l_prime):
        np = self.np

        if self.has_non_zero_Q_space:
            self.W = np.zeros((l, l_prime, l, l_prime), dtype=np.complex128)
            self.u_quart_ket = np.zeros(
                (l, l_prime, l_prime, l_prime), dtype=np.complex128
            )
            self.u_quart_bra = np.zeros(
                (l_prime, l_prime, l, l_prime), dtype=np.complex128
            )
        else:
            self.W = None
            self.u_quart_ket = None
            self.u_quart_bra = None

        self.h_prime = np.zeros((l_prime, l_prime), dtype=np.complex128)
        self.u_prime = np.zeros(
            (l_prime, l_prime, l_prime, l_prime), dtype=np.complex128
        )
        self.f_prime = np.zeros((l_prime, l_prime), dtype=np.complex128)

    def update_hamiltonian(self, current_time, y):
        """Overwrite tdcc method as fock matrix in the original basis is not needed"""
        if self.last_timestep == current_time:
            return

        self.last_timestep = current_time

        if self.system.has_one_body_time_evolution_operator:
            self.h = self.system.h_t(current_time)

        if self.system.has_two_body_time_evolution_operator:
            self.u = self.system.u_t(current_time)

    def update_oa_hamiltonian(self, C, C_tilde=None):
        np = self.np

        h = self.h
        u = self.u

        if C_tilde is None:
            C_tilde = C.T.conj()

        if self.has_non_zero_Q_space:
            self.construct_mean_field_operator(
                u, C, C_tilde, np=self.np, out=self.W
            )

            self.contract_W_partially_ket(
                self.W, C, C_tilde, np=self.np, out=self.u_quart_ket
            )
            self.contract_W_partially_bra(
                self.W, C, C_tilde, np=self.np, out=self.u_quart_bra
            )

            # non-antisymmetrized
            np.einsum(
                "bq,prbs->prqs",
                C,
                self.u_quart_bra,
                optimize=True,
                out=self.u_prime,
            )

            self.u_quart_ket -= np.tensordot(C, self.u_prime, axes=((1), (0)))
            self.u_quart_bra -= np.tensordot(
                self.u_prime, C_tilde, axes=((2), (0))
            ).transpose(0, 1, 3, 2)
        else:
            # non-antisymmetrized
            self.u_prime = self.system.transform_two_body_elements(
                self.u, C=C, C_tilde=C_tilde,
            )

        self.h_prime = self.system.transform_one_body_elements(h, C, C_tilde)
        self.u_prime = self.system._basis_set.anti_symmetrize_u(self.u_prime)
        self.f_prime = self.system.construct_fock_matrix(
            self.h_prime, self.u_prime
        )

    def __call__(self, current_time, y):
        np = self.np
        o_prime, v_prime = self.o_prime, self.v_prime

        t_old, l_old, C, C_tilde = self._amp_template.from_array(y)

        self.update_hamiltonian(current_time, y)
        self.update_oa_hamiltonian(C=C, C_tilde=C_tilde)

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

        l_new = [
            1j
            * rhs_l_func(
                self.f_prime,
                self.u_prime,
                *t_old,
                *l_old,
                o_prime,
                v_prime,
                np=self.np
            )
            for rhs_l_func in self.rhs_l_amplitudes()
        ]

        # Compute density matrices
        self.rho_qp = self.one_body_density_matrix(t_old, l_old)
        self.rho_qspr = self.two_body_density_matrix(t_old, l_old)

        # Solve P-space equations for eta
        eta = self.compute_p_space_equations()

        # P-space contribution to C and C_tilde
        C_new = np.dot(C, eta)
        C_tilde_new = -np.dot(eta, C_tilde)

        if self.has_non_zero_Q_space:
            # Q-space contribution to C and C_tildde

            # Compute the inverse of rho_qp needed in Q-space eqs.
            """
            If rho_qp is singular we can regularize it as,

            rho_qp_reg = rho_qp + eps*expm( -(1.0/eps) * rho_qp) Eq [3.14]
            Multidimensional Quantum Dynamics, Meyer

            with eps = 1e-8 (or some small number). It seems like it is standard in
            the MCTDHF literature to always work with the regularized rho_qp. Note
            here that expm refers to the matrix exponential which I can not find in
            numpy only in scipy.

            NOTE! scipy.linalg.expm gives nan instead of zero for large-norm negative
            values in version < 1.5,
            """
            from scipy.linalg import expm

            rho_qp_reg = self.rho_qp + self.eps_reg * expm(
                -(1.0 / self.eps_reg) * self.rho_qp
            )
            # small numbers in expm gives nan instead of 0
            rho_qp_reg[np.isnan(rho_qp_reg)] = 0
            rho_inv_pq = self.np.linalg.inv(rho_qp_reg)

            C_new += -1j * self.compute_q_space_ket_equations(
                C,
                self.h,
                self.h_prime,
                self.u_quart_ket,
                self.rho_qspr,
                rho_inv_pq,
            )
            C_tilde_new += 1j * self.compute_q_space_bra_equations(
                C_tilde,
                self.h,
                self.h_prime,
                self.u_quart_bra,
                self.rho_qspr,
                rho_inv_pq,
            )

        self.last_timestep = current_time

        # Return amplitudes and C and C_tilde
        return OACCVector(
            t=t_new, l=l_new, C=C_new, C_tilde=C_tilde_new, np=self.np
        ).asarray()

    def compute_q_space_ket_equations(
        self, C, h, h_prime, u_quart_ket, rho_qspr, rho_inv_pq
    ):
        np = self.np

        rhs = np.dot(h, C)
        rhs -= np.dot(C, h_prime)

        temp_ap = np.tensordot(
            u_quart_ket, rho_qspr, axes=((1, 2, 3), (3, 0, 1))
        )
        rhs += np.dot(temp_ap, rho_inv_pq)

        return rhs

    def compute_q_space_bra_equations(
        self, C_tilde, h, h_prime, u_quart_bra, rho_qspr, rho_inv_pq
    ):
        np = self.np

        rhs = np.dot(C_tilde, h)
        rhs -= np.dot(h_prime, C_tilde)

        temp_qb = np.tensordot(
            rho_qspr, u_quart_bra, axes=((1, 2, 3), (3, 0, 1))
        )
        rhs += np.dot(rho_inv_pq, temp_qb)

        return rhs
