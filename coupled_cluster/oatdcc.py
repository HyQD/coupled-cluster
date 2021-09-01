import abc
import warnings
from coupled_cluster.cc_helper import OACCVector
from coupled_cluster.tdcc import TimeDependentCoupledCluster


class OATDCC(TimeDependentCoupledCluster, metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of an orbital-adaptive
    time-dependent coupled cluster solver class.

    Note that this solver _only_ supports a basis of orthonomal orbitals. If the
    original atomic orbitals are not orthonormal, this can solved done by
    transforming the ground state orbitals to the Hartree-Fock basis.
    """

    def __init__(self, system, C=None, C_tilde=None):
        self.np = system.np

        self.system = system

        # these lines is copy paste from super().__init__, and would be nice to
        # remove.
        # See https://github.com/Schoyen/coupled-cluster/issues/36
        self.h = self.system.h
        self.u = self.system.u
        self.f = self.system.construct_fock_matrix(self.h, self.u)
        self.o = self.system.o
        self.v = self.system.v

        if C is None:
            C = self.np.eye(system.l)
        if C_tilde is None:
            C_tilde = C.T

        assert C.shape == C_tilde.T.shape

        n_prime = self.system.n
        l_prime = C.shape[1]
        m_prime = l_prime - n_prime

        self.o_prime = slice(0, n_prime)
        self.v_prime = slice(n_prime, l_prime)

        _amp = self.construct_amplitude_template(
            self.truncation, n_prime, m_prime, np=self.np
        )
        self._amp_template = OACCVector(*_amp, C, C_tilde, np=self.np)

        self.last_timestep = None

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

    def compute_one_body_expectation_value(
        self, current_time, y, mat, make_hermitian=True
    ):
        r"""Function computing the expectation value of a one-body operator
        :math:`\hat{A}`.  This is done by evaluating

        .. math:: \langle A \rangle = \rho^{q}_{p} A^{p}_{q},

        where :math:`p, q` are general single-particle indices and
        :math:`\rho^{q}_{p}` is the one-body density matrix.

        Parameters
        ----------
        current_time : float
            The current time step.
        y : np.ndarray
            The amplitudes and coefficients at the current time step.
        mat : np.ndarray
            The one-body operator to evaluate (:math:`\hat{A}`), as a matrix.
            The dimensionality of the matrix must be the same as the one-body
            density matrix, i.e., :math:`\mathbb{C}^{l \times l}`, where ``l``
            is the number of basis functions.
        make_hermitian : bool
            Whether or not to make the one-body density matrix Hermitian. This
            is done by :math:`\tilde{\boldsymbol{\rho}} =
            \frac{1}{2}(\boldsymbol{\rho}^{\dagger} + \boldsymbol{\rho}), where
            :math:`\tilde{\boldsymbol{\rho}}` is the Hermitian one-body density
            matrix. Default is ``make_hermitian=True``.

        Returns
        -------
        complex
            The expectation value of the one-body operator.

        See Also
        --------
        OATDCC.compute_one_body_density_matrix
        TimeDependentCoupledCluster.compute_one_body_expectation_value

        """
        t, l, C, C_tilde = self._amp_template.from_array(y)

        return super().compute_one_body_expectation_value(
            current_time,
            y,
            self.system.transform_one_body_elements(mat, C, C_tilde),
            make_hermitian=make_hermitian,
        )

    def compute_two_body_expectation_value(
        self, current_time, y, op, asym=True
    ):
        t, l, C, C_tilde = self._amp_template.from_array(y)

        return super().compute_two_body_expectation_value(
            current_time,
            y,
            self.system.transform_two_body_elements(op, C, C_tilde),
            asym=asym,
        )

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

    def update_hamiltonian(self, current_time, y=None, C=None, C_tilde=None):
        if self.last_timestep == current_time:
            return

        self.last_timestep = current_time

        if y is not None:
            _, _, C, C_tilde = self._amp_template.from_array(y)
        elif C is not None and C_tilde is not None:
            pass
        else:
            raise ValueError(
                "either the amplitude-array or (C and C_tilde) has to be not "
                + "None."
            )

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

        self.f_prime = self.system.construct_fock_matrix(
            self.h_prime, self.u_prime
        )

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
            self.rho_qp,
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
            self.rho_qp,
            self.rho_qspr,
            np=np,
        )
        """

        self.last_timestep = current_time

        # Return amplitudes and C and C_tilde
        return OACCVector(
            t=t_new, l=l_new, C=C_new, C_tilde=C_tilde_new, np=self.np
        ).asarray()


def compute_q_space_ket_equations(
    C, C_tilde, eta, h, h_prime, u, u_prime, rho_inv_pq, rho_qp, rho_qspr, np
):
    rhs = 1j * np.dot(C, eta)

    rhs += np.dot(h, C)
    rhs -= np.dot(C, h_prime)

    u_quart = np.einsum("rb,gq,ds,abgd->arqs", C_tilde, C, C, u, optimize=True)
    u_quart -= np.tensordot(C, u_prime, axes=((1), (0)))

    temp_ap = np.tensordot(u_quart, rho_qspr, axes=((1, 2, 3), (3, 0, 1)))
    rhs += np.dot(temp_ap, rho_inv_pq)

    return rhs


def compute_q_space_bra_equations(
    C, C_tilde, eta, h, h_prime, u, u_prime, rho_inv_pq, rho_qp, rho_qspr, np
):
    rhs = 1j * np.dot(eta, C_tilde)

    rhs += np.dot(C_tilde, h)
    rhs -= np.dot(h_prime, C_tilde)

    u_quart = np.einsum(
        "pa,rg,ds,agbd->prbs", C_tilde, C_tilde, C, u, optimize=True
    )
    u_quart -= np.tensordot(u_prime, C_tilde, axes=((2), (0))).transpose(
        0, 1, 3, 2
    )

    temp_qb = np.tensordot(rho_qspr, u_quart, axes=((1, 2, 3), (3, 0, 1)))
    rhs += np.dot(rho_inv_pq, temp_qb)

    return rhs
