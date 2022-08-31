import abc
import collections
import warnings
from coupled_cluster.cc_helper import AmplitudeContainer


class TimeDependentCoupledCluster(metaclass=abc.ABCMeta):
    """Time Dependent Coupled Cluster Parent Class

    Abstract base class defining the skeleton of a time-dependent Coupled
    Cluster solver class.

    Parameters
    ----------
    system : QuantumSystem
        Class instance defining the system to be solved
    """

    def __init__(self, system):
        self.np = system.np

        self.system = system

        self.h = self.system.h
        self.u = self.system.u
        self.f = self.system.construct_fock_matrix(self.h, self.u)
        self.o = self.system.o
        self.v = self.system.v

        self._amp_template = self.construct_amplitude_template(
            self.truncation, self.system.n, self.system.m, np=self.np
        )

        self.last_timestep = None

    @property
    @abc.abstractmethod
    def truncation(self):
        pass

    @staticmethod
    def construct_amplitude_template(truncation, n, m, np):
        """Constructs an empty AmplitudeContainer with the correct shapes, for
        convertion between arrays and amplitudes."""
        codes = {"S": 1, "D": 2, "T": 3, "Q": 4}
        levels = [codes[c] for c in truncation[2:]]

        # start with t_0
        t = [np.array([0], dtype=np.complex128)]
        l = []

        for lvl in levels:
            shape = lvl * [m] + lvl * [n]
            t.append(np.zeros(shape, dtype=np.complex128))
            l.append(np.zeros(shape[::-1], dtype=np.complex128))
        return AmplitudeContainer(t=t, l=l, np=np)

    def amplitudes_from_array(self, y):
        """Construct AmplitudeContainer from numpy array."""
        return self._amp_template.from_array(y)

    @property
    def amp_template(self):
        """Returns static _amp_template, for setting initial conditions etc"""
        return self._amp_template

    @abc.abstractmethod
    def rhs_t_0_amplitude(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def rhs_t_amplitudes(self):
        """Function that needs to be implemented as a generator. The generator
        should return the t-amplitudes right hand sides, in order of increasing
        excitation. For example, for ccsd, this function should contain:

            yield compute_t_1_amplitudes
            yield compute_t_2_amplitudes
        """
        pass

    @abc.abstractmethod
    def rhs_l_amplitudes(self):
        """Function that needs to be implemented as a generator. The generator
        should return the l-amplitudes right hand sides, in order of increasing
        excitation. For example, for ccsd, this function should contain:

            yield compute_l_1_amplitudes
            yield compute_l_2_amplitudes
        """
        pass

    @abc.abstractmethod
    def compute_energy(self, current_time, y):
        pass

    @abc.abstractmethod
    def compute_one_body_density_matrix(self, current_time, y):
        pass

    @abc.abstractmethod
    def compute_two_body_density_matrix(self, current_time, y):
        pass

    @abc.abstractmethod
    def compute_overlap(self, current_time, y_a, y_b):
        pass

    def compute_right_phase(self, current_time, y):
        r"""Function computing the inner product of the (potentially
        time-dependent) reference state and the right coupled-cluster wave
        function.
        That is,

        .. math:: \langle \Phi \rvert \Psi(t) \rangle = \exp(\tau_0),

        where :math:`\tau_0` is the zeroth cluster amplitude.

        Returns
        -------
        complex128
            The right-phase describing the weight of the reference determinant.
        """
        t_0 = self._amp_template.from_array(y).t[0][0]

        return self.np.exp(t_0)

    def compute_left_phase(self, current_time, y):
        r"""Function computing the inner product of the (potentially
        time-dependent) reference state and the left coupled-cluster wave
        function.
        That is,

        .. math:: \langle \tilde{\Psi}(t) \rvert \Phi \rangle
            = \exp(-\tau_0)[1 - \langle \Phi \rvert \hat{\Lambda}(t) \hat{T}(t)
            \lvert \Phi \rangle],

        where :math:`\tau_0` is the zeroth cluster amplitude.

        Returns
        -------
        complex128
            The left-phase describing the weight of the reference determinant.
        """
        t_0 = self._amp_template.from_array(y).t[0][0]

        return self.np.exp(-t_0) * self.compute_left_reference_overlap(
            current_time, y
        )

    def compute_reference_weight(self):
        r"""Function computing the weight of the reference state in the
        time-evolved coupled-cluster wave function. This is given by

        .. math:: W(t) = \frac{1}{4}
            \vert \langle \tilde{\Psi}(t) \rvert \Phi \rangle^{*}
            + \langle \Phi \rvert \Psi(t) \rangle \vert^2,

        where the inner-products are the left- and right-phase expressions.

        Returns
        -------
        complex128
            The weight of the reference state in the time-evolved wave function.
        """

        return 0.25 * (
            self.np.abs(
                self.compute_right_phase(current_time, y)
                + self.compute_left_phase(current_time, y).conj()
            )
            ** 2
        )

    @abc.abstractmethod
    def compute_left_reference_overlap(self, current_time, y):
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
            The amplitudes at the current time step.
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
        TimeDependentCoupledCluster.compute_one_body_density_matrix
        CoupledCluster.compute_one_body_expectation_value

        """
        rho_qp = self.compute_one_body_density_matrix(current_time, y)

        if make_hermitian:
            rho_qp = 0.5 * (rho_qp.conj().T + rho_qp)

        return self.np.trace(self.np.dot(rho_qp, mat))

    def compute_two_body_expectation_value(
        self, current_time, y, op, asym=True
    ):
        r"""Function computing the expectation value of a two-body operator
        :math:`\hat{A}`.  This is done by evaluating

        .. math:: \langle A \rangle = a\rho^{rs}_{pq} A^{pq}_{rs},

        where :math:`p, q, r, s` are general single-particle indices,
        :math:`\rho^{rs}_{pq}` is the two-body density matrix, and :math:`a` is
        a pre factor that is :math:`0.5` if :math:`A^{pq}_{rs}` are the
        anti-symmetrized matrix elements and :math:`1.0` else.

        Parameters
        ----------
        current_time : float
            The current time step.
        y : np.ndarray
            The amplitudes at the current time step.
        op : np.ndarray
            The two-body operator to evaluate (:math:`\hat{A}`), as an ndarray.
            The dimensionality of the matrix must be the same as the two-body
            density matrix, i.e., :math:`\mathbb{C}^{l \times l \times l \times
            l}`, where ``l`` is the number of basis functions.
        asym : bool
            Toggle whether or not ``op`` is anti-symmetrized with ``True``
            being used for anti-symmetric matrix elements. This determines the
            prefactor :math:`a` when tracing the two-body density matrix with
            the two-body operator. Default is ``True``.

        Returns
        -------
        complex
            The expectation value of the one-body operator.

        See Also
        --------
        TimeDependentCoupledCluster.compute_two_body_density_matrix
        CoupledCluster.compute_two_body_expectation_value

        """
        rho_rspq = self.compute_two_body_density_matrix(current_time, y)

        return (0.5 if asym else 1.0) * self.np.tensordot(
            op, rho_rspq, axes=((0, 1, 2, 3), (2, 3, 0, 1))
        )

    def compute_particle_density(self):
        """Computes one-particle density

        Returns
        -------
        np.array
            Particle density
        """
        np = self.np

        rho_qp = self.compute_one_body_density_matrix()

        if np.abs(np.trace(rho_qp) - self.n) > 1e-8:
            warn = "Trace of rho_qp = {0} != {1} = number of particles"
            warn = warn.format(np.trace(rho_qp), self.n)
            warnings.warn(warn)

        return self.system.compute_particle_density(rho_qp)

    def compute_particle_density(self, current_time, y):
        """Computes current one-body density

        Returns
        -------
        np.array
            One-body density of system at current time step
        """
        np = self.np

        rho_qp = self.compute_one_body_density_matrix(current_time, y)

        if np.abs(np.trace(rho_qp) - self.system.n) > 1e-8:
            warn = "Trace of rho_qp = {0} != {1} = number of particles"
            warn = warn.format(np.trace(rho_qp), self.system.n)
            warnings.warn(warn)

        return self.system.compute_particle_density(rho_qp)

    def update_hamiltonian(self, current_time, y):
        if self.last_timestep == current_time:
            return

        self.last_timestep = current_time

        if self.system.has_one_body_time_evolution_operator:
            self.h = self.system.h_t(current_time)

        if self.system.has_two_body_time_evolution_operator:
            self.u = self.system.u_t(current_time)

        self.f = self.system.construct_fock_matrix(self.h, self.u)

    def __call__(self, current_time, prev_amp):
        o, v = self.system.o, self.system.v

        prev_amp = self._amp_template.from_array(prev_amp)
        t_old, l_old = prev_amp

        self.update_hamiltonian(current_time, prev_amp)

        # Remove phase from t-amplitude list
        t_old = t_old[1:]

        t_new = [
            -1j * rhs_t_func(self.f, self.u, *t_old, o, v, np=self.np)
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        # Compute derivative of phase
        t_0_new = -1j * self.rhs_t_0_amplitude(
            self.f, self.u, *t_old, self.o, self.v, np=self.np
        )
        t_new = [t_0_new, *t_new]

        l_new = [
            1j * rhs_l_func(self.f, self.u, *t_old, *l_old, o, v, np=self.np)
            for rhs_l_func in self.rhs_l_amplitudes()
        ]

        self.last_timestep = current_time

        return AmplitudeContainer(t=t_new, l=l_new, np=self.np).asarray()
