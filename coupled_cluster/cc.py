import abc
import warnings

from coupled_cluster.cc_helper import (
    AmplitudeContainer,
    compute_reference_energy,
)
from coupled_cluster.mix import AlphaMixer, DIIS


class CoupledCluster(metaclass=abc.ABCMeta):
    """Coupled Cluster Abstract class

    Abstract base class defining the skeleton of a
    Coupled Cluster ground state solver class.

    Parameters
    ----------
    system : QuantumSystems
        Quantum systems class instance
    mixer : AlphaMixer
        AlpaMixer object
    verbose : bool
        Prints iterations for ground state computation if True
    """

    def __init__(self, system, mixer=DIIS, verbose=False):
        self.np = system.np

        self.system = system
        self.verbose = verbose
        self.mixer = mixer

        self.n = self.system.n
        self.l = self.system.l
        self.m = self.system.m

        self.h = self.system.h
        self.u = self.system.u
        self.f = self.system.construct_fock_matrix(self.h, self.u)

        self.o, self.v = self.system.o, self.system.v

    def get_amplitudes(self, get_t_0=False):
        """Getter for amplitudes

        Parameters
        ----------
        get_t_0 : bool
            Returns amplitude at t=0 if True

        Returns
        -------
        AmplitudeContainer
            Amplitudes in AmplitudeContainer object
        """

        if get_t_0:
            return AmplitudeContainer(
                t=[
                    self.np.array([0], dtype=self.np.complex128),
                    *self._get_t_copy(),
                ],
                l=self._get_l_copy(),
                np=self.np,
            )

        return AmplitudeContainer(
            t=self._get_t_copy(), l=self._get_l_copy(), np=self.np
        )

    @abc.abstractmethod
    def _get_t_copy(self):
        pass

    @abc.abstractmethod
    def _get_l_copy(self):
        pass

    @abc.abstractmethod
    def compute_energy(self):
        pass

    @abc.abstractmethod
    def compute_one_body_density_matrix(self):
        pass

    @abc.abstractmethod
    def compute_two_body_density_matrix(self):
        pass

    @abc.abstractmethod
    def compute_t_amplitudes(self):
        pass

    @abc.abstractmethod
    def compute_l_amplitudes(self):
        pass

    @abc.abstractmethod
    def setup_l_mixer(self, **kwargs):
        pass

    @abc.abstractmethod
    def setup_t_mixer(self, **kwargs):
        pass

    @abc.abstractmethod
    def compute_l_residuals(self):
        pass

    @abc.abstractmethod
    def compute_t_residuals(self):
        pass

    def compute_one_body_expectation_value(self, mat, make_hermitian=True):
        r"""Function computing the expectation value of a one-body operator
        :math:`\hat{A}`.  This is done by evaluating

        .. math:: \langle A \rangle = \rho^{q}_{p} A^{p}_{q},

        where :math:`p, q` are general single-particle indices and
        :math:`\rho^{q}_{p}` is the one-body density matrix.

        Parameters
        ----------
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
        CoupledCluster.compute_one_body_density_matrix

        """
        rho_qp = self.compute_one_body_density_matrix()

        if make_hermitian:
            rho_qp = 0.5 * (rho_qp.conj().T + rho_qp)

        return self.np.trace(self.np.dot(rho_qp, mat))

    def compute_two_body_expectation_value(self, op, asym=True):
        r"""Function computing the expectation value of a two-body operator
        :math:`\hat{A}`.  This is done by evaluating

        .. math:: \langle A \rangle = a\rho^{rs}_{pq} A^{pq}_{rs},

        where :math:`p, q, r, s` are general single-particle indices,
        :math:`\rho^{rs}_{pq}` is the two-body density matrix, and :math:`a` is
        a pre factor that is :math:`0.5` if :math:`A^{pq}_{rs}` are the
        anti-symmetrized matrix elements and :math:`1.0` else.

        Parameters
        ----------
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
        CoupledCluster.compute_two_body_density_matrix

        """
        rho_rspq = self.compute_two_body_density_matrix()

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

    def compute_reference_energy(self):
        """Computes reference energy

        Returns
        -------
        np.array
            Reference energy
        """

        return compute_reference_energy(
            self.f, self.u, self.o, self.v, np=self.np
        )

    def compute_ground_state(
        self, t_args=[], t_kwargs={}, l_args=[], l_kwargs={}
    ):
        """Compute ground state energy"""
        self.iterate_t_amplitudes(*t_args, **t_kwargs)
        self.iterate_l_amplitudes(*l_args, **l_kwargs)

        if self.verbose:
            print(
                f"Final {self.__class__.__name__} energy: "
                + f"{self.compute_energy()}"
            )

    def iterate_l_amplitudes(
        self, max_iterations=100, tol=1e-4, **mixer_kwargs
    ):
        np = self.np

        if not "np" in mixer_kwargs:
            mixer_kwargs["np"] = np

        self.setup_l_mixer(**mixer_kwargs)

        for i in range(max_iterations):
            self.compute_l_amplitudes()
            residuals = self.compute_l_residuals()

            if self.verbose:
                print(f"Iteration: {i}\tResiduals (l): {residuals}")

            if all(res < tol for res in residuals):
                break

        assert i < (max_iterations - 1), (
            f"The l amplitudes did not converge. Last residual: "
            + f"{self.compute_l_residuals()}"
        )

    def iterate_t_amplitudes(
        self, max_iterations=100, tol=1e-4, **mixer_kwargs
    ):
        np = self.np

        if not "np" in mixer_kwargs:
            mixer_kwargs["np"] = np

        self.setup_t_mixer(**mixer_kwargs)

        for i in range(max_iterations):
            self.compute_t_amplitudes()
            residuals = self.compute_t_residuals()

            if self.verbose:
                print(f"Iteration: {i}\tResiduals (t): {residuals}")

            if all(res < tol for res in residuals):
                break

        assert i < (max_iterations - 1), (
            f"The t amplitudes did not converge. Last residual: "
            + f"{self.compute_t_residuals()}"
        )
