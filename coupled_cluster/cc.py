import abc
import warnings

import scipy.optimize

from coupled_cluster.cc_helper import (
    AmplitudeContainer,
    compute_reference_energy,
)

from opt_einsum import contract


class CoupledCluster(metaclass=abc.ABCMeta):
    """Coupled Cluster Abstract class

    Abstract base class defining the skeleton of a
    Coupled Cluster ground state solver class.

    Parameters
    ----------
    system : QuantumSystems
        Quantum systems class instance
    verbose : bool
        Prints iterations for ground state computation if True
    """

    def __init__(self, system, verbose=False):
        self.np = system.np

        self.system = system
        self.verbose = verbose

        self.h = self.system.h
        self.u = self.system.u
        self.f = self.system.construct_fock_matrix(self.h, self.u)

    def get_initial_guess(self):
        return AmplitudeContainer.construct_amplitude_template(
            self.truncation,
            self.system.n,
            self.system.m,
            self.np,
            self.system.h.dtype,
        )

    def compute_energy(self, y):
        ob_density = self.compute_one_body_density_matrix(y)
        tb_density = self.compute_two_body_density_matrix(y)

        return (
            contract("pq, qp ->", self.h, ob_density)
            + 0.25 * contract("pqrs, rspq ->", self.u, tb_density)
            + self.system.nuclear_repulsion_energy
        )

    @property
    @abc.abstractmethod
    def truncation(self):
        pass

    @property
    @abc.abstractmethod
    def rhs_t_amplitudes(self):
        pass

    @property
    @abc.abstractmethod
    def rhs_l_amplitudes(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def construct_one_body_density_matrix(*args, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def construct_two_body_density_matrix(*args, **kwargs):
        pass

    def compute_one_body_density_matrix(self, y):
        amps = AmplitudeContainer.construct_container_from_array(
            y, self.truncation, self.system.n, self.system.m, self.np
        )

        return self.construct_one_body_density_matrix(
            *amps.t[1:], *amps.l, self.system.o, self.system.v, self.np
        )

    def compute_two_body_density_matrix(self, y):
        amps = AmplitudeContainer.construct_container_from_array(
            y, self.truncation, self.system.n, self.system.m, self.np
        )

        return self.construct_two_body_density_matrix(
            *amps.t[1:], *amps.l, self.system.o, self.system.v, self.np
        )

    def compute_one_body_expectation_value(self, y, mat, make_hermitian=True):
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
        ob_density = self.compute_one_body_density_matrix(y)

        if make_hermitian:
            ob_density = 0.5 * (ob_density.conj().T + ob_density)

        return self.np.trace(self.np.dot(ob_density, mat))

    def compute_particle_density(self, y):
        """Computes one-particle density

        Returns
        -------
        np.array
            Particle density
        """
        np = self.np

        ob_density = self.compute_one_body_density_matrix(y)

        if np.abs(np.trace(ob_density) - self.system.n) > 1e-8:
            warn = "Trace of ob_density = {0} != {1} = number of particles"
            warn = warn.format(np.trace(ob_density), self.system.n)
            warnings.warn(warn)

        return self.system.compute_particle_density(ob_density)

    # def compute_ground_state(self):
    #     amp_0 = self.get_initial_guess()

    #     res = scipy.optimize.minimize(
    #         self.compute_energy,
    #         amp_0.asarray(),
    #         method="BFGS",
    #         jac=self,
    #         options={"gtol": 1e-6, "disp": True},
    #     )

    #     print(self.compute_energy(res))

    def __call__(self, y):
        amps = AmplitudeContainer.construct_container_from_array(
            y, self.truncation, self.system.n, self.system.m, self.np
        )

        t_rhs = [
            rhs_t_func(
                self.f,
                self.u,
                *amps.t[1:],
                self.system.o,
                self.system.v,
                np=self.np
            )
            for rhs_t_func in self.rhs_t_amplitudes
        ]
        t_rhs.insert(0, amps.t[0])

        l_rhs = [
            rhs_l_func(
                self.f,
                self.u,
                *amps.t[1:],
                *amps.l,
                self.system.o,
                self.system.v,
                np=self.np
            )
            for rhs_l_func in self.rhs_l_amplitudes
        ]

        return AmplitudeContainer(t=t_rhs, l=l_rhs, np=self.np).asarray()
