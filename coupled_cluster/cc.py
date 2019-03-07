import abc
import tqdm
import warnings

from coupled_cluster.cc_helper import (
    AmplitudeContainer,
    compute_reference_energy,
    compute_particle_density,
)
from coupled_cluster.mix import AlphaMixer


class CoupledCluster(metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of a Coupled Cluster ground
    state solver class.
    """

    def __init__(self, system, mixer=AlphaMixer, verbose=False, np=None):
        if np is None:
            import numpy as np

        self.np = np

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

    def get_amplitudes(self):
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

    def compute_particle_density(self):
        np = self.np

        rho_qp = self.compute_one_body_density_matrix()

        if np.abs(np.trace(rho_qp) - self.n) > 1e-8:
            warn = "Trace of rho_qp = {0} != {1} = number of particles"
            warn = warn.format(np.trace(rho_qp), self.n)
            warnings.warn(warn)

        rho = compute_particle_density(rho_qp, self.system.spf, np=np)

        return rho

    def compute_reference_energy(self):
        return compute_reference_energy(
            self.f, self.u, self.o, self.v, np=self.np
        )

    def iterate_l_amplitudes(
        self, max_iterations=100, tol=1e-4, **mixer_kwargs
    ):
        np = self.np

        if not np in mixer_kwargs:
            mixer_kwargs["np"] = np

        self.setup_l_mixer(**mixer_kwargs)

        for i in range(max_iterations):
            self.compute_l_amplitudes()
            residuals = self.compute_l_residuals()

            if self.verbose:
                print(f"Iteration: {i}\tResiduals (l): {residuals}")

            if all(res < tol for res in residuals):
                break

    def iterate_t_amplitudes(
        self, max_iterations=100, tol=1e-4, **mixer_kwargs
    ):
        np = self.np

        if not np in mixer_kwargs:
            mixer_kwargs["np"] = np

        self.setup_t_mixer(**mixer_kwargs)

        for i in range(max_iterations):
            self.compute_t_amplitudes()
            residuals = self.compute_t_residuals()

            if self.verbose:
                print(f"Iteration: {i}\tResiduals (t): {residuals}")

            if all(res < tol for res in residuals):
                break
