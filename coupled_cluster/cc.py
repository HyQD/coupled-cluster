import abc
import tqdm
import warnings

from coupled_cluster.cc_helper import (
    AmplitudeContainer,
    compute_reference_energy,
    compute_spin_reduced_one_body_density_matrix,
    remove_diagonal_in_matrix,
    compute_particle_density,
)


class CoupledCluster(metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of a Coupled Cluster ground
    state solver class.
    """

    def __init__(self, system, verbose=False, np=None):
        if np is None:
            import numpy as np

        self.np = np

        self.system = system
        self.verbose = verbose

        self.n = self.system.n
        self.l = self.system.l
        self.m = self.system.m

        self.h = self.system.h
        self.u = self.system.u
        self.f = self.system.construct_fock_matrix(self.h, self.u)
        self.off_diag_f = remove_diagonal_in_matrix(self.f, np=self.np)

        self.o, self.v = self.system.o, self.system.v

    def get_amplitudes(self):
        return AmplitudeContainer(t=self._get_t_copy(), l=self._get_l_copy())

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
    def compute_t_amplitudes(self, theta, iterative=True):
        pass

    @abc.abstractmethod
    def compute_l_amplitudes(self, theta, iterative=True):
        pass

    def compute_particle_density(self):
        np = self.np

        rho_qp = self.compute_one_body_density_matrix()

        if np.abs(np.trace(rho_qp) - self.n) > 1e-8:
            warn = "Trace of rho_qp = {0} != {1} = number of particles"
            warn = warn.format(np.trace(rho_qp), self.n)
            warnings.warn(warn)

        rho_qp_reduced = compute_spin_reduced_one_body_density_matrix(rho_qp)
        rho = compute_particle_density(rho_qp_reduced, self.system.spf)

        return rho

    def compute_reference_energy(self):
        return compute_reference_energy(
            self.f, self.u, self.o, self.v, np=self.np
        )

    def iterate_l_amplitudes(self, max_iterations=100, tol=1e-4, theta=0.1):
        np = self.np

        assert 0 <= theta <= 1, "Mixing parameter theta must be in [0, 1]"

        l_list = self._get_l_copy()
        l_diff = [100 for l in l_list]

        for i in range(max_iterations):
            if self.verbose:
                print(f"Iteration: {i}\tDiff (l): {l_diff}")

            if all([l_d < tol for l_d in l_diff]):
                break

            self.compute_l_amplitudes(theta, iterative=True)
            new_l_list = self._get_l_copy()
            l_diff = [
                np.amax(np.abs(l - l_new))
                for l, l_new in zip(l_list, new_l_list)
            ]

            l_list = new_l_list

    def iterate_t_amplitudes(self, max_iterations=100, tol=1e-4, theta=0.1):
        np = self.np

        assert 0 <= theta <= 1, "Mixing parameter theta must be in [0, 1]"

        t_list = self._get_t_copy()
        t_diff = [100 for t in t_list]

        for i in range(max_iterations):
            if self.verbose:
                print(f"Iteration: {i}\tDiff (t): {t_diff}")

            if all([t_d < tol for t_d in t_diff]):
                break

            self.compute_t_amplitudes(theta, iterative=True)
            new_t_list = self._get_t_copy()
            t_diff = [
                np.amax(np.abs(t - t_new))
                for t, t_new in zip(t_list, new_t_list)
            ]

            t_list = new_t_list
