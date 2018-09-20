import abc
import numpy as np


class CoupledCluster(metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of a Coupled Cluster solver
    class.
    """

    def __init__(self, system, verbose=False):
        self.system = system
        self.verbose = verbose

        self.n = self.system.n
        self.l = self.system.l
        self.m = self.system.m

        self.h, self.f, self.u = self.system.h, self.system.f, self.system.u

        self.o, self.v = self.system.o, self.system.v

    @abc.abstractmethod
    def _compute_energy(self):
        pass

    @abc.abstractmethod
    def _compute_amplitudes(self):
        pass

    @abc.abstractmethod
    def _compute_lambda_amplitudes(self, theta):
        pass

    @abc.abstractmethod
    def _compute_one_body_density_matrix(self):
        pass

    @abc.abstractmethod
    def _get_t_copy(self):
        pass

    @abc.abstractmethod
    def _get_lambda_copy(self):
        pass

    @abc.abstractmethod
    def _set_t(self, t):
        pass

    @abc.abstractmethod
    def _set_l(self, l):
        pass

    def _timestep(self, u, t):
        l, t = u
        self._set_t(t)
        self._set_l(l)

        self.system.evolve_in_time(t)

        self._compute_amplitudes(0)
        self._compute_lambda_amplitudes(0)

        l = list(map(lambda l: 1j * l, self._get_lambda_copy()))
        t = list(map(lambda t: -1j * t, self._get_t_copy()))

        return (l, t)

    def evolve_amplitudes(self, t_start, t_end, num_timesteps):
        t = t_start
        h = (t_end - t_start) / (num_timesteps - 1)
        self._l_0 = self._get_lambda_copy()
        self._t_0 = self._get_t_copy()

        l = self._l_0
        t = self._t_0
        while t < t_end:
            u = (l, t)

            k_1 = self._timestep(u, t + h)

            t += h

    def compute_one_body_density(self, spf):
        rho_qp = self._compute_one_body_density_matrix()

        if self.verbose and np.abs(np.trace(rho_qp) - self.n) > 1e-8:
            print(
                (
                    "Warning: trace of rho_qp = {0} != {1} = "
                    + "number of particles"
                ).format(np.trace(rho_qp), self.n)
            )

        rho_qp_reduced = rho_qp[::2, ::2] + rho_qp[1::2, 1::2]
        rho = np.zeros(spf.shape[0])

        for i in range(len(rho)):
            rho[i] += np.dot(spf[i].conj(), np.dot(rho_qp_reduced, spf[i]))

        return rho

    def compute_reference_energy(self):
        h, u, o, v = self.h, self.u, self.o, self.v
        e_ref = np.einsum("ii ->", h[o, o]) + 0.5 * np.einsum(
            "ijij ->", u[o, o, o, o]
        )

        return e_ref

    def compute_lambda_amplitudes(
        self, max_iterations=100, tol=1e-4, theta=0.1
    ):
        assert 0 <= theta <= 1, "Mixing parameter theta must be in [0, 1]"

        iterations = 0

        diff_l_1 = 100
        diff_l_2 = 100

        l_1 = self.l_1.copy()
        l_2 = self.l_2.copy()

        while (
            diff_l_1 > tol or diff_l_2 > tol
        ) and iterations < max_iterations:
            if self.verbose:
                print(
                    "Iteration: {0}\tDiff (l_1): {1}\tDiff (l_2): {2}".format(
                        iterations, diff_l_1, diff_l_2
                    )
                )

            self._compute_lambda_amplitudes(theta)
            diff_l_1 = np.amax(np.abs(self.l_1 - l_1))
            diff_l_2 = np.amax(np.abs(self.l_2 - l_2))

            np.copyto(l_1, self.l_1)
            np.copyto(l_2, self.l_2)

            iterations += 1

    def compute_ground_state_energy(
        self, max_iterations=100, tol=1e-4, theta=0.1
    ):
        assert 0 <= theta <= 1, "Mixing parameter theta must be in [0, 1]"

        iterations = 0

        diff = 100
        energy = self._compute_energy()

        while diff > tol and iterations < max_iterations:
            if self.verbose:
                print("Iteration: {0}\tEnergy: {1}".format(iterations, energy))

            self._compute_amplitudes(theta)
            energy_prev = energy
            energy = self._compute_energy()
            diff = abs(energy - energy_prev)
            iterations += 1

        return energy, iterations
