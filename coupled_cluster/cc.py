import abc
import numpy as np
import tqdm
import warnings

from coupled_cluster.cc_helper import AmplitudeContainer


def compute_reference_energy(f, u, o, v, np=None):
    if np is None:
        import numpy as np

    return np.trace(f[o, o]) - 0.5 * np.trace(
        np.trace(u[o, o, o, o], axis1=1, axis2=3)
    )


def compute_spin_reduced_one_body_density_matrix(rho_qp):
    return rho_qp[::2, ::2] + rho_qp[1::2, 1::2]


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
        self.off_diag_f = self._create_off_diagonal_matrix(self.f)

        self.o, self.v = self.system.o, self.system.v

    def get_amplitudes(self):
        return AmplitudeContainer(t=self._get_t_copy(), l=self._get_l_copy())

    def _create_off_diagonal_matrix(self, matrix):
        off_diag = np.zeros_like(matrix)
        off_diag += matrix
        np.fill_diagonal(off_diag, 0)

        return off_diag

    def __err(self, func_name):
        raise NotImplementedError(
            "Class '{0}' does not have an implementation of '{1}'".format(
                self.__class__.__name__, func_name
            )
        )

    @abc.abstractmethod
    def _compute_energy(self):
        pass

    @abc.abstractmethod
    def _compute_t_amplitudes(self, theta, iterative=True):
        pass

    def _compute_l_amplitudes(self, theta, iterative=True):
        self.__err(self._compute_l_amplitudes.__name__)

    def _compute_time_overlap_with_ground_state(self):
        self.__err(self._compute_time_overlap_with_ground_state.__name__)

    def _compute_one_body_density_matrix(self):
        self.__err(self._compute_one_body_density_matrix.__name__)

    @abc.abstractmethod
    def _get_t_copy(self):
        pass

    def _get_l_copy(self):
        self.__err(self._get_l_copy.__name__)

    @abc.abstractmethod
    def _set_t(self, t):
        pass

    def _set_l(self, l):
        self.__err(self._set_l.__name__)

    def _timestep(self, l_i, t_i, time):
        self._set_l(l_i)
        self._set_t(t_i)

        self.h = self.system.h_t(time)
        self.u = self.system.u_t(time)
        self.f = self.system.construct_fock_matrix(self.h, self.u)

        t_new = [
            -1j * t_x for t_x in self._compute_t_amplitudes(0, iterative=False)
        ]
        l_new = [
            1j * l_x for l_x in self._compute_l_amplitudes(0, iterative=False)
        ]

        return (l_new, t_new)

    def evolve_amplitudes(self, t_start, t_end, num_timesteps):
        prob = np.zeros(num_timesteps, dtype=np.complex128)
        time = np.zeros(num_timesteps)

        time[0] = t_start
        h = (t_end - t_start) / (num_timesteps - 1)

        self._l_0 = self._get_l_copy()
        self._t_0 = self._get_t_copy()
        prob[0] = self._compute_time_evolution_probability()

        assert abs(prob[0].real - 1.0) < 1e-8

        l_i = self._l_0
        t_i = self._t_0
        for i in tqdm.tqdm(range(1, num_timesteps)):
            time[i] = i * h

            k_1_l, k_1_t = self._timestep(l_i, t_i, time[i])

            l_i2 = [l_x + h * k_1 / 2.0 for l_x, k_1 in zip(l_i, k_1_l)]
            t_i2 = [t_x + h * k_1 / 2.0 for t_x, k_1 in zip(t_i, k_1_t)]
            k_2_l, k_2_t = self._timestep(l_i2, t_i2, time[i] + h / 2.0)

            l_i3 = [l_x + h * k_2 / 2.0 for l_x, k_2 in zip(l_i, k_2_l)]
            t_i3 = [t_x + h * k_2 / 2.0 for t_x, k_2 in zip(t_i, k_2_t)]
            k_3_l, k_3_t = self._timestep(l_i3, t_i3, time[i] + h / 2.0)

            l_i4 = [l_x + h * k_3 for l_x, k_3 in zip(l_i, k_3_l)]
            t_i4 = [t_x + h * k_3 for t_x, k_3 in zip(t_i, k_3_t)]
            k_4_l, k_4_t = self._timestep(l_i4, t_i4, time[i] + h)

            l_i = [
                l_x + h / 6.0 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
                for l_x, k_1, k_2, k_3, k_4 in zip(
                    l_i, k_1_l, k_2_l, k_3_l, k_4_l
                )
            ]
            t_i = [
                t_x + h / 6.0 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
                for t_x, k_1, k_2, k_3, k_4 in zip(
                    t_i, k_1_t, k_2_t, k_3_t, k_4_t
                )
            ]

            self._set_l(l_i)
            self._set_t(t_i)

            prob[i] = self._compute_time_evolution_probability()

        return prob, time

    def compute_spin_reduced_one_body_density_matrix(self):
        rho_qp = self._compute_one_body_density_matrix()

        if np.abs(np.trace(rho_qp) - self.n) > 1e-8:
            warn = "Trace of rho_qp = {0} != {1} = number of particles"
            warn = warn.format(np.trace(rho_qp), self.n)
            warnings.warn(warn)

        rho_qp_reduced = compute_spin_reduced_one_body_density_matrix(rho_qp)
        rho = np.zeros(self.system.spf.shape[1:], dtype=self.system.spf.dtype)
        spf_slice = slice(0, self.system.spf.shape[0])

        for _i in np.ndindex(rho.shape):
            i = (spf_slice, *_i)
            rho[_i] += np.dot(
                self.system.spf[i].conj(),
                np.dot(rho_qp_reduced, self.system.spf[i]),
            )

        return rho

    def compute_reference_energy(self):
        return compute_reference_energy(self.f, self.u, self.o, self.v, np=np)

    def compute_l_amplitudes(self, max_iterations=100, tol=1e-4, theta=0.1):
        assert 0 <= theta <= 1, "Mixing parameter theta must be in [0, 1]"

        iterations = 0

        l_list = self._get_l_copy()
        diff_l = [100 for l in l_list]

        while (
            any([d_l > tol for d_l in diff_l])
        ) and iterations < max_iterations:
            if self.verbose:
                print(
                    "Iteration: {0}\tDiff (l): {1}".format(iterations, diff_l)
                )

            self._compute_l_amplitudes(theta, iterative=True)
            new_l_list = self._get_l_copy()
            diff_l = [
                np.amax(np.abs(l - new_l))
                for l, new_l in zip(l_list, new_l_list)
            ]

            l_list = new_l_list
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
                print(
                    "Iteration: {0}\tEnergy: {1}".format(
                        iterations, energy.real
                    )
                )

            self._compute_t_amplitudes(theta, iterative=True)
            energy_prev = energy
            energy = self._compute_energy()
            diff = abs(energy - energy_prev)
            iterations += 1

        return energy, iterations
