from coupled_cluster.tdcc import TimeDependentCoupledCluster

from coupled_cluster.rcc2.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)

from coupled_cluster.rcc2.rhs_l import (
    compute_l_1_amplitudes,
    compute_l_2_amplitudes,
)

from coupled_cluster.rcc2 import RCC2
from coupled_cluster.rcc2.energies import (
    compute_time_dependent_energy,
    compute_ground_state_energy_correction,
)

from coupled_cluster.rcc2.density_matrices import (
    compute_one_body_density_matrix,
)

from coupled_cluster.rcc2.time_dependent_overlap import (
    compute_time_dependent_overlap,
)

from coupled_cluster.cc_helper import AmplitudeContainer

from opt_einsum import contract


class TDRCC2(TimeDependentCoupledCluster):
    truncation = "CCSD"

    def __init__(self, system, cc2_b=False):
        super().__init__(system)
        self.cc2_b = cc2_b
        self.rcc2 = RCC2(system)
        self.h_t = self.system.h.copy()
        self.f0 = self.system.construct_fock_matrix(self.h, self.u)

    def update_hamiltonian(self, current_time, y):
        # print(f"Update Hamiltonian TDRCC2")
        if self.last_timestep == current_time:
            return

        self.last_timestep = current_time

        if self.system.has_one_body_time_evolution_operator:
            self.h = self.system.h_t(current_time)

        if self.system.has_two_body_time_evolution_operator:
            self.u = self.system.u_t(current_time)

        self.v_t = self.system.v_t(current_time)

        self.h_t = self.system.h + self.v_t

        self.f = self.system.construct_fock_matrix(self.h, self.u)

    def __call__(self, current_time, prev_amp):
        o, v = self.system.o, self.system.v

        prev_amp = self._amp_template.from_array(prev_amp)
        t_old, l_old = prev_amp
        t_0, t_1, t_2 = t_old

        self.update_hamiltonian(current_time, prev_amp)

        ##################################################################
        # T1-transform integrals
        # self.rcc2.t1_transform_integrals(t_1, self.h, self.u) only transforms h and u.
        # However, we also need the t1 transform of the bare matrix elements of the interaction operator, v_t1.
        x_transform = self.np.zeros(
            (self.system.l, self.system.l), dtype=t_1.dtype
        )
        y_transform = self.np.zeros(
            (self.system.l, self.system.l), dtype=t_1.dtype
        )

        x_transform += self.np.eye(self.system.l)
        x_transform[self.system.v, self.system.o] -= t_1

        y_transform += self.np.eye(self.system.l)
        y_transform[self.system.o, self.system.v] += t_1.T

        C_tilde = x_transform
        C = y_transform.T

        h_t1 = self.system.transform_one_body_elements(
            self.system.h, C, C_tilde
        )
        v_t1 = self.system.transform_one_body_elements(self.v_t, C, C_tilde)
        u_t1 = self.system.transform_two_body_elements(self.u, C, C_tilde)

        f1 = self.system.construct_fock_matrix(h_t1 + v_t1, u_t1)
        if self.cc2_b:
            f2 = f1.copy()
        else:
            f2 = self.f0 + v_t1

        ######################################################################
        # from scipy.linalg import expm
        # kappa = self.np.zeros((self.system.l, self.system.l), dtype=t_1.dtype)
        # kappa[v, o] += t_1
        # _C = expm(kappa)
        # _C_tilde = expm(-kappa)
        # Potential test for CC2: np.allclose(C, _C) and np.allclose(C_tilde, _C_tilde)
        ######################################################################

        # Remove phase from t-amplitude list
        t_old = t_old[1:]

        t_new = [
            -1j
            * rhs_t_func(
                f2,
                f1,
                u_t1,
                *t_old,
                o,
                v,
                np=self.np,
            )
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        # Compute derivative of phase
        t_0_new = -1j * self.rhs_t_0_amplitude(
            self.f, self.u, *t_old, self.o, self.v, np=self.np
        )
        t_new = [t_0_new, *t_new]

        # Note that compute_l_1_amplitudes(...) has a different signature from compute_l_2_amplitudes(...).
        # In particular, it requires the bare v_t1 matrix elements as input.
        l1_new = 1j * compute_l_1_amplitudes(
            f2,
            f1,
            v_t1,
            u_t1,
            *t_old,
            *l_old,
            o,
            v,
            np=self.np,
        )

        l2_new = 1j * compute_l_2_amplitudes(
            f2,
            f1,
            u_t1,
            *t_old,
            *l_old,
            o,
            v,
            np=self.np,
        )
        l_new = [l1_new, l2_new]

        self.last_timestep = current_time

        return AmplitudeContainer(t=t_new, l=l_new, np=self.np).asarray()

    def rhs_t_0_amplitude(self, *args, **kwargs):
        return self.np.array(
            [
                self.system.compute_reference_energy()
                + compute_ground_state_energy_correction(*args, **kwargs)
            ]
        )

    def rhs_t_amplitudes(self):
        yield compute_t_1_amplitudes
        yield compute_t_2_amplitudes

    def rhs_l_amplitudes(self):
        yield compute_l_1_amplitudes
        yield compute_l_2_amplitudes

    def compute_left_reference_overlap(self, current_time, y):
        np = self.np

        t_0, t_1, t_2, l_1, l_2 = self._amp_template.from_array(y).unpack()

        val = 1
        val -= 0.5 * contract("ijab,abij->", l_2, t_2)
        val += 0.5 * contract("ai,bj,ijab->", t_1, t_1, l_2, optimize=True)
        val -= contract("ia,ai->", l_1, t_1)

        return val

    def compute_energy(self, current_time, y):
        t_0, t_1, t_2, l_1, l_2 = self._amp_template.from_array(y).unpack()

        self.update_hamiltonian(current_time, y)

        (
            self.h_transformed,
            self.f_transformed,
            self.u_transformed,
        ) = self.rcc2.t1_transform_integrals(t_1, self.h, self.u)

        return (
            compute_time_dependent_energy(
                self.f,
                self.f_transformed,
                self.u_transformed,
                t_1,
                t_2,
                l_1,
                l_2,
                self.system.o,
                self.system.v,
                np=self.np,
            )
            + self.system.compute_reference_energy()
        )

    def compute_one_body_density_matrix(self, current_time, y):
        t_0, t_1, t_2, l_1, l_2 = self._amp_template.from_array(y).unpack()
        return compute_one_body_density_matrix(
            t_1, t_2, l_1, l_2, self.o, self.v, np=self.np
        )

    def compute_two_body_density_matrix(self, current_time, y):
        pass

    def compute_overlap(self, current_time, y_a, y_b, use_old=False):
        t0a, t1a, t2a, l1a, l2a = self._amp_template.from_array(y_a).unpack()
        t0b, t1b, t2b, l1b, l2b = self._amp_template.from_array(y_b).unpack()

        return compute_time_dependent_overlap(
            t1a,
            t2a,
            l1a,
            l2a,
            t0b,
            t1b,
            t2b,
            l1b,
            l2b,
            np=self.np,
            use_old=use_old,
        )
