from scipy.linalg import expm

from coupled_cluster.ccd.ccd import CoupledClusterDoubles
from coupled_cluster.cc_helper import (
    transform_two_body_tensor,
    construct_d_t_2_matrix,
)


class OACCD(CoupledClusterDoubles):
    """Implementation of the non-orthogonal coupled cluster doubles method. The
    code is based on a script written by Rolf H. Myhre and Simen Kvaal.

    https://doi.org/10.1063/1.5006160
    """

    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

        np = self.np
        n, m, l = self.n, self.m, self.l

        self.kappa = np.zeros((l, l), dtype=self.t_2.dtype)

        self.kappa_up = np.zeros((m, n), dtype=self.t_2.dtype)
        self.kappa_down = np.zeros((n, m), dtype=self.t_2.dtype)

    def setup_kappa_mixer(self, **kwargs):
        self.kappa_up_mixer = self.mixer(**kwargs)
        self.kappa_down_mixer = self.mixer(**kwargs)

    def compute_ground_state(
        self, max_iterations=100, tol=1e-4, **mixer_kwargs
    ):
        np = self.np

        if not np in mixer_kwargs:
            mixer_kwargs["np"] = np

        self.setup_kappa_mixer(**mixer_kwargs)

        kappa_diff = 100

        for i in range(max_iterations):
            if self.verbose:
                print(f"Iteration: {i}\tDiff (kappa): {kappa_diff}")

            if kappa_diff < tol:
                break

            self.S = expm(self.kappa)
            self.S_inv = expm(-self.kappa)

            self.h = self.S_inv @ self.system.h @ self.S
            self.u = transform_two_body_tensor(
                self.system.u, self.S, self.S_inv, np
            )
            self.f = self.system.construct_fock_matrix(self.h, self.u)

            self.d_t_2 = construct_d_t_2_matrix(self.f, self.o, self.v, np)
            self.d_l_2 = self.d_t_2.transpose(2, 3, 0, 1).copy()

            self.iterate_t_amplitudes(
                max_iterations=max_iterations, tol=tol, **mixer_kwargs
            )

            self.iterate_l_amplitudes(
                max_iterations=max_iterations, tol=tol, **mixer_kwargs
            )
