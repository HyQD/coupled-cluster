from scipy.linalg import expm

from coupled_cluster.ccd.ccd import CoupledClusterDoubles
from coupled_cluster.cc_helper import (
    transform_two_body_tensor,
    construct_d_t_1_matrix,
    construct_d_t_2_matrix,
    remove_diagonal_in_matrix,
)

from coupled_cluster.ccd.rhs_t import compute_t_2_amplitudes
from coupled_cluster.ccd.rhs_l import compute_l_2_amplitudes


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
        self,
        max_iterations=100,
        tol=1e-4,
        termination_tol=1e-4,
        tol_factor=0.1,
        **mixer_kwargs,
    ):
        np = self.np

        if not np in mixer_kwargs:
            mixer_kwargs["np"] = np

        self.kappa_up_mixer = self.mixer(**mixer_kwargs)
        self.kappa_down_mixer = self.mixer(**mixer_kwargs)
        self.t_2_mixer = self.mixer(**mixer_kwargs)
        self.l_2_mixer = self.mixer(**mixer_kwargs)

        amp_tol = 0.1
        o, v = self.o, self.v
        n, m = self.n, self.m

        self.t_2.fill(0)
        self.l_2.fill(0)

        for k_it in range(max_iterations):
            S = expm(self.kappa)
            S_inv = expm(-self.kappa)

            self.u = np.einsum(
                "pP,qQ,PQRS,Rr,Ss->pqrs",
                S_inv,
                S_inv,
                self.system.u,
                S,
                S,
                optimize=True,
            )
            self.h = np.einsum(
                "pP,PQ,Qq->pq", S_inv, self.system.h, S, optimize=True
            )
            self.f = self.system.construct_fock_matrix(self.h, self.u)

            print(f"\nIteration: {k_it}")

            d_t_1 = construct_d_t_1_matrix(self.f, o, v, np)
            d_l_1 = d_t_1.T.copy()
            d_t_2 = construct_d_t_2_matrix(self.f, o, v, np)
            d_l_2 = d_t_2.transpose(2, 3, 0, 1).copy()

            self.t_2_mixer.clear_vectors()
            self.l_2_mixer.clear_vectors()

            for t_it in range(max_iterations):
                self.rhs_t_2.fill(0)

                compute_t_2_amplitudes(
                    self.f,
                    self.u,
                    self.t_2,
                    self.o,
                    self.v,
                    np,
                    out=self.rhs_t_2,
                )

                self.t_2 = self.t_2_mixer.compute_new_vector(
                    self.t_2, self.rhs_t_2 / d_t_2, -self.rhs_t_2
                )

                residual_t_2 = np.linalg.norm(self.rhs_t_2)

                if np.abs(residual_t_2) < amp_tol:
                    break

            print(f"\nT converged in {t_it} iterations")
            print(f"T residual is {residual_t_2}")

            for l_it in range(max_iterations):
                self.rhs_l_2.fill(0)

                compute_l_2_amplitudes(
                    self.f,
                    self.u,
                    self.t_2,
                    self.l_2,
                    self.o,
                    self.v,
                    np,
                    out=self.rhs_l_2,
                )

                self.l_2 = self.l_2_mixer.compute_new_vector(
                    self.l_2, self.rhs_l_2 / d_l_2, -self.rhs_l_2
                )

                residual_l_2 = np.linalg.norm(self.rhs_l_2)

                if np.abs(residual_l_2) < amp_tol:
                    break

            print(f"\nLambda converged in {l_it} iterations")
            print(f"Lambda residual is {residual_l_2}")

            Ku_der = Ku_der_fun(
                self.n,
                self.m,
                self.o,
                self.v,
                self.t_2.transpose(2, 3, 0, 1),
                self.l_2,
                self.f,
                self.u,
                np,
            )
            Kd_der = Kd_der_fun(
                self.n,
                self.m,
                self.o,
                self.v,
                self.t_2.transpose(2, 3, 0, 1),
                self.l_2,
                self.f,
                self.u,
                np,
            )

            residual_up = np.linalg.norm(Ku_der)
            residual_down = np.linalg.norm(Kd_der)

            print(f"\nResidual norms: ru = {residual_up}")
            print(f"Residual norms: rd = {residual_down}")

            if np.abs(residual_up) < tol and np.abs(residual_down) < tol:
                break

            self.kappa_up = self.kappa_up_mixer.compute_new_vector(
                self.kappa_up, -Kd_der / d_t_1, Kd_der
            )
            self.kappa_down = self.kappa_down_mixer.compute_new_vector(
                self.kappa_down, -Ku_der.T / d_l_1, Ku_der.T
            )

            self.kappa[v, o] = self.kappa_up
            self.kappa[o, v] = self.kappa_down

            amp_tol = min(residual_up, residual_down) * tol_factor
            amp_tol = max(amp_tol, termination_tol)

            print("Total NOCCD energy: {0}".format(self.compute_energy()))

    # def compute_ground_state(
    #    self,
    #    max_iterations=100,
    #    tol=1e-4,
    #    termination_tol=1e-4,
    #    tol_factor=0.1,
    #    **mixer_kwargs,
    # ):
    #    # Note: The tolerance in the t- and l-amplitudes can be lowered.
    #    np = self.np

    #    if not np in mixer_kwargs:
    #        mixer_kwargs["np"] = np

    #    self.setup_kappa_mixer(**mixer_kwargs)
    #    self.setup_t_mixer(**mixer_kwargs)
    #    self.setup_l_mixer(**mixer_kwargs)

    #    amp_tol = 0.1
    #    # self.t_2.fill(0)
    #    # self.l_2.fill(0)

    #    for i in range(max_iterations):
    #        self.S = expm(self.kappa)
    #        self.S_inv = expm(-self.kappa)

    #        self.h = self.S_inv @ self.system.h @ self.S
    #        self.u = transform_two_body_tensor(
    #            self.system.u, self.S, self.S_inv, np
    #        )
    #        self.f = self.system.construct_fock_matrix(self.h, self.u)
    #        self.off_diag_f = remove_diagonal_in_matrix(self.f, np)

    #        self.d_t_1 = construct_d_t_1_matrix(self.f, self.o, self.v, np)
    #        self.d_t_2 = construct_d_t_2_matrix(self.f, self.o, self.v, np)
    #        self.d_l_1 = self.d_t_1.T.copy()
    #        self.d_l_2 = self.d_t_2.transpose(2, 3, 0, 1).copy()

    #        # self.t_2_mixer.clear_vectors()
    #        # self.l_2_mixer.clear_vectors()

    #        # for j in range(max_iterations):
    #        #    self.rhs_t_2.fill(0)
    #        #    compute_t_2_amplitudes(
    #        #        self.f,
    #        #        self.u,
    #        #        self.t_2,
    #        #        self.o,
    #        #        self.v,
    #        #        out=self.rhs_t_2,
    #        #        np=np
    #        #    )
    #        #    self.t_2 = self.t_2_mixer.compute_new_vector(
    #        #        self.t_2, self.rhs_t_2 / self.d_t_2, self.rhs_t_2
    #        #    )

    #        #    residual_t_2 = np.linalg.norm(self.rhs_t_2)

    #        #    if abs(residual_t_2) < amp_tol:
    #        #        print(f"Tol t_2: {residual_t_2}\tIterations: {j}")
    #        #        break

    #        # for j in range(max_iterations):
    #        #    self.rhs_l_2.fill(0)
    #        #    compute_l_2_amplitudes(
    #        #        self.f,
    #        #        self.u,
    #        #        self.t_2,
    #        #        self.l_2,
    #        #        self.o,
    #        #        self.v,
    #        #        out=self.rhs_l_2,
    #        #        np=np
    #        #    )
    #        #    self.l_2 = self.l_2_mixer.compute_new_vector(
    #        #        self.l_2, self.rhs_l_2 / self.d_l_2, self.rhs_l_2
    #        #    )

    #        #    residual_l_2 = np.linalg.norm(self.rhs_l_2)

    #        #    if abs(residual_l_2) < amp_tol:
    #        #        print(f"Tol l_2: {residual_l_2}\tIterations: {j}")
    #        #        break

    #        self.iterate_t_amplitudes(
    #            max_iterations=max_iterations, tol=amp_tol, **mixer_kwargs
    #        )

    #        self.iterate_l_amplitudes(
    #            max_iterations=max_iterations, tol=amp_tol, **mixer_kwargs
    #        )

    #        kappa_up_derivative = Ku_der_fun(
    #            self.n,
    #            self.m,
    #            self.o,
    #            self.v,
    #            self.t_2.transpose(2, 3, 0, 1),
    #            self.l_2,
    #            self.f,
    #            self.u,
    #            np,
    #        )
    #        kappa_down_derivative = Kd_der_fun(
    #            self.n,
    #            self.m,
    #            self.o,
    #            self.v,
    #            self.t_2.transpose(2, 3, 0, 1),
    #            self.l_2,
    #            self.f,
    #            self.u,
    #            np,
    #        )

    #        residual_up = np.linalg.norm(kappa_up_derivative)
    #        residual_down = np.linalg.norm(kappa_down_derivative)

    #        if self.verbose:
    #            print(
    #                (
    #                    f"Iteration: {i}\n"
    #                    + f"Residual up: {residual_up}\n"
    #                    + f"Residual down: {residual_down}"
    #                )
    #            )

    #        if abs(residual_up) < tol and abs(residual_down) < tol:
    #            break

    #        self.kappa_up = self.kappa_up_mixer.compute_new_vector(
    #            self.kappa_up,
    #            -kappa_down_derivative / self.d_t_1,
    #            kappa_down_derivative,
    #        )
    #        self.kappa_down = self.kappa_down_mixer.compute_new_vector(
    #            self.kappa_down,
    #            -kappa_up_derivative.T / self.d_t_1.T,
    #            kappa_up_derivative.T,
    #        )

    #        self.kappa[self.o, self.v] = self.kappa_down
    #        self.kappa[self.v, self.o] = self.kappa_up

    #        amp_tol = min(residual_up, residual_down) * tol_factor
    #        amp_tol = max(amp_tol, termination_tol)

    #        print("Energy: {0}".format(self.compute_energy()))


def Ku_der_fun(nocc, nvirt, o, v, T2, L2, F, W, np):
    result = np.zeros((nvirt, nocc))
    result += 0.5 * np.einsum(
        "Ikcd,cdAk->AI", L2, W[v, v, v, o], optimize=["einsum_path", (0, 1)]
    )
    result += -0.5 * np.einsum(
        "lkAc,Iclk->AI", L2, W[o, v, o, o], optimize=["einsum_path", (0, 1)]
    )
    result += 0.5 * np.einsum(
        "lkAc,lkcd,Id->AI",
        L2,
        T2,
        F[o, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "Ikcd,lkec,dlAe->AI",
        L2,
        T2,
        W[v, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "lkAc,mkcd,Imdl->AI",
        L2,
        T2,
        W[o, o, v, o],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "Ikcd,lkcd,lA->AI",
        L2,
        T2,
        F[o, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "lkcd,mkcd,ImAl->AI",
        L2,
        T2,
        W[o, o, v, o],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "lkcd,lkec,IdAe->AI",
        L2,
        T2,
        W[o, v, v, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += -0.25 * np.einsum(
        "lkAc,lked,Iced->AI",
        L2,
        T2,
        W[o, v, v, v],
        optimize=["einsum_path", (1, 2), (0, 1)],
    )
    result += 0.25 * np.einsum(
        "Ikcd,mlcd,mlAk->AI",
        L2,
        T2,
        W[o, o, v, o],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += np.einsum("IA->AI", F[o, v], optimize=["einsum_path", (0,)])
    return result


def Kd_der_fun(nocc, nvirt, o, v, T2, L2, F, W, np):
    result = np.zeros((nvirt, nocc))
    result += -1.0 * np.einsum(
        "AI->AI", F[v, o], optimize=["einsum_path", (0,)]
    )
    result += 0.5 * np.einsum(
        "lkAc,lkIc->AI", T2, W[o, o, o, v], optimize=["einsum_path", (0, 1)]
    )
    result += -0.5 * np.einsum(
        "Ikcd,Akcd->AI", T2, W[v, o, v, v], optimize=["einsum_path", (0, 1)]
    )
    result += np.einsum(
        "lkcd,mkAc,dmIl->AI",
        L2,
        T2,
        W[v, o, o, o],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += np.einsum(
        "lkcd,Ikec,Adel->AI",
        L2,
        T2,
        W[v, v, v, o],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "lkcd,Ikcd,Al->AI",
        L2,
        T2,
        F[v, o],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "lkcd,mkcd,AmIl->AI",
        L2,
        T2,
        W[v, o, o, o],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "lkcd,lkec,AdIe->AI",
        L2,
        T2,
        W[v, v, o, v],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "lkcd,lkAc,dI->AI",
        L2,
        T2,
        F[v, o],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += -0.25 * np.einsum(
        "lkcd,Imcd,Amlk->AI",
        L2,
        T2,
        W[v, o, o, o],
        optimize=["einsum_path", (0, 1), (0, 1)],
    )
    result += 0.25 * np.einsum(
        "lkcd,lkAe,cdIe->AI",
        L2,
        T2,
        W[v, v, o, v],
        optimize=["einsum_path", (0, 2), (0, 1)],
    )
    result += 0.5 * np.einsum(
        "lkcd,lkec,Imfd,Amef->AI",
        L2,
        T2,
        T2,
        W[v, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "lkcd,lnAd,mkec,mnIe->AI",
        L2,
        T2,
        T2,
        W[o, o, o, v],
        optimize=["einsum_path", (2, 3), (0, 2), (0, 1)],
    )
    result += -1.0 * np.einsum(
        "lkcd,mkec,Ilfd,Amef->AI",
        L2,
        T2,
        T2,
        W[v, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    )
    result += -0.5 * np.einsum(
        "lkcd,lnAe,mkcd,mnIe->AI",
        L2,
        T2,
        T2,
        W[o, o, o, v],
        optimize=["einsum_path", (0, 2), (1, 2), (0, 1)],
    )
    result += -0.125 * np.einsum(
        "lkcd,Imcd,lkef,Amef->AI",
        L2,
        T2,
        T2,
        W[v, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    )
    result += 0.25 * np.einsum(
        "lkcd,lkAd,mnec,mnIe->AI",
        L2,
        T2,
        T2,
        W[o, o, o, v],
        optimize=["einsum_path", (2, 3), (0, 2), (0, 1)],
    )
    result += 0.25 * np.einsum(
        "lkcd,mnAd,lkec,mnIe->AI",
        L2,
        T2,
        T2,
        W[o, o, o, v],
        optimize=["einsum_path", (0, 2), (1, 2), (0, 1)],
    )
    result += 0.25 * np.einsum(
        "lkcd,Ilcd,mkef,Amef->AI",
        L2,
        T2,
        T2,
        W[v, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    )
    result += 0.25 * np.einsum(
        "lkcd,mkcd,Ilef,Amef->AI",
        L2,
        T2,
        T2,
        W[v, o, v, v],
        optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    )
    result += 0.125 * np.einsum(
        "lkcd,lkAe,mncd,mnIe->AI",
        L2,
        T2,
        T2,
        W[o, o, o, v],
        optimize=["einsum_path", (0, 2), (1, 2), (0, 1)],
    )
    return result
