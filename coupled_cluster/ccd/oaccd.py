from scipy.linalg import expm

from coupled_cluster.ccd.ccd import CoupledClusterDoubles
from coupled_cluster.cc_helper import (
    transform_two_body_tensor,
    construct_d_t_1_matrix,
    construct_d_t_2_matrix,
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

        self.setup_kappa_mixer(**mixer_kwargs)

        amp_tol = 0.1

        self.t_2.fill(0)
        self.l_2.fill(0)

        for k_it in range(max_iterations):
            S = expm(self.kappa)
            S_inv = expm(-self.kappa)

            self.h = S_inv @ self.system.h @ S
            self.u = transform_two_body_tensor(self.system.u, S, S_inv, np)
            self.f = self.system.construct_fock_matrix(self.h, self.u)

            print(f"\nIteration: {k_it}")

            d_t_1 = construct_d_t_1_matrix(self.f, self.o, self.v, np)
            d_l_1 = d_t_1.T.copy()
            self.d_t_2 = construct_d_t_2_matrix(self.f, self.o, self.v, np)
            self.d_l_2 = self.d_t_2.transpose(2, 3, 0, 1).copy()

            self.iterate_t_amplitudes(
                max_iterations=max_iterations, tol=amp_tol, **mixer_kwargs
            )
            self.iterate_l_amplitudes(
                max_iterations=max_iterations, tol=amp_tol, **mixer_kwargs
            )

            kappa_down_rhs = compute_kappa_down_rhs(
                self.f, self.u, self.t_2, self.l_2, self.o, self.v, np
            )
            kappa_up_rhs = compute_kappa_up_rhs(
                self.f, self.u, self.t_2, self.l_2, self.o, self.v, np
            )

            residual_up = np.linalg.norm(kappa_up_rhs)
            residual_down = np.linalg.norm(kappa_down_rhs)

            print(f"\nResidual norms: rd = {residual_down}")
            print(f"Residual norms: ru = {residual_up}")

            if np.abs(residual_up) < tol and np.abs(residual_down) < tol:
                break

            self.kappa_up = self.kappa_up_mixer.compute_new_vector(
                self.kappa_up, -kappa_up_rhs / d_t_1, kappa_up_rhs
            )
            self.kappa_down = self.kappa_down_mixer.compute_new_vector(
                self.kappa_down, -kappa_down_rhs / d_l_1, kappa_down_rhs
            )

            self.kappa[self.v, self.o] = self.kappa_up
            self.kappa[self.o, self.v] = self.kappa_down

            amp_tol = min(residual_up, residual_down) * tol_factor
            amp_tol = max(amp_tol, termination_tol)

            print("Total NOCCD energy: {0}".format(self.compute_energy()))


def compute_kappa_down_rhs(f, u, t_2, l_2, o, v, np):
    # L2 = l_2
    # F = f
    # W = u
    # nocc = o.stop
    # nvirt = v.stop - o.stop

    # T2 = t_2.transpose(2, 3, 0, 1)
    # result = np.zeros((nvirt, nocc))
    # result += 0.5 * np.einsum(
    #    "Ikcd,cdAk->AI", L2, W[v, v, v, o], optimize=["einsum_path", (0, 1)]
    # )

    # Result should be kappa_down -> kappa^{i}_{a}
    res = 0.5 * np.tensordot(l_2, u[v, v, v, o], axes=((1, 2, 3), (3, 0, 1)))
    # np.testing.assert_allclose(result.T, res)

    # result += -0.5 * np.einsum(
    #    "lkAc,Iclk->AI", L2, W[o, v, o, o], optimize=["einsum_path", (0, 1)]
    # )

    res -= 0.5 * np.tensordot(u[o, v, o, o], l_2, axes=((1, 2, 3), (3, 0, 1)))
    # np.testing.assert_allclose(result.T, res)

    # result += 0.5 * np.einsum(
    #    "lkAc,lkcd,Id->AI",
    #    L2,
    #    T2,
    #    F[o, v],
    #    optimize=["einsum_path", (1, 2), (0, 1)],
    # )

    temp_da = 0.5 * np.tensordot(t_2, l_2, axes=((0, 2, 3), (3, 0, 1)))
    res += np.dot(f[o, v], temp_da)
    # np.testing.assert_allclose(result.T, res)

    # result += -1.0 * np.einsum(
    #    "Ikcd,lkec,dlAe->AI",
    #    L2,
    #    T2,
    #    W[v, o, v, v],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_elid = -np.tensordot(t_2, l_2, axes=((1, 3), (2, 1)))
    res += np.tensordot(temp_elid, u[v, o, v, v], axes=((0, 1, 3), (3, 1, 0)))
    # np.testing.assert_allclose(result.T, res)

    # result += -1.0 * np.einsum(
    #    "lkAc,mkcd,Imdl->AI",
    #    L2,
    #    T2,
    #    W[o, o, v, o],
    #    optimize=["einsum_path", (1, 2), (0, 1)],
    # )

    temp_dmla = -np.tensordot(t_2, l_2, axes=((0, 3), (3, 1)))
    res += np.tensordot(u[o, o, v, o], temp_dmla, axes=((1, 2, 3), (1, 0, 2)))
    # np.testing.assert_allclose(result.T, res)

    # result += -0.5 * np.einsum(
    #    "Ikcd,lkcd,lA->AI",
    #    L2,
    #    T2,
    #    F[o, v],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_il = -0.5 * np.tensordot(l_2, t_2, axes=((1, 2, 3), (3, 0, 1)))
    res += np.dot(temp_il, f[o, v])
    # np.testing.assert_allclose(result.T, res)

    # result += -0.5 * np.einsum(
    #    "lkcd,mkcd,ImAl->AI",
    #    L2,
    #    T2,
    #    W[o, o, v, o],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_lm = -0.5 * np.tensordot(l_2, t_2, axes=((1, 2, 3), (3, 0, 1)))
    res += np.tensordot(u[o, o, v, o], temp_lm, axes=((1, 3), (1, 0)))
    # np.testing.assert_allclose(result.T, res)

    # result += -0.5 * np.einsum(
    #    "lkcd,lkec,IdAe->AI",
    #    L2,
    #    T2,
    #    W[o, v, v, v],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_de = -0.5 * np.tensordot(l_2, t_2, axes=((0, 1, 2), (2, 3, 1)))
    res += np.tensordot(u[o, v, v, v], temp_de, axes=((1, 3), (0, 1)))
    # np.testing.assert_allclose(result.T, res)

    # result += -0.25 * np.einsum(
    #    "lkAc,lked,Iced->AI",
    #    L2,
    #    T2,
    #    W[o, v, v, v],
    #    optimize=["einsum_path", (1, 2), (0, 1)],
    # )

    temp_edac = -0.25 * np.tensordot(t_2, l_2, axes=((2, 3), (0, 1)))
    res += np.tensordot(u[o, v, v, v], temp_edac, axes=((1, 2, 3), (3, 0, 1)))
    # np.testing.assert_allclose(result.T, res)

    # result += 0.25 * np.einsum(
    #    "Ikcd,mlcd,mlAk->AI",
    #    L2,
    #    T2,
    #    W[o, o, v, o],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_ikml = 0.25 * np.tensordot(l_2, t_2, axes=((2, 3), (0, 1)))
    res += np.tensordot(temp_ikml, u[o, o, v, o], axes=((1, 2, 3), (3, 0, 1)))
    # np.testing.assert_allclose(result.T, res)

    # result += np.einsum("IA->AI", F[o, v], optimize=["einsum_path", (0,)])

    res += f[o, v]
    # np.testing.assert_allclose(result.T, res, atol=1e-10)

    # return result.T.copy()
    return res


def compute_kappa_up_rhs(f, u, t_2, l_2, o, v, np):
    # L2 = l_2
    # F = f
    # W = u
    # nocc = o.stop
    # nvirt = v.stop - o.stop

    # T2 = t_2.transpose(2, 3, 0, 1)
    # result = np.zeros((nvirt, nocc))
    # result += -1.0 * np.einsum(
    #    "AI->AI", F[v, o], optimize=["einsum_path", (0,)]
    # )

    res = -f[v, o]
    # np.testing.assert_allclose(result, res)

    # result += 0.5 * np.einsum(
    #    "lkAc,lkIc->AI", T2, W[o, o, o, v], optimize=["einsum_path", (0, 1)]
    # )

    res += 0.5 * np.tensordot(t_2, u[o, o, o, v], axes=((1, 2, 3), (3, 0, 1)))
    # np.testing.assert_allclose(result, res)

    # result += -0.5 * np.einsum(
    #    "Ikcd,Akcd->AI", T2, W[v, o, v, v], optimize=["einsum_path", (0, 1)]
    # )

    res -= 0.5 * np.tensordot(u[v, o, v, v], t_2, axes=((1, 2, 3), (3, 0, 1)))
    # np.testing.assert_allclose(result, res)

    # result += np.einsum(
    #    "lkcd,mkAc,dmIl->AI",
    #    L2,
    #    T2,
    #    W[v, o, o, o],
    #    optimize=["einsum_path", (0, 2), (0, 1)],
    # )

    temp_amld = np.tensordot(t_2, l_2, axes=((1, 3), (2, 1)))
    res += np.tensordot(temp_amld, u[v, o, o, o], axes=((1, 2, 3), (1, 3, 0)))
    # np.testing.assert_allclose(result, res)

    # result += np.einsum(
    #    "lkcd,Ikec,Adel->AI",
    #    L2,
    #    T2,
    #    W[v, v, v, o],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_eild = np.tensordot(t_2, l_2, axes=((1, 3), (2, 1)))
    res += np.tensordot(u[v, v, v, o], temp_eild, axes=((1, 2, 3), (3, 0, 2)))
    # np.testing.assert_allclose(result, res)

    # result += 0.5 * np.einsum(
    #    "lkcd,Ikcd,Al->AI",
    #    L2,
    #    T2,
    #    F[v, o],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_li = 0.5 * np.tensordot(l_2, t_2, axes=((1, 2, 3), (3, 0, 1)))
    res += np.dot(f[v, o], temp_li)
    # np.testing.assert_allclose(result, res)

    # result += 0.5 * np.einsum(
    #    "lkcd,mkcd,AmIl->AI",
    #    L2,
    #    T2,
    #    W[v, o, o, o],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_lm = 0.5 * np.tensordot(l_2, t_2, axes=((1, 2, 3), (3, 0, 1)))
    res += np.tensordot(u[v, o, o, o], temp_lm, axes=((1, 3), (1, 0)))
    # np.testing.assert_allclose(result, res)

    # result += 0.5 * np.einsum(
    #    "lkcd,lkec,AdIe->AI",
    #    L2,
    #    T2,
    #    W[v, v, o, v],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_ed = 0.5 * np.tensordot(t_2, l_2, axes=((1, 2, 3), (2, 0, 1)))
    res += np.tensordot(u[v, v, o, v], temp_ed, axes=((1, 3), (1, 0)))
    # np.testing.assert_allclose(result, res)

    # result += -0.5 * np.einsum(
    #    "lkcd,lkAc,dI->AI",
    #    L2,
    #    T2,
    #    F[v, o],
    #    optimize=["einsum_path", (0, 2), (0, 1)],
    # )

    temp_ad = -0.5 * np.tensordot(t_2, l_2, axes=((1, 2, 3), (2, 0, 1)))
    res += np.dot(temp_ad, f[v, o])
    # np.testing.assert_allclose(result, res)

    # result += -0.25 * np.einsum(
    #    "lkcd,Imcd,Amlk->AI",
    #    L2,
    #    T2,
    #    W[v, o, o, o],
    #    optimize=["einsum_path", (0, 1), (0, 1)],
    # )

    temp_lkim = -0.25 * np.tensordot(l_2, t_2, axes=((2, 3), (0, 1)))
    res += np.tensordot(u[v, o, o, o], temp_lkim, axes=((1, 2, 3), (3, 0, 1)))
    # np.testing.assert_allclose(result, res)

    # result += 0.25 * np.einsum(
    #    "lkcd,lkAe,cdIe->AI",
    #    L2,
    #    T2,
    #    W[v, v, o, v],
    #    optimize=["einsum_path", (0, 2), (0, 1)],
    # )

    temp_lkie = 0.25 * np.tensordot(l_2, u[v, v, o, v], axes=((2, 3), (0, 1)))
    res += np.tensordot(t_2, temp_lkie, axes=((1, 2, 3), (3, 0, 1)))
    # np.testing.assert_allclose(result, res)

    # result += 0.5 * np.einsum(
    #    "lkcd,lkec,Imfd,Amef->AI",
    #    L2,
    #    T2,
    #    T2,
    #    W[v, o, v, v],
    #    optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    # )

    temp_ed = 0.5 * np.tensordot(t_2, l_2, axes=((1, 2, 3), (2, 0, 1)))
    temp_aedi = np.tensordot(u[v, o, v, v], t_2, axes=((1, 3), (3, 0)))
    res += np.tensordot(temp_aedi, temp_ed, axes=((1, 2), (0, 1)))
    # np.testing.assert_allclose(result, res)

    # result += -1.0 * np.einsum(
    #    "lkcd,lnAd,mkec,mnIe->AI",
    #    L2,
    #    T2,
    #    T2,
    #    W[o, o, o, v],
    #    optimize=["einsum_path", (2, 3), (0, 2), (0, 1)],
    # )

    temp_ankc = -np.tensordot(t_2, l_2, axes=((1, 2), (3, 0)))
    temp_anem = np.tensordot(temp_ankc, t_2, axes=((2, 3), (3, 1)))
    res += np.tensordot(temp_anem, u[o, o, o, v], axes=((1, 2, 3), (1, 3, 0)))
    # np.testing.assert_allclose(result, res)

    # result += -1.0 * np.einsum(
    #    "lkcd,mkec,Ilfd,Amef->AI",
    #    L2,
    #    T2,
    #    T2,
    #    W[v, o, v, v],
    #    optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    # )

    temp_ldem = -np.tensordot(l_2, t_2, axes=((1, 2), (3, 0)))
    temp_emfi = np.tensordot(temp_ldem, t_2, axes=((0, 1), (3, 0)))
    res += np.tensordot(u[v, o, v, v], temp_emfi, axes=((1, 2, 3), (1, 0, 2)))
    # np.testing.assert_allclose(result, res)

    # result += -0.5 * np.einsum(
    #    "lkcd,lnAe,mkcd,mnIe->AI",
    #    L2,
    #    T2,
    #    T2,
    #    W[o, o, o, v],
    #    optimize=["einsum_path", (0, 2), (1, 2), (0, 1)],
    # )

    temp_lm = -0.5 * np.tensordot(l_2, t_2, axes=((1, 2, 3), (3, 0, 1)))
    temp_lnie = np.tensordot(temp_lm, u[o, o, o, v], axes=((1), (0)))
    res += np.tensordot(t_2, temp_lnie, axes=((1, 2, 3), (3, 0, 1)))
    # np.testing.assert_allclose(result, res)

    # result += -0.125 * np.einsum(
    #    "lkcd,Imcd,lkef,Amef->AI",
    #    L2,
    #    T2,
    #    T2,
    #    W[v, o, v, v],
    #    optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    # )

    temp_lkim = -0.125 * np.tensordot(l_2, t_2, axes=((2, 3), (0, 1)))
    temp_amlk = np.tensordot(u[v, o, v, v], t_2, axes=((2, 3), (0, 1)))
    res += np.tensordot(temp_amlk, temp_lkim, axes=((1, 2, 3), (3, 0, 1)))
    # np.testing.assert_allclose(result, res)

    # result += 0.25 * np.einsum(
    #    "lkcd,lkAd,mnec,mnIe->AI",
    #    L2,
    #    T2,
    #    T2,
    #    W[o, o, o, v],
    #    optimize=["einsum_path", (2, 3), (0, 2), (0, 1)],
    # )

    temp_ac = 0.25 * np.tensordot(t_2, l_2, axes=((1, 2, 3), (3, 0, 1)))
    temp_ci = np.tensordot(t_2, u[o, o, o, v], axes=((0, 2, 3), (3, 0, 1)))
    res += np.dot(temp_ac, temp_ci)
    # np.testing.assert_allclose(result, res)

    # result += 0.25 * np.einsum(
    #    "lkcd,mnAd,lkec,mnIe->AI",
    #    L2,
    #    T2,
    #    T2,
    #    W[o, o, o, v],
    #    optimize=["einsum_path", (0, 2), (1, 2), (0, 1)],
    # )

    temp_ed = 0.25 * np.tensordot(t_2, l_2, axes=((1, 2, 3), (2, 0, 1)))
    temp_adie = np.tensordot(t_2, u[o, o, o, v], axes=((2, 3), (0, 1)))
    res += np.tensordot(temp_adie, temp_ed, axes=((1, 3), (1, 0)))
    # np.testing.assert_allclose(result, res)

    # result += 0.25 * np.einsum(
    #    "lkcd,Ilcd,mkef,Amef->AI",
    #    L2,
    #    T2,
    #    T2,
    #    W[v, o, v, v],
    #    optimize=["einsum_path", (0, 1), (0, 1), (0, 1)],
    # )

    temp_ki = 0.25 * np.tensordot(l_2, t_2, axes=((0, 2, 3), (3, 0, 1)))
    temp_ak = np.tensordot(u[v, o, v, v], t_2, axes=((1, 2, 3), (2, 0, 1)))
    res += np.dot(temp_ak, temp_ki)
    # np.testing.assert_allclose(result, res)

    # result += 0.25 * np.einsum(
    #    "lkcd,mkcd,Ilef,Amef->AI",
    #    L2,
    #    T2,
    #    T2,
    #    W[v, o, v, v],
    #    optimize=["einsum_path", (0, 1), (0, 2), (0, 1)],
    # )

    temp_lm = 0.25 * np.tensordot(l_2, t_2, axes=((1, 2, 3), (3, 0, 1)))
    temp_amil = np.tensordot(u[v, o, v, v], t_2, axes=((2, 3), (0, 1)))
    res += np.tensordot(temp_amil, temp_lm, axes=((1, 3), (1, 0)))
    # np.testing.assert_allclose(result, res)

    # result += 0.125 * np.einsum(
    #    "lkcd,lkAe,mncd,mnIe->AI",
    #    L2,
    #    T2,
    #    T2,
    #    W[o, o, o, v],
    #    optimize=["einsum_path", (0, 2), (1, 2), (0, 1)],
    # )

    temp_lkmn = 0.125 * np.tensordot(l_2, t_2, axes=((2, 3), (0, 1)))
    temp_aemn = np.tensordot(t_2, temp_lkmn, axes=((2, 3), (0, 1)))
    res += np.tensordot(temp_aemn, u[o, o, o, v], axes=((1, 2, 3), (3, 0, 1)))
    # np.testing.assert_allclose(result, res, atol=1e-10)

    # return result
    return res
