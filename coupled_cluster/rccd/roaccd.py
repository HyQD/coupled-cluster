from scipy.linalg import expm

from coupled_cluster.rccd.rccd import RCCD
from coupled_cluster.cc_helper import (
    construct_d_t_1_matrix,
    construct_d_t_2_matrix,
    OACCVector,
)

from coupled_cluster.rccd.rhs_t import compute_t_2_amplitudes
from coupled_cluster.rccd.rhs_l import compute_l_2_amplitudes
from coupled_cluster.mix import DIIS


class ROACCD(RCCD):
    """Orbital Adaptive Coupled Cluster Doubles

    Implementation of the non-orthogonal coupled cluster method with
    double excitations. The code is based on a script written by
    Rolf H. Myhre and Simen Kvaal.

    Requires orthonormal basis functions.

    https://doi.org/10.1063/1.5006160

    Parameters
    ----------
    system : QuantumSystem
        QuantumSystem class instance description of system
    """

    def __init__(self, system, **kwargs):
        if "mixer" not in kwargs:
            kwargs["mixer"] = DIIS

        super().__init__(system, **kwargs)

        np = self.np
        n, m, l = self.n, self.m, self.l

        self.kappa = np.zeros((l, l), dtype=self.t_2.dtype)

        self.kappa_up = np.zeros((m, n), dtype=self.t_2.dtype)
        self.kappa_down = np.zeros((n, m), dtype=self.t_2.dtype)

    def get_amplitudes(self, get_t_0=False):
        """Getter for amplitudes, overwrites CC.get_amplitudes to also include
        coefficients.

        Parameters
        ----------
        get_t_0 : bool
            Returns amplitude at t=0 if True

        Returns
        -------
        OACCVector
            Amplitudes and coefficients in OACCVector object
        """

        amps = super().get_amplitudes(get_t_0=get_t_0)
        return OACCVector(*amps, C=self.C, C_tilde=self.C_tilde, np=self.np)

    def setup_kappa_mixer(self, **kwargs):
        self.kappa_up_mixer = self.mixer(**kwargs)
        self.kappa_down_mixer = self.mixer(**kwargs)

    def compute_ground_state(
        self,
        max_iterations=100,
        tol=1e-4,
        termination_tol=1e-4,
        tol_factor=0.1,
        change_system_basis=True,
        **mixer_kwargs,
    ):
        """Compute ground state

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations
        tol : float
            Tolerance parameter, e.g. 1e-4
        tol_factor : float
            Tolerance factor
        change_system_basis : bool
            Whether or not to change the basis when the ground state is
            reached. Default is ``True``.
        """
        np = self.np

        if not "np" in mixer_kwargs:
            mixer_kwargs["np"] = np

        self.setup_kappa_mixer(**mixer_kwargs)

        amp_tol = 0.1

        for k_it in range(max_iterations):
            S = expm(self.kappa)
            S_inv = expm(-self.kappa)

            self.h = self.system.transform_one_body_elements(
                self.system.h, S, S_inv
            )
            self.u = self.system.transform_two_body_elements(
                self.system.u, S, S_inv
            )
            self.f = self.system.construct_fock_matrix(self.h, self.u)

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

            rho_qp = self.compute_one_body_density_matrix()
            rho_qspr = self.compute_two_body_density_matrix()

            kappa_up_rhs = np.einsum(
                "pi,ap->ai", self.h[:, self.o], rho_qp[self.v, :]
            ) - np.einsum("aq,qi->ai", self.h[self.v, :], rho_qp[:, self.o])
            kappa_up_rhs -= 0.5 * np.einsum(
                "aqrs,rsiq->ai",
                self.u[self.v, :, :, :],
                rho_qspr[:, :, self.o, :],
            )
            kappa_up_rhs -= 0.5 * np.einsum(
                "pars,rspi->ai",
                self.u[:, self.v, :, :],
                rho_qspr[:, :, :, self.o],
            )
            kappa_up_rhs += 0.5 * np.einsum(
                "pqri,rapq->ai",
                self.u[:, :, :, self.o],
                rho_qspr[:, self.v, :, :],
            )
            kappa_up_rhs += 0.5 * np.einsum(
                "pqis,aspq->ai",
                self.u[:, :, self.o, :],
                rho_qspr[self.v, :, :, :],
            )

            kappa_down_rhs = np.einsum(
                "pa,ip->ia", self.h[:, self.v], rho_qp[self.o, :]
            ) - np.einsum("iq,qa->ia", self.h[self.o, :], rho_qp[:, self.v])
            kappa_down_rhs -= 0.5 * np.einsum(
                "iqrs,rsaq->ia",
                self.u[self.o, :, :, :],
                rho_qspr[:, :, self.v, :],
            )
            kappa_down_rhs -= 0.5 * np.einsum(
                "pirs,rspa->ia",
                self.u[:, self.o, :, :],
                rho_qspr[:, :, :, self.v],
            )
            kappa_down_rhs += 0.5 * np.einsum(
                "pqra,ripq->ia",
                self.u[:, :, :, self.v],
                rho_qspr[:, self.o, :, :],
            )
            kappa_down_rhs += 0.5 * np.einsum(
                "pqas,ispq->ia",
                self.u[:, :, self.v, :],
                rho_qspr[self.o, :, :, :],
            )

            residual_up = np.linalg.norm(kappa_up_rhs)
            residual_down = np.linalg.norm(kappa_down_rhs)

            self.kappa_up = self.kappa_up_mixer.compute_new_vector(
                self.kappa_up, -0.5 * kappa_up_rhs / d_t_1, kappa_up_rhs
            )
            self.kappa_down = self.kappa_down_mixer.compute_new_vector(
                self.kappa_down, -0.5 * kappa_down_rhs / d_l_1, kappa_down_rhs
            )

            self.kappa[self.v, self.o] = self.kappa_up
            self.kappa[self.o, self.v] = self.kappa_down

            amp_tol = min(residual_up, residual_down) * tol_factor
            amp_tol = max(amp_tol, termination_tol)

            if np.abs(residual_up) < tol and np.abs(residual_down) < tol:
                break

            if self.verbose:
                print(f"\nIteration: {k_it}")
                print(f"\nResidual norms: rd = {residual_down}")
                print(f"Residual norms: ru = {residual_up}")
                print(f"Energy: {self.compute_energy()}")

        S = expm(self.kappa)
        S_inv = expm(-self.kappa)
        self.C = S
        self.C_tilde = S_inv

        self.h = self.system.transform_one_body_elements(
            self.system.h, self.C, self.C_tilde
        )
        self.u = self.system.transform_two_body_elements(
            self.system.u, self.C, self.C_tilde
        )
        self.f = self.system.construct_fock_matrix(self.h, self.u)

        if change_system_basis:
            if self.verbose:
                print("Changing system basis...")

            self.system.change_basis(C=self.C, C_tilde=self.C_tilde)
            self.C = np.eye(self.system.l)
            self.C_tilde = np.eye(self.system.l)

        if self.verbose:
            print(
                f"Final {self.__class__.__name__} energy: "
                + f"{self.compute_energy()}"
            )


def compute_kappa_up_rhs(f, u, t2, l2, o, v, np):
    pass


def compute_kappa_down_rhs(f, u, t2, l2, o, v, np):
    pass
