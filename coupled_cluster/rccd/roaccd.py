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

from coupled_cluster.rccd.p_space_equations import (
    compute_R_ia,
    compute_R_tilde_ai,
)

from opt_einsum import contract


class ROACCD(RCCD):
    """Restricted Orbital Adaptive Coupled Cluster Doubles

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

    def compute_energy(self):
        rho_qp = self.compute_one_body_density_matrix()
        rho_qspr = self.compute_two_body_density_matrix()

        return (
            contract("pq,qp->", self.h, rho_qp, optimize=True)
            + 0.5 * contract("pqrs,rspq->", self.u, rho_qspr, optimize=True)
            + self.system.nuclear_repulsion_energy
        )

    def compute_one_body_expectation_value(self, mat, make_hermitian=True):
        return super().compute_one_body_expectation_value(
            self.system.transform_one_body_elements(mat, self.C, self.C_tilde),
            make_hermitian=make_hermitian,
        )

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
            self.C = expm(self.kappa)
            self.C_tilde = expm(-self.kappa)

            self.h = self.system.transform_one_body_elements(
                self.system.h, self.C, self.C_tilde
            )
            self.u = self.system.transform_two_body_elements(
                self.system.u, self.C, self.C_tilde
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

            kappa_down_rhs = compute_R_ia(
                self.h, self.u, rho_qp, rho_qspr, self.o, self.v, np
            )

            kappa_up_rhs = compute_R_tilde_ai(
                self.h, self.u, rho_qp, rho_qspr, self.o, self.v, np
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

        self.C = expm(self.kappa)
        self.C_tilde = expm(-self.kappa)

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
