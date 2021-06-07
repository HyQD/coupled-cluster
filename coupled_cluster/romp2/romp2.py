from scipy.linalg import expm

from coupled_cluster.rccd.rccd import RCCD
from coupled_cluster.cc_helper import (
    construct_d_t_1_matrix,
    construct_d_t_2_matrix,
    OACCVector,
)

from coupled_cluster.romp2.rhs_t import (
    compute_t_2_amplitudes,
    compute_l_2_amplitudes,
)
from coupled_cluster.mix import DIIS

from coupled_cluster.romp2.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)

from coupled_cluster.romp2.p_space_equations import compute_R_tilde_ai

from opt_einsum import contract


class ROMP2(RCCD):
    """Orbital-optimized second-order Møller-Plesset perturbation theory (OMP2)

    Parameters
    ----------
    system : QuantumSystem
        QuantumSystem class instance description of system

    References
    ----------
    .. [1] U. Bozkaya, J. M. Turney, Y. Yamaguchi, H. F. Schaefer, C. D. Sherrill
          "Quadratically convergent algorithm for orbital optimization in the orbital-optimized coupled-cluster doubles method
          and in orbital-optimized second-order Møller-Plesset perturbation theory", J. Chem. Phys. 135, 104103, 2011.

    """

    def __init__(self, system, **kwargs):
        if "mixer" not in kwargs:
            kwargs["mixer"] = DIIS

        super().__init__(system, **kwargs)

        np = self.np
        n, m, l = self.n, self.m, self.l

        self.kappa = np.zeros((l, l), dtype=self.t_2.dtype)

        self.kappa_up = np.zeros((m, n), dtype=self.t_2.dtype)

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

    def compute_t_amplitudes(self):
        np = self.np

        self.rhs_t_2.fill(0)
        self.rhs_t_2 = compute_t_2_amplitudes(
            self.f, self.u, self.t_2, self.o, self.v, out=self.rhs_t_2, np=np
        )

        trial_vector = self.t_2
        direction_vector = np.divide(self.rhs_t_2, self.d_t_2)
        error_vector = self.rhs_t_2.copy()

        self.t_2 = self.t_2_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

    def compute_l_amplitudes(self):
        np = self.np

        self.rhs_l_2.fill(0)
        compute_l_2_amplitudes(
            self.f,
            self.u,
            self.t_2,
            self.l_2,
            self.o,
            self.v,
            out=self.rhs_l_2,
            np=np,
        )

        trial_vector = self.l_2
        direction_vector = np.divide(self.rhs_l_2, self.d_l_2)
        error_vector = self.rhs_l_2.copy()

        self.l_2 = self.l_2_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

    def compute_one_body_density_matrix(self):
        return compute_one_body_density_matrix(
            self.t_2, self.l_2, self.o, self.v, np=self.np
        )

    def compute_two_body_density_matrix(self):
        return compute_two_body_density_matrix(
            self.t_2, self.l_2, self.o, self.v, np=self.np
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

        e_old = self.compute_energy() + self.system.nuclear_repulsion_energy

        for i in range(max_iterations):

            self.f = self.system.construct_fock_matrix(self.h, self.u)

            self.d_t_1 = construct_d_t_1_matrix(self.f, self.o, self.v, np)
            self.d_t_2 = construct_d_t_2_matrix(self.f, self.o, self.v, np)

            self.t_2 += (
                compute_t_2_amplitudes(
                    self.f, self.u, self.t_2, self.o, self.v, np
                )
                / self.d_t_2
            )

            self.l_2 += compute_l_2_amplitudes(
                self.f, self.u, self.t_2, self.l_2, self.o, self.v, np
            ) / self.d_t_2.transpose(2, 3, 0, 1)

            rho_qp = self.compute_one_body_density_matrix()
            rho_qspr = self.compute_two_body_density_matrix()

            ############################################################
            # This part of the code is common to most (if not all)
            # orbital-optimized methods.
            v, o = self.v, self.o
            w_ai = compute_R_tilde_ai(
                self.h, self.u, rho_qp, rho_qspr, o, v, np
            )
            residual_w_ai = np.linalg.norm(w_ai)

            self.kappa[self.v, self.o] -= 0.5 * w_ai / self.d_t_1

            C = expm(self.kappa - self.kappa.T)
            Ctilde = C.T

            self.h = self.system.transform_one_body_elements(
                self.system.h, C, Ctilde
            )

            self.u = self.system.transform_two_body_elements(
                self.system.u, C, Ctilde
            )
            ############################################################
            energy = self.compute_energy()

            if self.verbose:
                print(f"\nIteration: {i}")
                print(f"Residual norms: |w_ai| = {residual_w_ai}")
                print(f"Energy: {energy}")

            if np.abs(residual_w_ai) < tol:
                break

            e_old = energy

        self.C = C
        self.C_tilde = C.T.conj()

        if change_system_basis:
            if self.verbose:
                print("Changing system basis...")

            self.system.change_basis(C=self.C, C_tilde=self.C_tilde)
            self.C = np.eye(self.system.l)
            self.C_tilde = np.eye(self.system.l)

        if self.verbose:
            print(
                f"Final {self.__class__.__name__} energy: "
                + f"{self.compute_energy()+self.system.nuclear_repulsion_energy}"
            )
