from coupled_cluster.cc import CoupledCluster

from coupled_cluster.rccsd.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)

from coupled_cluster.rccsd.rhs_l import (
    compute_l_1_amplitudes,
    compute_l_2_amplitudes,
)

from coupled_cluster.cc_helper import (
    construct_d_t_1_matrix,
    construct_d_t_2_matrix,
)

from coupled_cluster.rccsd.density_matrices import (
    compute_one_body_density_matrix,
)

from opt_einsum import contract


class RCCSD(CoupledCluster):
    r"""Restricted Coupled Cluster Singels Doubles

    Restricted coupled-cluster solver with single-, and double
    excitations. The excitation and de-exciation operators are parametrized according to chapter 13.7.5 in [1]_,

    .. math:: \hat{T}_1 &= \sum_{ai} \tau^a_i E_{ai} \\
              \hat{T}_2 &= \frac{1}{2}\sum_{abij} \tau^{ab}_{ij} E_{ai} E_{bj} \\
              \hat{\Lambda}_1 &= \frac{1}{2} \sum_{ai} \lambda^i_a E_{ia} \\
              \hat{\Lambda}_2 &= \frac{1}{2} \sum_{abij} \lambda^{ij}_{ab} \left(\frac{1}{3} E_{ia} E_{jb} + \frac{1}{6} E_{ja} E_{ib} \right)
    
    Parameters
    ----------
    system : QuantumSystems
        QuantumSystems class instance describing the system to be solved
    include_singles : bool
        Include singles

    Attributes
    ----------
    t_1, t_2 : np.ndarray
        :math:`\hat{T}`-amplitudes :math:`\tau^a_i, \tau^{ab}_{ij}`
    l_1, l_2 : np.ndarray
        :math:`\hat{\Lambda}`-amplitudes :math:`\lambda^i_a, \lambda^{ij}_{ab}` 

    References
    ----------
    .. [1] T. Helgaker, P. Jorgensen, J. Olsen "Molecular electronic-structure theory",
           John Wiley & Sons, 2014.

    """

    def __init__(self, system, include_singles=True, **kwargs):
        super().__init__(system, **kwargs)

        np = self.np

        """
        Manually handle restricted properties for now before quantum systems have been 
        updated
        """
        # self.n, self.m, self.l = system.n // 2, system.m // 2, system.l // 2
        # self.o, self.v = slice(0, self.n), slice(self.n, self.l)

        n, m = self.n, self.m

        self.include_singles = include_singles

        # Singles
        self.rhs_t_1 = np.zeros((m, n), dtype=self.u.dtype)  # ai
        self.rhs_l_1 = np.zeros((n, m), dtype=self.u.dtype)  # ia

        self.t_1 = np.zeros_like(self.rhs_t_1)
        self.l_1 = np.zeros_like(self.rhs_l_1)

        self.d_t_1 = construct_d_t_1_matrix(self.f, self.o, self.v, np)
        self.d_l_1 = self.d_t_1.transpose(1, 0).copy()

        # Doubles
        self.rhs_t_2 = np.zeros((m, m, n, n), dtype=self.u.dtype)  # abij
        self.rhs_l_2 = np.zeros((n, n, m, m), dtype=self.u.dtype)  # ijab

        self.t_2 = np.zeros_like(self.rhs_t_2)
        self.l_2 = np.zeros_like(self.rhs_l_2)

        self.d_t_2 = construct_d_t_2_matrix(self.f, self.o, self.v, np)
        self.d_l_2 = self.d_t_2.transpose(2, 3, 0, 1).copy()

        # Mixer
        self.t_mixer = None
        self.l_mixer = None

        # Go!
        self.compute_initial_guess()

    def compute_initial_guess(self):

        r"""compute_initial_guess

        Compute the initial guess for the coupled-cluster amplitudes. 
        Currently the only option is the MP2 initial guess,
        
        .. math:: \tau^a_i &= \lambda^i_a = 0, \, \forall a,i \\
                  \tau^{ab}_{ij} &= \frac{u^{ab}_{ij}}{\epsilon_i+\epsilon_j - \epsilon_a - \epsilon_b} \\ 
                  \lambda^{ij}_{ab} &= \frac{u^{ij}_{ab}}{\epsilon_i+\epsilon_j - \epsilon_a - \epsilon_b}.

        """

        np = self.np
        o, v = self.o, self.v

        # Doubles
        np.copyto(self.rhs_t_2, self.u[v, v, o, o])
        np.divide(self.rhs_t_2, self.d_t_2, out=self.t_2)

        np.copyto(self.rhs_l_2, self.u[o, o, v, v])
        np.divide(self.rhs_l_2, self.d_l_2, out=self.l_2)

    def _get_t_copy(self):
        return [self.t_1.copy(), self.t_2.copy()]

    def _get_l_copy(self):
        return [self.l_1.copy(), self.l_2.copy()]

    def compute_l_residuals(self):
        return [
            self.np.linalg.norm(self.rhs_l_1),
            self.np.linalg.norm(self.rhs_l_2),
        ]

    def compute_t_residuals(self):
        return [
            self.np.linalg.norm(self.rhs_t_1),
            self.np.linalg.norm(self.rhs_t_2),
        ]

    def setup_l_mixer(self, **kwargs):
        if self.l_mixer is None:
            self.l_mixer = self.mixer(**kwargs)

        self.l_mixer.clear_vectors()

    def setup_t_mixer(self, **kwargs):
        if self.t_mixer is None:
            self.t_mixer = self.mixer(**kwargs)

        self.t_mixer.clear_vectors()

    def compute_energy(self):
        r"""compute_energy

        Compute the total restricted coupled-cluster energy

        .. math:: E_{\text{RCCSD}} = E_{\text{ref}} + 2 f^i_a \tau^a_i + (2 \tau^{ab}_{ij} - \tau^{a}_i \tau^b_j) (u^{ij}_{ab} - u^{ij}_{ba} )

        Returns
        -------
        float
            The total coupled-cluster energy of the current state.
        """
        np = self.np
        o, v = self.o, self.v

        e_corr = 2 * contract("ia,ai->", self.f[o, v], self.t_1)

        e_corr += 2 * contract("abij,ijab->", self.t_2, self.u[o, o, v, v])
        e_corr -= contract("abij,ijba->", self.t_2, self.u[o, o, v, v])

        e_corr += 2 * contract(
            "ai,bj,ijab->",
            self.t_1,
            self.t_1,
            self.u[o, o, v, v],
            optimize=True,
        )
        e_corr -= contract(
            "ai,bj,ijba->",
            self.t_1,
            self.t_1,
            self.u[o, o, v, v],
            optimize=True,
        )

        e_ref = self.system.compute_reference_energy()

        return e_corr + e_ref

    def compute_t_amplitudes(self):
        np = self.np

        trial_vector = np.array([], dtype=self.u.dtype)
        direction_vector = np.array([], dtype=self.u.dtype)
        error_vector = np.array([], dtype=self.u.dtype)

        # Singles
        if self.include_singles:
            self.rhs_t_1.fill(0)
            self.rhs_t_1 = compute_t_1_amplitudes(
                self.f,
                self.u,
                self.t_1,
                self.t_2,
                self.o,
                self.v,
                out=self.rhs_t_1,
                np=np,
            )
            trial_vector = self.t_1.ravel()
            direction_vector = (self.rhs_t_1 / self.d_t_1).ravel()
            error_vector = self.rhs_t_1.ravel().copy()

        # Doubles
        self.rhs_t_2.fill(0)
        self.rhs_t_2 = compute_t_2_amplitudes(
            self.f,
            self.u,
            self.t_1,
            self.t_2,
            self.o,
            self.v,
            out=self.rhs_t_2,
            np=np,
        )

        trial_vector = np.concatenate((trial_vector, self.t_2.ravel()), axis=0)
        direction_vector = np.concatenate(
            (direction_vector, (self.rhs_t_2 / self.d_t_2).ravel()), axis=0
        )
        error_vector = np.concatenate(
            (error_vector, self.rhs_t_2.ravel().copy()), axis=0
        )

        new_vectors = self.t_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

        n_t1 = 0

        if self.include_singles:
            n_t1 = self.m * self.n
            self.t_1 = np.reshape(new_vectors[:n_t1], self.t_1.shape)

        self.t_2 = np.reshape(new_vectors[n_t1:], self.t_2.shape)

    def compute_l_amplitudes(self):
        np = self.np

        trial_vector = np.array([], dtype=self.u.dtype)  # Empty array
        direction_vector = np.array([], dtype=self.u.dtype)  # Empty array
        error_vector = np.array([], dtype=self.u.dtype)  # Empty array

        """
        The transposes of l1 and l2 are just a temporary solution until 
        we switch it in the generated right hand sides.
        """
        # Singles
        if self.include_singles:
            self.rhs_l_1.fill(0)
            self.rhs_l_1 = compute_l_1_amplitudes(
                self.f,
                self.u,
                self.t_1,
                self.t_2,
                self.l_1,
                self.l_2,
                self.o,
                self.v,
                out=self.rhs_l_1,
                np=np,
            )

            trial_vector = self.l_1.ravel()
            direction_vector = (self.rhs_l_1 / self.d_l_1).ravel()
            error_vector = self.rhs_l_1.ravel().copy()

        # Doubles
        self.rhs_l_2.fill(0)
        self.rhs_l_2 = compute_l_2_amplitudes(
            self.f,
            self.u,
            self.t_1,
            self.t_2,
            self.l_1,
            self.l_2,
            self.o,
            self.v,
            out=self.rhs_l_2,
            np=np,
        )

        trial_vector = np.concatenate((trial_vector, self.l_2.ravel()), axis=0)
        direction_vector = np.concatenate(
            (direction_vector, (self.rhs_l_2 / self.d_l_2).ravel()), axis=0
        )
        error_vector = np.concatenate(
            (error_vector, self.rhs_l_2.ravel().copy()), axis=0
        )

        new_vectors = self.l_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

        n_l1 = 0

        if self.include_singles:
            n_l1 = self.m * self.n
            self.l_1 = np.reshape(new_vectors[:n_l1], self.l_1.shape)

        self.l_2 = np.reshape(new_vectors[n_l1:], self.l_2.shape)

    def compute_one_body_density_matrix(self):
        r"""compute_one_body_density_matrix

        Computes the coupled-cluster one-body density matrix [2]_,

        .. math:: \rho^q_p \equiv \langle \tilde{\Psi} | a_p^\dagger a_q | \Psi \rangle.

        Returns
        -------
        np.array
            One-body density matrix

        References
        ----------
        .. [2] I. Shavitt, R. Bartlett "Many-body methods in chemistry and physics: MBPT and coupled-cluster theory",
           Cambridge university press, 2009.

        """

        return compute_one_body_density_matrix(
            self.t_1, self.t_2, self.l_1, self.l_2, self.o, self.v, np=self.np
        )

    def compute_two_body_density_matrix(self):
        pass
