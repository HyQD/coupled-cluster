from coupled_cluster.tdcc import TimeDependentCoupledCluster
from coupled_cluster.ccd.rhs_t import compute_t_2_amplitudes
from coupled_cluster.ccd.rhs_l import compute_l_2_amplitudes
from coupled_cluster.ccd.energies import (
    compute_time_dependent_energy,
    compute_ccd_ground_state_energy,
)
from coupled_cluster.ccd.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)
from coupled_cluster.ccd.time_dependent_overlap import (
    compute_time_dependent_overlap,
)
from coupled_cluster.ccd import CoupledClusterDoubles


class TDCCD(TimeDependentCoupledCluster):
    """Time Dependent Coupled Cluster Doubles

    Computes time development of system, employd coupled
    cluster method with double exctiations.

    Parameters
    ----------
    cc : CoupledCluster
        Class instance defining the ground state solver
    system : QuantumSystem
        Class instance defining the system to be solved
    np : module
        Matrix/linear algebra library to be uses, like numpy or cupy
    integrator : Integrator
        Integrator class instance (RK4, GaussIntegrator)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(CoupledClusterDoubles, *args, **kwargs)

    def rhs_t_0_amplitude(self, *args, **kwargs):
        return self.np.array([compute_ccd_ground_state_energy(*args, **kwargs)])

    def rhs_t_amplitudes(self):
        yield compute_t_2_amplitudes

    def rhs_l_amplitudes(self):
        yield compute_l_2_amplitudes

    def compute_energy(self):
        """Computes energy at current time step.
        
        Returns
        -------
        float
            Energy 
        """
        t_0, t_2, l_2 = self._amplitudes.unpack()

        return compute_time_dependent_energy(
            self.f, self.u, t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_one_body_density_matrix(self):
        """Computes one-body density matrix at
        current time step.

        Returns
        -------
        np.array
            One-body density matrix
        """
        t_0, t_2, l_2 = self._amplitudes.unpack()

        return compute_one_body_density_matrix(
            t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_two_body_density_matrix(self):
        """Computes two-body density matrix at
        current time step.

        Returns
        -------
        np.array
            Two-body density matrix
        """

        t_0, t_2, l_2 = self._amplitudes.unpack()

        return compute_two_body_density_matrix(
            t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_time_dependent_overlap(self):
        """Computes overlap of current time-developed
        state with the ground state. 

        Returns
        -------
        np.complex128
            Probability of ground state
        """
        t_0, t_2, l_2 = self._amplitudes.unpack()

        return compute_time_dependent_overlap(
            self.cc.t_2, self.cc.l_2, t_2, l_2, np=self.np
        )
