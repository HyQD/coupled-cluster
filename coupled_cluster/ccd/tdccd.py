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
from coupled_cluster.ccd.overlap import compute_overlap
from coupled_cluster.ccd import CCD
from coupled_cluster.cc_helper import AmplitudeContainer


class TDCCD(TimeDependentCoupledCluster):
    """Time Dependent Coupled Cluster Doubles

    Computes time development of system, employed coupled
    cluster method with double exctiations.

    Parameters
    ----------
    system : QuantumSystem
        Class instance defining the system to be solved
    np : module
        Matrix/linear algebra library to be uses, like numpy or cupy
    integrator : Integrator
        Integrator class instance (RK4, GaussIntegrator)
    """

    truncation = "CCD"

    def rhs_t_0_amplitude(self, *args, **kwargs):
        return self.np.array([compute_ccd_ground_state_energy(*args, **kwargs)])

    def rhs_t_amplitudes(self):
        yield compute_t_2_amplitudes

    def rhs_l_amplitudes(self):
        yield compute_l_2_amplitudes

    def compute_left_reference_overlap(self, current_time, y):
        t_0, t_2, l_2 = self._amp_template.from_array(y).unpack()

        return 1 - 0.25 * self.np.tensordot(
            l_2, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))
        )

    def compute_energy(self, current_time, y):
        """Computes energy of a given amplitudes array.

        Returns
        -------
        float
            Energy
        """

        t_0, t_2, l_2 = self._amp_template.from_array(y).unpack()
        self.update_hamiltonian(current_time, y)

        return compute_time_dependent_energy(
            self.f, self.u, t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_one_body_density_matrix(self, current_time, y):
        """Computes one-body density matrix

        Returns
        -------
        np.array
            One-body density matrix
        """
        t_0, t_2, l_2 = self._amp_template.from_array(y).unpack()

        return compute_one_body_density_matrix(
            t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_two_body_density_matrix(self, current_time, y):
        """Computes two-body density matrix at
        current time step.

        Returns
        -------
        np.array
            Two-body density matrix
        """

        t_0, t_2, l_2 = self._amp_template.from_array(y).unpack()

        return compute_two_body_density_matrix(
            t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_overlap(self, current_time, y_a, y_b):
        """Computes overlap of current two states a and b.

        Parameters
        ----------
        y_a : np array
            Array of amplitudes of state a

        y_a : np array
            Array of amplitudes of state b

        Returns
        -------
        np.complex128
            Probability of ground state
        """
        t_0_a, t_2_a, l_2_a = self._amp_template.from_array(y_a).unpack()
        t_0_b, t_2_b, l_2_b = self._amp_template.from_array(y_b).unpack()

        return compute_overlap(t_2_a, l_2_a, t_2_b, l_2_b, np=self.np)
