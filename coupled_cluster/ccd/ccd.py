from coupled_cluster.cc import CoupledCluster
from coupled_cluster.ccd.energies import (
    compute_ccd_ground_state_energy_correction,
)
from coupled_cluster.ccd.rhs_t import compute_rhs_t_2_amplitudes
from coupled_cluster.ccd.rhs_l import compute_rhs_l_2_amplitudes
from coupled_cluster.ccd.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)
from coupled_cluster.cc_helper import construct_d_t_2_matrix


class CCD(CoupledCluster):
    r"""Coupled-cluster with doubles excitations.

    Parameters
    ----------
    system : QuantumSystem
        QuantumSystem class describing the system
    """

    rhs_t_amplitudes = [compute_rhs_t_2_amplitudes]
    rhs_l_amplitudes = [compute_rhs_l_2_amplitudes]
    construct_one_body_density_matrix = staticmethod(
        compute_one_body_density_matrix
    )
    construct_two_body_density_matrix = staticmethod(
        compute_two_body_density_matrix
    )
