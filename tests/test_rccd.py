import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.mix import DIIS, AlphaMixer
from coupled_cluster.rccd import RCCD
from coupled_cluster.ccd import CCD

def test_rccd_vs_ccd():

    molecule = "li 0.0 0.0 0.0;h 0.0 0.0 3.08"
    basis = "cc-pvdz"

    system = construct_pyscf_system_rhf(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )

    rccd = RCCD(system, mixer=DIIS, verbose=False)

    conv_tol = 1e-10
    t_kwargs = dict(tol=conv_tol)
    l_kwargs = dict(tol=conv_tol)

    rccd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)
    e_rccd = rccd.compute_energy()+system.nuclear_repulsion_energy
    dm1 = rccd.compute_one_body_density_matrix()
    dip_mom_rccd = np.zeros(3)
    for i in range(3):
        dip_mom_rccd[i] = np.trace(np.dot(dm1, system.dipole_moment[i]))
    
    system = construct_pyscf_system_rhf(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=True,
        anti_symmetrize=True,
    )
    ccd = CCD(system, mixer=DIIS, verbose=False)
    ccd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)
    e_ccd = ccd.compute_energy()+system.nuclear_repulsion_energy
    dm1 = ccd.compute_one_body_density_matrix()
    dip_mom_ccd = np.zeros(3)
    for i in range(3):
        dip_mom_ccd[i] = np.trace(np.dot(dm1, system.dipole_moment[i]))
    
    assert abs(e_ccd-e_rccd) < 1e-10
    assert abs(dip_mom_ccd[2]-dip_mom_rccd[2]) < 1e-10

if __name__ == "__main__":
    test_rccd_vs_ccd()