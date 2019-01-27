from quantum_systems import construct_psi4_system
from coupled_cluster.ccd import CoupledClusterDoubles
from tdhf import HartreeFock
import numpy as np


def ccd_groundstate(name, system):
    print("** Compute groundstate of %s **" % name)
    print("Number of electrons: {0}".format(system.n))
    print("Number of spin orbitals: {0}".format(system.l))
    # Compute the Hartree-Fock state
    hf = HartreeFock(system, verbose=True)
    C = hf.scf(max_iters=100, tolerance=1e-8)

    system._h = np.einsum("ap,bq,ab->pq", C.conj(), C, system.h, optimize=True)
    system._u = np.einsum(
        "ap,bq,gr,ds,abgd->pqrs",
        C.conj(),
        C.conj(),
        C,
        C,
        system.u,
        optimize=True,
    )

    ccd = CoupledClusterDoubles(system, verbose=False)
    ccd.iterate_t_amplitudes(theta=0)
    ccd.iterate_l_amplitudes(theta=0)

    print("Eccd: {0}".format(ccd.compute_energy().real))
    print("Ecorr: {0}".format((ccd.compute_energy() - hf.e_hf).real))

    rho_pq = ccd.compute_one_body_density_matrix()
    print("tr(rho_pq): {0}".format(np.trace(rho_pq).real))

    rho_pqrs = ccd.compute_two_body_density_matrix()
    print("tr(rho_pqrs) {0}".format(np.einsum("pqpq->", rho_pqrs).real))

    print()


# Noble gases + Be
He = """
He 0.0 0.0 0.0
symmetry c1
"""
options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(He, options)
ccd_groundstate("He", system)

Be = """
Be 0.0 0.0 0.0
symmetry c1
"""
options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(Be, options)
ccd_groundstate("Be", system)

Ne = """
Ne 0.0 0.0 0.0
symmetry c1
"""
options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(Ne, options)
ccd_groundstate("Ne", system)
