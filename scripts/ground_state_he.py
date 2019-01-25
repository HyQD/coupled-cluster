from quantum_systems import construct_psi4_system
from coupled_cluster.ccd import CoupledClusterDoubles
from tdhf import HartreeFock
import numpy as np

def ccd_groundstate(name,system):
    print("** Compute groundstate of %s **" % name)
    print("Number of electrons: {0}".format(system.n))
    print("Number of spin orbitals: {0}".format(system.l))
    # Compute the Hartree-Fock state
    hf = HartreeFock(system,verbose=False)
    C = hf.scf(max_iters=30, tolerance=1e-8)
    
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
    ccd.iterate_t_amplitudes()
    print("Ehf: {0}".format(hf.e_hf.real))
    print("Eccd: {0}".format(ccd.compute_energy().real))
    print("Ecorr: {0}".format((ccd.compute_energy()-hf.e_hf).real))
    print()

He = """
He 0.0 0.0 0.0
symmetry c1
"""
options = {"basis": "cc-pvqz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(He, options)
ccd_groundstate("He",system)

Be = """
Be 0.0 0.0 0.0
symmetry c1
"""
options = {"basis": "cc-pvtz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(Be, options)
ccd_groundstate("Be",system)

Ne = """
Ne 0.0 0.0 0.0
symmetry c1
"""
options = {"basis": "cc-pvtz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(Ne, options)
#ccd_groundstate("Ne",system)

Ar = """
Ar 0.0 0.0 0.0
symmetry c1
"""
options = {"basis": "cc-pvtz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(Ar, options)
ccd_groundstate("Ar",system)

Kr = """
Kr 0.0 0.0 0.0
symmetry c1
"""
options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(Kr, options)
ccd_groundstate("Kr",system)

r = 1.1
h20 = (
    """
O
H 1 r
H 1 r 2 104
symmetry c1
r = %f
"""
    % r
)
options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(h20, options)
ccd_groundstate("H2O",system)

r = 2.0
N2 = """
0 1
N
N 1 %f
symmetry c1
units bohr
""" % (
    r
)
options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(N2, options)
ccd_groundstate("N2",system)

CO = """
C 0.0 0.0 -1.079696382067556
O 0.0 0.0  0.810029743390272
symmetry c1
no_reorient
no_com
units bohr
"""
options = {"basis": "STO-3G", "scf_type": "pk", "e_convergence": 1e-8}
system = construct_psi4_system(CO, options)
ccd_groundstate("CO",system)


