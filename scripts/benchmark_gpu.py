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

He = """
He 0.0 0.0 0.0
symmetry c1
"""

Ne = """
Ne 0.0 0.0 0.0
symmetry c1
"""

Ar = """
Ar 0.0 0.0 0.0
symmetry c1
"""

options = {"basis": "cc-pvtz", "scf_type": "pk", "e_convergence": 1e-8}


# for CO er benchmark verdiene med hensyn på STO-3G basis
E_RHF_Dalton = {"He": -2.855160477243, "CO": -111.143611707173}
E_CCSD_Dalton = {"He": -0.0324343538481968, "CO": -0.1083477443172285}
Dipmom_Dalton = {
    "He": np.array([0.0, 0.0, 0.0]),
    "CO": np.array([0.0, 0.0, 0.31848768]),
}

CO = """
C 0.0 0.0 -1.079696382067556
O 0.0 0.0  0.810029743390272
symmetry c1
no_reorient
no_com
units bohr
"""
