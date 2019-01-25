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
