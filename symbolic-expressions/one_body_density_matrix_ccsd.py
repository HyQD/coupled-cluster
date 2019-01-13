from one_body_density_matrix import (
    get_ccsd_one_body_density_matrix,
    symbol_list,
)
from sympy import latex

for label, (p, q) in symbol_list:
    print(label + latex(get_ccsd_one_body_density_matrix(p=p, q=q)))
