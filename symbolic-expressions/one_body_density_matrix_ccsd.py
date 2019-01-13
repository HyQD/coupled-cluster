from one_body_density_matrix import (
    get_ccsd_one_body_density_matrix,
    symbol_list,
)
from sympy import latex


# Note that the summation indices might not correspond to the labels in the
# symbol list. This can be remedied by carefully repeated indices with
# non-repeated indices until the labels properly match the expressions.
for label, (p, q) in symbol_list:
    print(label + latex(get_ccsd_one_body_density_matrix(p=p, q=q)))
