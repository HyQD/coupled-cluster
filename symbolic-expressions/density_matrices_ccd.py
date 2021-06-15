from density_matrices import (
    get_ccd_one_body_density_matrix,
    symbol_list_one_body,
    get_ccd_two_body_density_matrix,
    symbol_list_two_body,
)
from sympy import latex


# Note that the summation indices might not correspond to the labels in the
# symbol list. This can be remedied by carefully repeated indices with
# non-repeated indices until the labels properly match the expressions.
print("One-body density matrix for CCD:")
for label, (p, q) in symbol_list_one_body:
    print(label + latex(get_ccd_one_body_density_matrix(p=p, q=q)))

print("\nTwo-body density matrix for CCD:")
for label, (p, q, r, s) in symbol_list_two_body:
    print(label + latex(get_ccd_two_body_density_matrix(p=p, q=q, r=r, s=s)))
