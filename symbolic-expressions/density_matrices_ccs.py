from density_matrices import (
    get_ccs_one_body_density_matrix,
    symbol_list_one_body,
    get_ccs_two_body_density_matrix,
    symbol_list_two_body,
)
from sympy import latex

print("One-body density matrix for CCS:")
for label, (p, q) in symbol_list_one_body:
    print(label + latex(get_ccs_one_body_density_matrix(p=p, q=q)))


print("\nTwo-body density matrix for CCS:")
for label, (p, q, r, s) in symbol_list_two_body:
    print(label + latex(get_ccs_two_body_density_matrix(p=p, q=q, r=r, s=s)))
