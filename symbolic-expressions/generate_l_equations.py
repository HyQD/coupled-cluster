from sympy import diff, latex, symbols
from helper_functions import eval_equation, beautify_equation
from generate_cc_lagrangian import generate_lagrangian
from cluster_operators import get_t_2_operator, get_l_2_operator

from sympy.physics.secondquant import (
    evaluate_deltas,
    PermutationOperator,
    simplify_index_permutations,
)


def generate_ccd_l_equations():
    ccd_lagrangian = eval_equation(
        generate_lagrangian(get_t_2_operator, get_l_2_operator)
    )

    i, j = symbols("i, j", below_fermi=True)
    a, b = symbols("a, b", above_fermi=True)
    # t_equations = eval_equation(diff(ccd_lagrangian, "l"))
    # l_equations = eval_equation(diff(ccd_lagrangian, "t"))

    t_equations = beautify_equation(diff(ccd_lagrangian, "l"))
    # t_equations = evaluate_deltas(diff(ccd_lagrangian, "l").doit())
    # t_equations = substitute_dummies(t_equations, new_indices=True,
    #        pretty_indices=pretty_dummies_dict)
    t_equations = simplify_index_permutations(
        t_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )

    l_equations = beautify_equation(diff(ccd_lagrangian, "t"))
    l_equations = simplify_index_permutations(
        l_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )

    print("t-equations: ", latex(t_equations))
    print("l-equations: ", latex(l_equations))


if __name__ == "__main__":
    generate_ccd_l_equations()
