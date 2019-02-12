from sympy import diff, latex, symbols
from helper_functions import eval_equation, beautify_equation
from generate_cc_lagrangian import generate_lagrangian
from cluster_operators import get_t_2_operator, get_l_2_operator

from sympy.physics.secondquant import (
    evaluate_deltas,
    PermutationOperator,
    simplify_index_permutations,
)


def generate_ccd_amplitude_equations():
    ccd_lagrangian = eval_equation(
        generate_lagrangian(get_t_2_operator, get_l_2_operator)
    )

    i, j = symbols("I, J", below_fermi=True)
    a, b = symbols("A, B", above_fermi=True)

    t_equations = beautify_equation(diff(ccd_lagrangian, "l"))
    t_equations = simplify_index_permutations(
        t_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )

    l_equations = beautify_equation(diff(ccd_lagrangian, "t"))
    l_equations = simplify_index_permutations(
        l_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )

    return t_equations, l_equations


if __name__ == "__main__":
    t, l = generate_ccd_amplitude_equations()

    print("t-equations:")
    for term in t.args:
        print(latex(term))

    print("\n\nl-equations:")
    for term in l.args:
        print(latex(term))
