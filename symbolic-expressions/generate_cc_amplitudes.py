from sympy import diff, latex, symbols
from helper_functions import eval_equation, beautify_equation
from generate_cc_lagrangian import generate_lagrangian
from cluster_operators import (
    get_t_1_operator,
    get_t_2_operator,
    get_l_1_operator,
    get_l_2_operator,
)

from sympy.physics.secondquant import (
    evaluate_deltas,
    PermutationOperator,
    simplify_index_permutations,
)


def generate_ccd_amplitude_equations():
    ccd_lagrangian = eval_equation(
        generate_lagrangian([get_t_2_operator], [get_l_2_operator])
    )

    i, j = symbols("I, J", below_fermi=True)
    a, b = symbols("A, B", above_fermi=True)

    t_equations = beautify_equation(diff(ccd_lagrangian, "l_2"))
    t_equations = simplify_index_permutations(
        t_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )

    l_equations = beautify_equation(diff(ccd_lagrangian, "t_2"))
    l_equations = simplify_index_permutations(
        l_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )

    return t_equations, l_equations


def generate_ccsd_amplitude_equations():
    ccsd_lagrangian = eval_equation(
        generate_lagrangian(
            [get_t_1_operator, get_t_2_operator],
            [get_l_1_operator, get_l_2_operator],
        )
    )

    i, j = symbols("I, J", below_fermi=True)
    a, b = symbols("A, B", above_fermi=True)

    t_1_equations = beautify_equation(diff(ccsd_lagrangian, "l_1"))

    t_2_equations = beautify_equation(diff(ccsd_lagrangian, "l_2"))
    t_2_equations = simplify_index_permutations(
        t_2_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )

    l_1_equations = beautify_equation(diff(ccsd_lagrangian, "t_1"))

    l_2_equations = beautify_equation(diff(ccsd_lagrangian, "t_2"))
    l_2_equations = simplify_index_permutations(
        l_2_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )

    return [t_1_equations, t_2_equations], [l_1_equations, l_2_equations]


if __name__ == "__main__":
    # (t_1, t_2), (l_1, l_2) = generate_ccsd_amplitude_equations()

    # print("t_1-equations:")
    # for term in t_1.args:
    #     print(latex(term))

    # print("\n\nt_2-equations:")
    # for term in t_2.args:
    #     print(latex(term))

    # print("\n\nl_1-equations:")
    # for term in l_1.args:
    #     print(latex(term))

    # print("\n\nl_2-equations:")
    # for term in l_2.args:
    #     print(latex(term))

    t_2, l_2 = generate_ccd_amplitude_equations()

    print("t_2 equations")
    for term in t_2.args:
        print(latex(term))
    
    print("\n\nl_2 equations")
    for term in l_2.args:
        print(latex(term))
