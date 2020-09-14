from sympy import diff, latex, symbols
from helper_functions import eval_equation, beautify_equation
from generate_cc_lagrangian import generate_lagrangian
from cluster_operators import (
    get_t_1_operator,
    get_t_2_operator,
    get_t_3_operator,
    get_l_1_operator,
    get_l_2_operator,
    get_l_3_operator,
)

from sympy.physics.secondquant import (
    evaluate_deltas,
    PermutationOperator,
    simplify_index_permutations,
)


def generate_ccs_amplitude_equations(verbose=False):
    if verbose:
        print("Generating Lagrangian")
    ccs_lagrangian = eval_equation(
        generate_lagrangian(
            [get_t_1_operator],
            [get_l_1_operator],
        )
    )

    i = symbols("I", below_fermi=True)
    a = symbols("A", above_fermi=True)

    if verbose:
        print("Computing t_1 equations")
    t_1_equations = beautify_equation(diff(ccs_lagrangian, "l_1"))

    if verbose:
        print("Computing l_1 equations")
    l_1_equations = beautify_equation(diff(ccs_lagrangian, "t_1"))

    return [t_1_equations], [l_1_equations]


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


def generate_ccsd_amplitude_equations(verbose=False):
    if verbose:
        print("Generating Lagrangian")
    ccsd_lagrangian = eval_equation(
        generate_lagrangian(
            [get_t_1_operator, get_t_2_operator],
            [get_l_1_operator, get_l_2_operator],
        )
    )

    i, j = symbols("I, J", below_fermi=True)
    a, b = symbols("A, B", above_fermi=True)

    if verbose:
        print("Computing t_1 equations")
    t_1_equations = beautify_equation(diff(ccsd_lagrangian, "l_1"))

    if verbose:
        print("Computing t_2 equations")
    t_2_equations = beautify_equation(diff(ccsd_lagrangian, "l_2"))
    t_2_equations = simplify_index_permutations(
        t_2_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )

    if verbose:
        print("Computing l_1 equations")
    l_1_equations = beautify_equation(diff(ccsd_lagrangian, "t_1"))

    if verbose:
        print("Computing l_2 equations\n\n")
    l_2_equations = beautify_equation(diff(ccsd_lagrangian, "t_2"))
    l_2_equations = simplify_index_permutations(
        l_2_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )

    return [t_1_equations, t_2_equations], [l_1_equations, l_2_equations]


def generate_ccsdt_amplitude_equations(verbose=False):

    if verbose:
        print("Generating Lagrangian")
    ccsdt_lagrangian = eval_equation(
        generate_lagrangian(
            [get_t_1_operator, get_t_2_operator, get_t_3_operator],
            [get_l_1_operator, get_l_2_operator, get_l_3_operator],
        )
    )

    i, j, k = symbols("I, J, K", below_fermi=True)
    a, b, c = symbols("A, B, C", above_fermi=True)

    if verbose:
        print("Computing t_1 equations")
    t_1_equations = beautify_equation(diff(ccsdt_lagrangian, "l_1"))

    if verbose:
        print("Computing t_2 equations")
    t_2_equations = beautify_equation(diff(ccsdt_lagrangian, "l_2"))
    t_2_equations = simplify_index_permutations(
        t_2_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )
    if verbose:
        print("Computing t_3 equations")
    t_3_equations = beautify_equation(diff(ccsdt_lagrangian, "l_3"))
    t_3_equations = simplify_index_permutations(
        t_3_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )

    if verbose:
        print("Computing l_1 equations")
    l_1_equations = beautify_equation(diff(ccsdt_lagrangian, "t_1"))

    if verbose:
        print("Computing l_2 equations")
    l_2_equations = beautify_equation(diff(ccsdt_lagrangian, "t_2"))
    l_2_equations = simplify_index_permutations(
        l_2_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )
    if verbose:
        print("Computing l_3 equations\n\n")
    l_3_equations = beautify_equation(diff(ccsdt_lagrangian, "t_3"))
    l_3_equations = simplify_index_permutations(
        l_3_equations, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )

    return (
        [t_1_equations, t_2_equations, t_3_equations],
        [l_1_equations, l_2_equations, l_3_equations],
    )


if __name__ == "__main__":
    [t_1], [l_1] = generate_ccs_amplitude_equations()

    print("t_1-equations:")
    for term in t_1.args:
        print(latex(term))

    print("\n\nl_1-equations:")
    for term in l_1.args:
        print(latex(term))

    # (t_1, t_2, t_3), (l_1, l_2, l_3) = generate_ccsdt_amplitude_equations()

    # f = open("amplitude_eqns.txt", "w+")

    # print("t_1-equations:")
    # f.write("t_1 equations\n")
    # for term in t_1.args:
    #    print(latex(term))
    #    f.write(latex(term) + "\n")

    # print("\n\nt_2-equations:")
    # f.write("\n\nt_2 equations\n")
    # for term in t_2.args:
    #    print(latex(term))
    #    f.write(latex(term) + "\n")

    # print("\n\nt_3-equations:")
    # f.write("\n\nt_3 equations\n")
    # for term in t_3.args:
    #    print(latex(term))
    #    f.write(latex(term) + "\n")

    # print("\n\nl_1-equations:")
    # f.write("\n\nl_1 equations\n")
    # for term in l_1.args:
    #    print(latex(term))
    #    f.write(latex(term) + "\n")

    # print("\n\nl_2-equations:")
    # f.write("\n\nl_2 equations\n")
    # for term in l_2.args:
    #    print(latex(term))
    #    f.write(latex(term) + "\n")

    # print("\n\nl_3-equations:")
    # f.write("\n\nl_3 equations\n")
    # for term in l_3.args:
    #    print(latex(term))
    #    f.write(latex(term) + "\n")

    # f.close()
