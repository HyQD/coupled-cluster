from sympy.physics.secondquant import Commutator, wicks
from sympy import factorial
from cluster_operators import get_hamiltonian, get_clusters


def generate_lagrangian(t_operators, l_operators):
    hamiltonian = get_hamiltonian()

    comm_term = sum(hamiltonian)
    equation = wicks(comm_term)
    i = 0

    while comm_term != 0:
        t = get_clusters(t_operators)

        comm_term = wicks(Commutator(comm_term, t))
        equation += comm_term / factorial(i + 1)
        i += 1

    equation = (1 + get_clusters(l_operators)) * equation

    return equation


if __name__ == "__main__":
    from cluster_operators import (
        get_t_1_operator,
        get_t_2_operator,
        get_l_1_operator,
        get_l_2_operator,
    )
    from helper_functions import eval_equation
    from sympy import latex

    lagrangian = eval_equation(
        generate_lagrangian([get_t_2_operator], [get_l_2_operator])
    )

    print("-" * 41 + "CCD" + "-" * 40)

    print(latex(lagrangian))
    print("\n\n")
    for term in lagrangian.args:
        print(latex(term))

    print("-" * 40 + "CCSD" + "-" * 40)

    lagrangian = eval_equation(
        generate_lagrangian(
            [get_t_1_operator, get_t_2_operator],
            [get_l_1_operator, get_l_2_operator],
        )
    )

    print(latex(lagrangian))
    print("\n\n")
    for term in lagrangian.args:
        print(latex(term))
