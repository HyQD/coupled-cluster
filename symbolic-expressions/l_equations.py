from sympy.physics.secondquant import (
    Commutator,
    PermutationOperator,
    simplify_index_permutations,
    NO,
    Fd,
    F,
)
from hamiltonian import get_hamiltonian
from sympy import factorial, Rational, symbols
from cluster_operators import get_clusters, get_t_2_operator, get_l_2_operator
from helper_functions import beautify_equation, eval_equation


def compute_hausdorff(comm_term, t_func, num_terms=4):
    equation = comm_term

    for i in range(num_terms):
        T = get_clusters(t_func)

        comm_term = eval_equation(Commutator(comm_term, T), wicks_kwargs={})
        # comm_term = wicks(Commutator(comm_term, T))
        # comm_term = substitute_dummies(evaluate_deltas(comm_term))

        equation += Rational(1, factorial(i + 1)) * comm_term

    return eval_equation(equation)


def get_similarity_transformed_operator(t_func, num_terms=4):
    H = get_hamiltonian()
    # center = Commutator(H, get_clusters(t_func))
    center = H

    return compute_hausdorff(center, t_func, num_terms=num_terms)


def get_lagrange_multipliers(l_func):
    return 1 + get_clusters(l_func)


def get_doubles_amplitudes(eq):
    i, j = symbols("i, j", below_fermi=True)
    a, b = symbols("a, b", above_fermi=True)

    eq = NO(Fd(i) * Fd(j) * F(b) * F(a)) * eq
    eq = eval_equation(eq)
    eq = simplify_index_permutations(
        eq, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )

    return eq


def get_ccd_l_equations():
    t_func = get_t_2_operator
    l_func = get_l_2_operator

    eq = get_similarity_transformed_operator(t_func, num_terms=2)
    # eq = get_lagrange_multipliers(l_func) * eq

    eq = get_doubles_amplitudes(eq)

    return eval_equation(eq)


if __name__ == "__main__":
    print(get_ccd_l_equations())