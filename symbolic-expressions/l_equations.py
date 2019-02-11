from sympy.physics.secondquant import (
    Commutator,
    PermutationOperator,
    simplify_index_permutations,
    NO,
    Fd,
    F,
    wicks,
)
from hamiltonian import get_hamiltonian
from sympy import factorial, Rational, symbols
from cluster_operators import (
    get_clusters,
    get_t_2_operator,
    get_l_2_operator,
    get_hamiltonian,
)
from helper_functions import beautify_equation, eval_equation


def compute_hausdorff(comm_term, t_operators, l_operators):
    equation = wicks(comm_term)
    i = 0

    while comm_term != 0:
        t = get_clusters(t_operators)

        comm_term = wicks(Commutator(comm_term, t))
        equation += comm_term / factorial(i + 1)
        i += 1
        print(i)

    equation = (1 + get_clusters(l_operators)) * equation

    return eval_equation(equation)


def get_similarity_transformed_operator(t_func, num_terms=4):
    H = get_hamiltonian()
    # center = Commutator(H, get_clusters(t_func))
    center = H

    return compute_hausdorff(center, t_func, num_terms=num_terms)


def get_lagrange_multipliers(l_func):
    return 1 + get_clusters(l_func)


def get_doubles_operator():
    i, j = symbols("i, j", below_fermi=True)
    a, b = symbols("a, b", above_fermi=True)

    X_ijab = NO(Fd(i) * Fd(j) * F(b) * F(a))

    return X_ijab


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
    H = sum(get_hamiltonian())
    X_ijab = get_doubles_operator()

    comm_term = Commutator(H, X_ijab)

    return eval_equation(compute_hausdorff(comm_term, t_func, l_func))


if __name__ == "__main__":
    print(get_ccd_l_equations())
