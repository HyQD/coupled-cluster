from sympy.physics.secondquant import (
    AntiSymmetricTensor,
    F,
    Fd,
    NO,
    Commutator,
    wicks,
    evaluate_deltas,
    substitute_dummies,
    PermutationOperator,
    simplify_index_permutations,
)
from sympy import symbols, Dummy, Rational, factorial, latex
from cluster_operators import (
    get_clusters,
    get_t_1_operator,
    get_t_2_operator,
    get_t_3_operator,
    get_l_1_operator,
    get_l_2_operator,
    get_l_3_operator,
)
from helper_functions import eval_equation

pretty_dummies_t = {"above": "defgh", "below": "lmn", "general": "tuvw"}
pretty_dummies_d = {"above": "cdefgh", "below": "klmn", "general": "tuvw"}

wicks_kwargs = {
    "simplify_dummies": True,
    "keep_only_fully_contracted": True,
    "simplify_kronecker_deltas": True,
}

sub_kwargs = {"new_indices": True}


def get_hamiltonian():
    p, q, r, s = symbols("p, q, r, s", cls=Dummy)
    f = AntiSymmetricTensor("f", (p,), (q,))
    u = AntiSymmetricTensor("u", (p, q), (r, s))

    f = f * NO(Fd(p) * F(q))
    u = u * NO(Fd(p) * Fd(q) * F(s) * F(r))

    return f, Rational(1, 4) * u


def compute_hausdorff(operator, t_func, num_terms=4, sub_kwargs=sub_kwargs):
    comm_term = operator
    equation = operator

    for i in range(num_terms):
        t = get_clusters(t_func)

        comm_term = wicks(Commutator(comm_term, t))
        comm_term = substitute_dummies(evaluate_deltas(comm_term))

        equation += comm_term / factorial(i + 1)

    equation = equation.expand()
    equation = evaluate_deltas(equation)
    equation = substitute_dummies(equation, **sub_kwargs)

    return equation


def get_energy_equation(equation):
    return wicks(equation, **wicks_kwargs)


def get_singles_amplitudes(equation, sub_kwargs=sub_kwargs):
    i = symbols("i", below_fermi=True)
    a = symbols("a", above_fermi=True)

    eq = wicks(NO(Fd(i) * F(a)) * equation, **wicks_kwargs)
    eq = substitute_dummies(eq, **sub_kwargs)

    return eq


def get_doubles_amplitudes(equation, sub_kwargs=sub_kwargs):
    i, j = symbols("i, j", below_fermi=True)
    a, b = symbols("a, b", above_fermi=True)

    eq = wicks(NO(Fd(i) * Fd(j) * F(b) * F(a)) * equation, **wicks_kwargs)
    eq = simplify_index_permutations(
        eq, [PermutationOperator(a, b), PermutationOperator(i, j)]
    )
    eq = substitute_dummies(eq, **sub_kwargs)

    return eq


def get_triples_amplitudes(equation, sub_kwargs=sub_kwargs):
    i, j, k = symbols("i, j, k", below_fermi=True)
    a, b, c = symbols("a, b, c", above_fermi=True)

    eq = wicks(
        NO(Fd(i) * Fd(j) * Fd(k) * F(c) * F(b) * F(a)) * equation,
        **wicks_kwargs
    )
    eq = simplify_index_permutations(
        eq,
        [
            PermutationOperator(a, b),
            PermutationOperator(a, c),
            PermutationOperator(b, c),
            PermutationOperator(i, j),
            PermutationOperator(i, k),
            PermutationOperator(j, k),
        ],
    )

    eq = substitute_dummies(eq, **sub_kwargs)

    return eq


def get_ccd_equations():
    sub_kwargs_ccd = sub_kwargs
    sub_kwargs_ccd["pretty_indices"] = pretty_dummies_d

    hamiltonian = sum(get_hamiltonian())
    doubles_cluster_function = get_t_2_operator
    i, j = symbols("i, j", below_fermi=True, cls=Dummy)
    a, b = symbols("a, b", above_fermi=True, cls=Dummy)
    c_abij = NO(Fd(a) * Fd(b) * F(j) * F(i))
    l_operator = Commutator(hamiltonian, c_abij)
    # l_operator = Commutator(hamiltonian,
    #        get_clusters(doubles_cluster_function))
    # l_amplitudes = 0

    t_equation = compute_hausdorff(
        hamiltonian, doubles_cluster_function, sub_kwargs=sub_kwargs_ccd
    )
    l_equation = compute_hausdorff(
        l_operator, doubles_cluster_function, sub_kwargs=sub_kwargs_ccd
    )
    l_equation = (1 + get_clusters(get_l_2_operator)) * l_equation
    l_equation = eval_equation(l_equation)

    energy = get_energy_equation(t_equation)
    t_amplitudes = get_doubles_amplitudes(t_equation, sub_kwargs=sub_kwargs_ccd)
    l_amplitudes = get_doubles_amplitudes(l_equation, sub_kwargs=sub_kwargs_ccd)

    return energy, [t_amplitudes], [l_amplitudes]


def get_ccsd_equations():
    sub_kwargs_ccsd = sub_kwargs
    sub_kwargs_ccsd["pretty_indices"] = pretty_dummies_d

    hamiltonian = sum(get_hamiltonian())

    cluster_function = [get_t_1_operator, get_t_2_operator]

    t_equation = compute_hausdorff(
        hamiltonian, cluster_function, sub_kwargs=sub_kwargs_ccsd
    )

    energy = get_energy_equation(t_equation)
    t_amplitudes_s = get_singles_amplitudes(
        t_equation, sub_kwargs=sub_kwargs_ccsd
    )
    t_amplitudes_d = get_doubles_amplitudes(
        t_equation, sub_kwargs=sub_kwargs_ccsd
    )

    return energy, [t_amplitudes_s, t_amplitudes_d]


def get_ccdt_equations():
    sub_kwargs_ccdt = sub_kwargs
    sub_kwargs_ccdt["pretty_indices"] = pretty_dummies_t

    hamiltonian = sum(get_hamiltonian())

    cluster_function = [get_t_2_operator, get_t_3_operator]

    equation = compute_hausdorff(
        hamiltonian, cluster_function, sub_kwargs=sub_kwargs_ccdt
    )

    energy = get_energy_equation(equation)
    amplitudes_d = get_doubles_amplitudes(equation, sub_kwargs=sub_kwargs_ccdt)
    amplitudes_t = get_triples_amplitudes(equation, sub_kwargs=sub_kwargs_ccdt)

    return energy, [amplitudes_d, amplitudes_t]


if __name__ == "__main__":
    energy, t_amp, l_amp = get_ccd_equations()
    print("\nCCD:")
    print("E = ", latex(energy))
    print("(t-doubles) 0 = ", latex(t_amp[0]))
    print("(l-doubles) 0 = ", latex(l_amp[0]))

    # print("\n")
    # energy, amp = get_ccsd_equations()
    # print("CCSD:")
    # print("E = ", latex(energy))
    # print()
    # print("(singles) 0 = ", latex(amp[0]))
    # print()
    # print("(doubles) 0 = ", latex(amp[1]))

    # print("\n")
    # energy, amp = get_ccdt_equations()
    # print("CCDT:")
    # print("E = ", latex(energy))
    # print()
    # print("(doubles) 0 = ", latex(amp[0]))
    # print()
    # print("(triples) 0 = ", latex(amp[1]))
