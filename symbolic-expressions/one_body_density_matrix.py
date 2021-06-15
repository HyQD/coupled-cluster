import collections
import math

from sympy.physics.secondquant import F, Fd, Commutator, evaluate_deltas
from sympy import symbols, Dummy, Rational

from helper_functions import eval_equation, beautify_equation
from cluster_operators import (
    get_t_1_operator,
    get_t_2_operator,
    get_l_1_operator,
    get_l_2_operator,
    get_clusters,
)

symbol_list_one_body = [
    ("rho^{b}_{a} = ", symbols("a, b", above_fermi=True, cls=Dummy)),
    (
        "rho^{i}_{a} = ",
        (
            symbols("a", above_fermi=True, cls=Dummy),
            symbols("i", below_fermi=True, cls=Dummy),
        ),
    ),
    (
        "rho^{a}_{i} = ",
        (
            symbols("i", below_fermi=True, cls=Dummy),
            symbols("a", above_fermi=True, cls=Dummy),
        ),
    ),
    ("rho^{j}_{i} = ", symbols("i, j", below_fermi=True, cls=Dummy)),
]

symbol_list_two_body = [
    ("rho^{cd}_{ab} = ", symbols("a, b, c, d", above_fermi=True, cls=Dummy)),
    # We have that:
    # rho[o, v, v, v] = -rho[v, o, v, v]
    (
        "rho^{kd}_{ab} = ",
        (
            *symbols("a, b", above_fermi=True, cls=Dummy),
            symbols("k", below_fermi=True, cls=Dummy),
            symbols("d", above_fermi=True, cls=Dummy),
        ),
    ),
    # (
    #     "rho^{cl}_{ab} = ",
    #     (
    #         *symbols("a, b, c", above_fermi=True, cls=Dummy),
    #         symbols("l", below_fermi=True, cls=Dummy),
    #     ),
    # ),
    # We have that:
    # rho[v, v, o, v] = -rho[v, v, v, o]
    (
        "rho^{cd}_{ib} = ",
        (
            symbols("i", below_fermi=True, cls=Dummy),
            *symbols("b, c, d", above_fermi=True, cls=Dummy),
        ),
    ),
    # (
    #     "rho^{cd}_{aj} = ",
    #     (
    #         symbols("a", above_fermi=True, cls=Dummy),
    #         symbols("j", below_fermi=True, cls=Dummy),
    #         *symbols("c, d", above_fermi=True, cls=Dummy),
    #     ),
    # ),
    (
        "rho^{kl}_{ab} = ",
        (
            *symbols("a, b", above_fermi=True, cls=Dummy),
            *symbols("k, l", below_fermi=True, cls=Dummy),
        ),
    ),
    (
        "rho^{kd}_{ib} = ",
        (
            symbols("i", below_fermi=True, cls=Dummy),
            symbols("b", above_fermi=True, cls=Dummy),
            symbols("k", below_fermi=True, cls=Dummy),
            symbols("d", above_fermi=True, cls=Dummy),
        ),
    ),
    # We have that:
    # rho[o, v, o, v] = -rho[o, v, v, o] = -rho[v, o, o, v]
    #       = rho[v, o, v, o]
    # (
    #     "rho^{kd}_{aj} = ",
    #     (
    #         symbols("a", above_fermi=True, cls=Dummy),
    #         symbols("j", below_fermi=True, cls=Dummy),
    #         symbols("k", below_fermi=True, cls=Dummy),
    #         symbols("d", above_fermi=True, cls=Dummy),
    #     ),
    # ),
    # (
    #     "rho^{cl}_{ib} = ",
    #     (
    #         symbols("i", below_fermi=True, cls=Dummy),
    #         symbols("b", above_fermi=True, cls=Dummy),
    #         symbols("c", above_fermi=True, cls=Dummy),
    #         symbols("l", below_fermi=True, cls=Dummy),
    #     ),
    # ),
    # (
    #     "rho^{cl}_{aj} = ",
    #     (
    #         symbols("a", above_fermi=True, cls=Dummy),
    #         symbols("j", below_fermi=True, cls=Dummy),
    #         symbols("c", above_fermi=True, cls=Dummy),
    #         symbols("l", below_fermi=True, cls=Dummy),
    #     ),
    # ),
    # We have that:
    # rho[v, o, o, o] = -rho[o, v, o, o]
    (
        "rho^{cl}_{ij} = ",
        (
            symbols("i", below_fermi=True, cls=Dummy),
            symbols("j", below_fermi=True, cls=Dummy),
            symbols("c", above_fermi=True, cls=Dummy),
            symbols("l", below_fermi=True, cls=Dummy),
        ),
    ),
    # (
    #     "rho^{kd}_{ij} = ",
    #     (
    #         symbols("i", below_fermi=True, cls=Dummy),
    #         symbols("j", below_fermi=True, cls=Dummy),
    #         symbols("k", below_fermi=True, cls=Dummy),
    #         symbols("d", above_fermi=True, cls=Dummy),
    #     ),
    # ),
    # We have that:
    # rho[o, o, v, o] = -rho[o, o, o, v]
    (
        "rho^{kl}_{aj} = ",
        (
            symbols("a", above_fermi=True, cls=Dummy),
            symbols("j", below_fermi=True, cls=Dummy),
            symbols("k", below_fermi=True, cls=Dummy),
            symbols("l", below_fermi=True, cls=Dummy),
        ),
    ),
    # (
    #     "rho^{kl}_{ib} = ",
    #     (
    #         symbols("i", below_fermi=True, cls=Dummy),
    #         symbols("b", above_fermi=True, cls=Dummy),
    #         symbols("k", below_fermi=True, cls=Dummy),
    #         symbols("l", below_fermi=True, cls=Dummy),
    #     ),
    # ),
    ("rho^{kl}_{ij} = ", symbols("i, j, k, l", below_fermi=True, cls=Dummy)),
]


def get_one_body_density_operator(p, q):
    return Fd(p) * F(q)


def get_two_body_density_operator(p, q, r, s):
    return Fd(p) * Fd(q) * F(s) * F(r)


def get_one_body_density_matrix(
    cc_t_functions, cc_l_functions, num_commutators, p=None, q=None
):
    if p is None:
        p = symbols("p", cls=Dummy)

    if q is None:
        q = symbols("q", cls=Dummy)

    c_pq = get_one_body_density_operator(p, q)

    if not isinstance(cc_t_functions, collections.Iterable):
        cc_t_functions = [cc_t_functions]

    if not isinstance(cc_l_functions, collections.Iterable):
        cc_l_functions = [cc_l_functions]

    T = get_clusters(cc_t_functions)
    L = get_clusters(cc_l_functions)

    rho_eq = eval_equation(c_pq)
    rho_eq += eval_equation(Commutator(c_pq, T))
    rho_eq += eval_equation(L * c_pq)

    comm = c_pq

    for i in range(1, num_commutators + 1):
        comm = Commutator(comm, get_clusters(cc_t_functions))
        rho_eq += Rational(1, int(math.factorial(i))) * eval_equation(L * comm)

    rho = beautify_equation(rho_eq)

    return rho


def get_two_body_density_matrix(
    cc_t_functions,
    cc_l_functions,
    num_commutators,
    p=None,
    q=None,
    r=None,
    s=None,
):
    if p is None:
        p = symbols("p", cls=Dummy)

    if q is None:
        q = symbols("q", cls=Dummy)

    if r is None:
        r = symbols("r", cls=Dummy)

    if s is None:
        s = symbols("s", cls=Dummy)

    c_pqrs = get_two_body_density_operator(p, q, r, s)

    if not isinstance(cc_t_functions, collections.Iterable):
        cc_t_functions = [cc_t_functions]

    if not isinstance(cc_l_functions, collections.Iterable):
        cc_l_functions = [cc_l_functions]

    T = get_clusters(cc_t_functions)
    L = get_clusters(cc_l_functions)

    rho_eq = eval_equation(c_pqrs)
    rho_eq += eval_equation(Commutator(c_pqrs, T))
    rho_eq += eval_equation(L * c_pqrs)

    # from sympy import latex
    # print(latex(beautify_equation(rho_eq)))

    comm = c_pqrs

    for i in range(1, num_commutators + 1):
        comm = Commutator(comm, get_clusters(cc_t_functions))
        rho_eq += Rational(1, int(math.factorial(i))) * eval_equation(L * comm)
        # print()
        # print(latex(beautify_equation(rho_eq)))

    rho = beautify_equation(rho_eq)
    # print()

    return rho


def get_ccs_one_body_density_matrix(p=None, q=None):
    return get_one_body_density_matrix(
        get_t_1_operator, get_l_1_operator, 1, p=p, q=q
    )


def get_ccs_two_body_density_matrix(p=None, q=None, r=None, s=None):
    return get_two_body_density_matrix(
        get_t_1_operator, get_l_1_operator, 1, p=p, q=q, r=r, s=s
    )


def get_ccd_one_body_density_matrix(p=None, q=None):
    rho = get_one_body_density_matrix(
        get_t_2_operator, get_l_2_operator, 1, p=p, q=q
    )

    return rho


def get_ccd_two_body_density_matrix(p=None, q=None, r=None, s=None):
    return get_two_body_density_matrix(
        get_t_2_operator, get_l_2_operator, 5, p=p, q=q, r=r, s=s
    )


def get_ccsd_one_body_density_matrix(p=None, q=None):
    rho = get_one_body_density_matrix(
        [get_t_1_operator, get_t_2_operator],
        [get_l_1_operator, get_l_2_operator],
        5,
        p=p,
        q=q,
    )

    return rho


def get_ccsd_two_body_density_matrix(p=None, q=None, r=None, s=None):
    rho = get_two_body_density_matrix(
        [get_t_1_operator, get_t_2_operator],
        [get_l_1_operator, get_l_2_operator],
        5,
        p=p,
        q=q,
        r=r,
        s=s,
    )

    return rho


if __name__ == "__main__":
    from sympy import latex

    p = symbols("p", above_fermi=True, cls=Dummy)
    q = symbols("q", above_fermi=True, cls=Dummy)
    p = q = None

    print("CCS:", latex(get_ccs_one_body_density_matrix(p=p, q=q)))
    print("CCD:", latex(get_ccd_one_body_density_matrix(p=p, q=q)))
    print("CCSD:", latex(get_ccsd_one_body_density_matrix(p=p, q=q)))
