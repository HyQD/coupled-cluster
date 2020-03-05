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

symbol_list = [
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


def get_one_body_density_operator(p, q):
    return Fd(p) * F(q)


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


def get_ccs_one_body_density_matrix(p=None, q=None):
    return get_one_body_density_matrix(
        get_t_1_operator, get_l_1_operator, 1, p=p, q=q
    )


def get_ccd_one_body_density_matrix(p=None, q=None):
    rho = get_one_body_density_matrix(
        get_t_2_operator, get_l_2_operator, 1, p=p, q=q
    )

    return rho


def get_ccsd_one_body_density_matrix(p=None, q=None):
    rho = get_one_body_density_matrix(
        [get_t_1_operator, get_t_2_operator],
        [get_l_1_operator, get_l_2_operator],
        2,
        p=p,
        q=q,
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
