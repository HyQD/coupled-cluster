from helper_functions import beautify_equation, eval_equation
from cluster_operators import (
    get_clusters,
    get_t_1_operator,
    get_t_2_operator,
    get_l_1_operator,
    get_l_2_operator,
)
from sympy import Rational, latex


def ccd_contribution(T, T_t, L, L_t):
    tilde_t_eq = Rational(1, 1)
    tilde_t_eq += eval_equation(-L_t * T_t)
    tilde_t_eq += eval_equation(L_t * T)

    tilde_eq = Rational(1, 1)
    tilde_eq += eval_equation(-L * T)
    tilde_eq += eval_equation(L * T_t)

    return tilde_t_eq, tilde_eq


def ccsd_contribution(T, T_2, T_t, T_t_2, L, L_t):
    tilde_t_eq, tilde_eq = ccd_contribution(T, T_t, L, L_t)

    tilde_t_eq += eval_equation(-L_t * T_t * T)
    tilde_t_eq += eval_equation(Rational(1, 2) * L_t * T_t * T_t_2)
    tilde_t_eq += eval_equation(Rational(1, 2) * L_t * T * T_2)

    tilde_eq += eval_equation(-L * T * T_t)
    tilde_eq += eval_equation(Rational(1, 2) * L * T * T_2)
    tilde_eq += eval_equation(Rational(1, 2) * L * T_t * T_t_2)

    return tilde_t_eq, tilde_eq


def get_ccd_overlap():
    T = get_clusters(get_t_2_operator)
    T_t = get_clusters(get_t_2_operator, ast_symb="t(t)")
    L = get_clusters(get_l_2_operator)
    L_t = get_clusters(get_l_2_operator, ast_symb="l(t)")

    tilde_t_eq, tilde_eq = ccd_contribution(T, T_t, L, L_t)

    tilde_t_eq = beautify_equation(tilde_t_eq)
    tilde_eq = beautify_equation(tilde_eq)

    return tilde_t_eq, tilde_eq


def get_ccsd_overlap():
    t_funcs = [get_t_1_operator, get_t_2_operator]
    l_funcs = [get_l_1_operator, get_l_2_operator]

    T = get_clusters(t_funcs)
    T_2 = get_clusters(t_funcs)
    T_t = get_clusters(t_funcs, ast_symb="t(t)")
    T_t_2 = get_clusters(t_funcs, ast_symb="t(t)")
    L = get_clusters(l_funcs)
    L_t = get_clusters(l_funcs, ast_symb="l(t)")

    tilde_t_eq, tilde_eq = ccsd_contribution(T, T_2, T_t, T_t_2, L, L_t)

    tilde_t_eq = beautify_equation(tilde_t_eq)
    tilde_eq = beautify_equation(tilde_eq)

    return tilde_t_eq, tilde_eq


if __name__ == "__main__":
    t_eq, eq = get_ccd_overlap()
    print("t_eq =", latex(t_eq))
    print("eq =", latex(eq))
    print()

    t_eq, eq = get_ccsd_overlap()
    print("t_eq =", latex(t_eq))
    print("eq =", latex(eq))
