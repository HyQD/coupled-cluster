import collections
from sympy.physics.secondquant import (
    AntiSymmetricTensor,
    NO,
    Fd,
    F,
    KroneckerDelta,
)
from sympy import symbols, Dummy, Rational

from itertools import permutations
from functools import reduce
import operator


def eval_derivative(self, s):
    """Code created by Simen Kvaal."""
    # For use with Sympy diff()
    if s != self.symbol:
        return S.zero

    new_below = "IJKLM"
    new_above = "ABCDE"

    if len(self.upper + self.lower) > len(new_below + new_above):
        raise NotImplementedError

    # ASSUMPTION: tensor being differentiated has same occ/vir
    # structure as the original tensor.
    new_upper_symbols = []
    new_lower_symbols = []
    below_count = 0
    above_count = 0
    for s in self.upper + self.lower:
        properties = {
            "above_fermi": s.assumptions0.get("above_fermi"),
            "below_fermi": s.assumptions0.get("below_fermi"),
        }

        if properties["above_fermi"]:
            n = new_above[above_count]
            above_count += 1
        else:
            n = new_below[below_count]
            below_count += 1

        if s in self.upper:
            new_upper_symbols.append(symbols(n, **properties))
        else:
            new_lower_symbols.append(symbols(n, **properties))

    result = 0
    c_sigma = 1
    sign_sigma = 1

    for sigma in permutations(self.upper):
        # THIS IS A HACK ! FIND A BETTER WAY TO COMPUTE SIGN QUICKLY !
        if c_sigma % 2 == 0:
            sign_sigma *= -1

        c_sigma += 1

        c_tau = 1
        sign_tau = 1

        for tau in permutations(self.lower):

            if c_tau % 2 == 0:
                sign_tau *= -1

            c_tau += 1

            result += (
                sign_sigma
                * sign_tau
                * reduce(
                    operator.mul,
                    [
                        KroneckerDelta(s, j)
                        for s, j in zip(
                            sigma + tau, new_upper_symbols + new_lower_symbols
                        )
                    ],
                )
            )

    return result


# Add function computing the derivative of an anti-symmetric tensor
AntiSymmetricTensor._eval_derivative = eval_derivative


def get_clusters(cc_functions, *args, **kwargs):
    if not isinstance(cc_functions, collections.Iterable):
        cc_functions = [cc_functions]

    return sum([func(*args, **kwargs) for func in cc_functions])


def get_hamiltonian():
    """Generates normal ordered Hamiltonian. Remember to include the reference
    energy in the energy expressions. That is,

        E_ref = f^{i}_{i} - 0.5 * u^{ij}_{ij}
            = h^{i}_{i} + 0.5 * u^{ij}_{ij}.
    """
    p, q, r, s = symbols("p, q, r, s", cls=Dummy)
    f = AntiSymmetricTensor("f", (p,), (q,))
    u = AntiSymmetricTensor("u", (p, q), (r, s))

    f = f * NO(Fd(p) * F(q))
    u = u * NO(Fd(p) * Fd(q) * F(s) * F(r))

    return f, Rational(1, 4) * u


def get_t_1_operator(ast_symb="t_1"):
    i = symbols("i", below_fermi=True, cls=Dummy)
    a = symbols("a", above_fermi=True, cls=Dummy)

    t_ai = AntiSymmetricTensor(ast_symb, (a,), (i,))
    c_ai = NO(Fd(a) * F(i))

    T_1 = t_ai * c_ai

    return T_1


def get_t_2_operator(ast_symb="t_2"):
    i, j = symbols("i, j", below_fermi=True, cls=Dummy)
    a, b = symbols("a, b", above_fermi=True, cls=Dummy)

    t_abij = AntiSymmetricTensor(ast_symb, (a, b), (i, j))
    c_abij = NO(Fd(a) * Fd(b) * F(j) * F(i))

    T_2 = Rational(1, 4) * t_abij * c_abij

    return T_2


def get_t_3_operator(ast_symb="t_3"):
    i, j, k = symbols("i, j, k", below_fermi=True, cls=Dummy)
    a, b, c = symbols("a, b, c", above_fermi=True, cls=Dummy)

    t_abcijk = AntiSymmetricTensor(ast_symb, (a, b, c), (i, j, k))
    c_abcijk = NO(Fd(a) * Fd(b) * Fd(c) * F(k) * F(j) * F(i))

    T_3 = Rational(1, 36) * t_abcijk * c_abcijk

    return T_3


def get_l_1_operator(ast_symb="l_1"):
    i = symbols("i", below_fermi=True, cls=Dummy)
    a = symbols("a", above_fermi=True, cls=Dummy)

    l_ia = AntiSymmetricTensor(ast_symb, (i,), (a,))
    c_ia = NO(Fd(i) * F(a))

    L_1 = l_ia * c_ia

    return L_1


def get_l_2_operator(ast_symb="l_2"):
    i, j = symbols("i, j", below_fermi=True, cls=Dummy)
    a, b = symbols("a, b", above_fermi=True, cls=Dummy)

    l_ijab = AntiSymmetricTensor(ast_symb, (i, j), (a, b))
    c_ijab = NO(Fd(i) * Fd(j) * F(b) * F(a))

    L_2 = Rational(1, 4) * l_ijab * c_ijab

    return L_2


def get_l_3_operator(ast_symb="l_3"):
    i, j, k = symbols("i, j, k", below_fermi=True, cls=Dummy)
    a, b, c = symbols("a, b, c", above_fermi=True, cls=Dummy)

    l_ijkabc = AntiSymmetricTensor(ast_symb, (i, j, k), (a, b, c))
    c_ijkabc = NO(Fd(i) * Fd(j) * Fd(k) * F(c) * F(b) * F(a))

    L_3 = Rational(1, 36) * l_ijkabc * c_ijkabc

    return L_3
