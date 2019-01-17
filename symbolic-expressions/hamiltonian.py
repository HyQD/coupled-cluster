from sympy.physics.secondquant import AntiSymmetricTensor, NO, Fd, F
from sympy import symbols, Dummy, Rational


def get_hamiltonian():
    p, q, r, s = symbols("p, q, r, s", cls=Dummy)
    f = AntiSymmetricTensor("f", (p,), (q,))
    u = AntiSymmetricTensor("u", (p, q), (r, s))

    f = f * NO(Fd(p) * F(q))
    u = u * NO(Fd(p) * Fd(q) * F(s) * F(r))

    return f + Rational(1, 4) * u
