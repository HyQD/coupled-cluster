from sympy.physics.secondquant import wicks, evaluate_deltas, substitute_dummies
from sympy import expand

pretty_dummies_dict = {
    "above": "abcdef",
    "below": "ijklmn",
    "general": "pqrstu",
}

wicks_kwargs = {
    "simplify_dummies": True,
    "keep_only_fully_contracted": True,
    "simplify_kronecker_deltas": True,
}

sub_kwargs = {"new_indices": True, "pretty_indices": pretty_dummies_dict}


def beautify_equation(eq, sub_kwargs=sub_kwargs):
    eq = evaluate_deltas(expand(eq))
    eq = substitute_dummies(eq, **sub_kwargs)

    return eq


def eval_equation(eq, wicks_kwargs=wicks_kwargs, sub_kwargs=sub_kwargs):
    eq = wicks(eq, **wicks_kwargs)
    eq = beautify_equation(eq, sub_kwargs=sub_kwargs)

    return eq
