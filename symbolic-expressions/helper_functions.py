from symbols.physics.secondquant import (
    wicks,
    evaluate_deltas,
    substitute_dummies,
)

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


def eval_equation(eq, wicks_kwargs=wicks_kwargs, sub_kwargs=sub_kwargs):
    eq = wicks(eq, **wicks_kwargs)
    eq = evaluate_deltas(eq.expand())
    eq = substitute_dummies(eq, **sub_kwargs)

    return eq
