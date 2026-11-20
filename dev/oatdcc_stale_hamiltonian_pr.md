# Fix stale orbital-transformed Hamiltonians in OATDCC

## Summary

- Cache only the laboratory-frame Hamiltonian integrals `h(t)` and `u(t)` by
  time in `OATDCC.update_hamiltonian()`.
- Recompute `h_prime`, `u_prime`, and `f_prime` for every supplied orbital
  state, including repeated evaluations at the same time.
- Add a regression test for same-time evaluations with different
  biorthogonal orbitals, while verifying that `system.h_t()` remains cached.

## Problem

The previous implementation returned immediately whenever `current_time`
matched `last_timestep`. That is valid for laboratory-frame integrals, which
depend only on time, but not for their orbital-basis transformations, which
also depend on the current `C` and `C_tilde` matrices.

Consequently, consecutive calls such as RK4's `k2`/`k3` pair or fixed-point
iterations in the one-stage Gauss method could evaluate the amplitude and
orbital equations using transformed integrals from a different trial state.

## Implementation

`OATDCC.update_hamiltonian()` now:

1. resolves the current orbital matrices;
2. updates `h(t)` and `u(t)` only when the time changes; and
3. always transforms those integrals with the current orbitals and rebuilds
   the Fock matrix.

This preserves the useful time cache without treating time as a complete
cache key for state-dependent quantities. There is no public API change.

The development comparison scripts retain an explicit historical stale
subclass so the investigation remains reproducible after production
`ROATDCCD` adopts the corrected behavior.

## Validation

The new regression test constructs two states with different biorthogonal
orbital matrices, evaluates them at the same time, and checks that:

- the second `h_prime`, `u_prime`, and `f_prime` match direct transformations;
- `system.h_t()` is called only once.

Targeted test suite:

```text
PYTHONPATH=. python -m pytest -q \
  tests/test_oatdccd.py tests/test_roatdccd.py tests/test_tdomp2.py

7 passed, 3 warnings
```

Full repository test suite:

```text
PYTHONPATH=. python -m pytest -q

332 passed, 3 skipped, 13 warnings
```

The standalone same-time and two-step RK4 diagnostics also report zero error
for every transformed integral at every stage.

## Numerical evidence

For He/cc-pVDZ with a length-gauge discrete delta pulse:

- corrected ROATDCCD agrees with TDRCCSD to at most `5.9e-11` a.u. in the RK4
  dipole tests;
- its RK4 post-pulse energy drift converges with fitted order `4.91`, compared
  with `4.93` for TDRCCSD;
- with the one-stage Gauss method, stale ROATDCCD falsely converges to an
  identically zero dipole, while corrected ROATDCCD agrees with TDRCCSD within
  `7.9e-11` a.u.; and
- the corrected split cache is numerically identical to a fully uncached
  control in state, energy, and dipole comparisons.

The three-stage Gauss method does not expose the old one-entry cache because
its RHS calls cycle through distinct stage times. This is an accidental
property of that evaluation order, not evidence that the old cache was valid.
