# Stale transformed Hamiltonian in OATDCC propagation

## Summary

`OATDCC.update_hamiltonian()` returns early when `current_time` equals
`self.last_timestep`. This is valid for the laboratory-frame integrals
`h(t)` and `u(t)`, which depend only on time, but it is not valid for the
orbital-basis integrals

```text
h'(t, C, C_tilde) = C_tilde h(t) C
u'(t, C, C_tilde) = transform(u(t), C, C_tilde).
```

Those quantities also depend on the orbital coefficients stored in the ODE
state. Two RHS evaluations at the same time can therefore require different
`h_prime`, `u_prime`, and `f_prime` values.

This is not specific to a particular integrator. It occurs whenever the RHS
or an observable is evaluated as

```text
evaluate(t, state_1)
evaluate(t, state_2),  where state_1 != state_2.
```

Time-only caching treats these evaluations as equivalent even though their
orbital transformations differ. Implicit iteration, repeated stage abscissae,
step boundaries, rejected steps, restarts, and observable sampling can all
produce this pattern.

An observable can therefore also receive stale transformed integrals when it
is evaluated at the same time as the last RHS call but with a different state.

## Minimal same-time reproduction

A direct He/cc-pVDZ test using two different orbital states at the same time
gave

```text
quantity   stale error   error after forced refresh
h_prime    1.63e-1       0
u_prime    4.94e-1       0
f_prime    1.37e-1       0
```

This test does not depend on an integration algorithm; it isolates the invalid
cache assumption itself.

## Examples of how integrators expose the issue

### Implicit Gauss integration

The implicit Gauss integrator solves its stage equations by fixed-point
iteration. During that iteration it repeatedly evaluates the RHS at the same
stage times,

```text
t_stage = t_n + dt * c_i,
```

but with successively updated trial states. For OATDCC, each trial state has
new `C` and `C_tilde` matrices. The first evaluation at a stage time computes
the transformed integrals; later evaluations at that same time hit the early
return and incorrectly reuse the first trial state's transformed integrals.

### Classical RK4 reproduction

The issue was also reproduced with two steps of classical RK4 for OATDCCD on
He/cc-pVDZ with `dt = 0.01` and a constant dipole field. The test was confined
to a system with only a one-body time-evolution operator:

```python
assert system.has_one_body_time_evolution_operator
assert not system.has_two_body_time_evolution_operator
```

For every RK4 stage, the cached `h_prime`, `u_prime`, and `f_prime` were
compared with direct transformations using the orbital matrices from that
stage's state. Representative norm differences were

```text
stage          h_prime error   u_prime error   f_prime error
step 1: k2     0               0               0
step 1: k3     1.95e-4         1.87e-4         2.20e-4
step 1: k4     0               0               0
step 2: k1     2.06e-6         1.25e-6         2.00e-6
step 2: k3     1.95e-4         1.87e-4         2.20e-4
```

Classical RK4 evaluates both `k2` and `k3` at `t + dt/2`, using different
trial states. The `k3` call therefore hits the time-only cache and reuses the
transformed integrals from `k2`. Similarly, one step's `k4` and the next
step's `k1` share the same time but use different states, producing the
smaller `k1` discrepancy above.

The nonzero `u_prime` error does not involve a two-body time-evolution
operator. The laboratory-frame `u` is static in this test, but its
orbital-basis representation still changes with `C` and `C_tilde`.

## Consequences

The amplitude and orbital derivatives are evaluated using an inconsistent
combination of the current amplitudes/orbitals and stale transformed
Hamiltonian. This can alter nonlinear solver convergence, time trajectories,
energies, and other observables. The problem applies to all solvers derived
from `OATDCC`, including OATDCCD, ROATDCCD, TDOMP2, and TDROMP2, independently
of which integration algorithm is selected.

## Recommended fix

Separate the two kinds of work in `update_hamiltonian()`:

1. Cache updates of the laboratory-frame `h(t)` and `u(t)` by time.
2. Always transform the current `h` and `u` with the supplied `C` and
   `C_tilde`, and rebuild `f_prime`, on every OATDCC RHS or observable
   evaluation.

Removing the early return entirely is the simplest correct fix. A more
efficient implementation may retain the time cache only around calls to
`system.h_t()` and `system.u_t()`; the orbital transformations themselves
must not be skipped based on time alone.

## Regression test

Construct two OATDCC state vectors with different biorthogonal orbital
matrices, call `update_hamiltonian()` for both at the same time, and compare
the second result against a direct `system.transform_one_body_elements()` and
`system.transform_two_body_elements()` calculation. The test should fail with
the current time-only cache and pass once the transformations are recomputed
for every state.

The development scripts `dev/test_oatdcc_same_time_cache.py` and
`dev/test_oatdcc_rk4_cache.py` reproduce the direct and RK4 cases,
respectively. Both explicitly require a one-body time-evolution operator and
reject systems with a two-body time-evolution operator.

## Split-cache validation and He benchmark

A development implementation of the recommended fix is provided in
`dev/corrected_roatdccd.py`. It retains a time cache for the laboratory-frame
`h(t)` and `u(t)`, but transforms them and rebuilds `f_prime` for every supplied
orbital state. The production `OATDCC` source is not changed by this
experiment.

### Correctness against a fully uncached control

`dev/validate_corrected_roatdccd_cache.py` propagates the corrected and fully
uncached implementations side by side for two complete RK4 steps. This covers
the repeated `t + dt/2` stage and a shared step endpoint. At every accepted
step it compares the state vector, energy, and dipole moment. The result was

```text
maximum state difference    0
maximum energy difference   0
maximum dipole difference   0

                             lab updates   transformations
split cache                        5              10
fully uncached                    10              10
```

Thus the split cache gives the same evaluated dynamics as recomputing
everything, while avoiding redundant evaluations of the time-dependent
laboratory-frame Hamiltonian.

### Physical test

The systematic benchmark uses

- He/cc-pVDZ with the PySCF backend;
- the converged ROACCD ground state as the initial state;
- ROATDCCD, TDRCCSD, and the old stale-cache ROATDCCD;
- a length-gauge, z-polarized `DiscreteDeltaPulse` with strength `1e-3`;
- classical RK4 to `T = 5` a.u. at `dt = 0.2`, `0.1`, and `0.05`;
- only a one-body time-evolution operator.

For this two-electron system, ROATDCCD and TDRCCSD should both reproduce the
finite-basis exact dynamics. The post-pulse energy ranges were

| `dt` | TDRCCSD | corrected ROATDCCD | stale ROATDCCD |
| ---: | ---: | ---: | ---: |
| 0.20 | 9.8040e-9 | 6.6393e-9 | 4.6921e-7 |
| 0.10 | 3.2857e-10 | 2.2514e-10 | 4.2116e-7 |
| 0.05 | 1.0483e-11 | 7.3421e-12 | 4.0122e-7 |

The corrected ROATDCCD energy range converges with fitted order `4.91`
(pairwise orders `4.88` and `4.94`), consistent with the TDRCCSD fitted order
`4.93`. The stale result has fitted order `0.11` and remains near `4e-7` a.u.
as `dt` is refined.

The maximum absolute dipole differences relative to TDRCCSD were

| `dt` | corrected ROATDCCD | stale ROATDCCD |
| ---: | ---: | ---: |
| 0.20 | 1.4869e-11 | 2.8113e-4 |
| 0.10 | 5.1686e-11 | 7.1720e-5 |
| 0.05 | 5.8133e-11 | 1.9425e-5 |

The corrected dipole is therefore indistinguishable from TDRCCSD on the plot
scale; its residual difference is at the `1e-11` level. In contrast, the
stale-cache trajectory has a visible phase/amplitude error. Together with the
exact uncached-control agreement, these results strongly support the cache
split as the correct fix for this test case.

The production runs and analysis are reproducible with
`dev/run_corrected_roatdccd_delta_rk4.py` and
`dev/compare_corrected_roatdccd_results.py`. The associated energy, dipole,
direct method-comparison, and three-way plots are stored in `dev/results/`.
