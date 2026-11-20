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

Whether the current cache fails inside an implicit Gauss step depends on the
order in which the stage RHS values are evaluated. The installed
`gauss_integrator` solves its stage equations by fixed-point iteration using
the call sequence

```text
c_1, c_2, c_3, c_1, c_2, c_3, ...
```

for `s = 3`. Although each stage time is revisited with an updated trial state,
two consecutive RHS calls have different times. The one-entry
`last_timestep` cache is therefore invalidated between visits, and the stale
early return is not taken in this implementation. A Gauss implementation that
performs consecutive updates at one stage time could still expose the issue;
the important condition is consecutive same-time calls with different orbital
states, not the name of the integrator.

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

### Gauss `s = 3`, `eps = 1e-10`

The same three methods were propagated with the installed sixth-order
three-stage Gauss integrator. Its internal stage nodes all lie inside the first
`DiscreteDeltaPulse` interval, giving the full effective kick `1e-3` for every
time step. This differs from the earlier RK4 effective kick `8.3333e-4`, so
absolute RK4 and Gauss dipole amplitudes should not be compared as if the
applied impulses were identical.

The post-pulse energy ranges were

| `dt` | TDRCCSD | stale ROATDCCD | corrected ROATDCCD |
| ---: | ---: | ---: | ---: |
| 0.20 | 2.6334e-13 | 1.7586e-13 | 1.7586e-13 |
| 0.10 | 3.8636e-14 | 2.4869e-14 | 2.4869e-14 |
| 0.05 | 1.2390e-13 | 8.3045e-14 | 8.3045e-14 |

These values have reached a `1e-14`--`1e-13` solver/roundoff floor: the range
decreases from `dt = 0.2` to `0.1` but rises at `0.05`, so fitting the nominal
sixth-order truncation error would not be meaningful at `eps = 1e-10`.

Stale and corrected ROATDCCD produced the same energy ranges and differed by
at most `1.1e-19` a.u. in the sampled dipole. Their maximum dipole differences
from TDRCCSD were

| `dt` | stale ROATDCCD | corrected ROATDCCD |
| ---: | ---: | ---: |
| 0.20 | 1.3620e-11 | 1.3620e-11 |
| 0.10 | 4.8698e-12 | 4.8698e-12 |
| 0.05 | 1.5941e-12 | 1.5941e-12 |

Thus this particular Gauss implementation does not activate the stale-cache
path during the propagation. The result does not validate the old cache in
general; it shows that the bug is sensitive to RHS call ordering and is masked
here by the cyclic stage evaluation described above.

### Gauss `s = 1`, `eps = 1e-10`

With one Gauss stage (implicit midpoint), the fixed-point call sequence is

```text
c_1, c_1, c_1, ...
```

so successive trial orbital states are evaluated at exactly the same time.
This directly activates the stale early return. The default `maxit = 20` was
insufficient for TDRCCSD at `dt = 0.2`; the iteration ceiling was therefore
raised to 100 for all methods and time steps while retaining the requested
`eps = 1e-10` convergence criterion.

The stale ROATDCCD calculation produced an identically zero dipole at every
time step. Its tiny energy range, `1.3e-15`--`2.7e-15` a.u., is not evidence of
accurate energy conservation: the erroneous trajectory remains at the
unperturbed state and never responds to the pulse. Corrected ROATDCCD instead
reproduces TDRCCSD. The post-pulse energy ranges were

| `dt` | TDRCCSD | stale ROATDCCD | corrected ROATDCCD |
| ---: | ---: | ---: | ---: |
| 0.20 | 6.9278e-14 | 1.3323e-15 | 5.0182e-14 |
| 0.10 | 1.6760e-12 | 1.7764e-15 | 1.1293e-12 |
| 0.05 | 6.9056e-13 | 2.6645e-15 | 4.5874e-13 |

The maximum dipole differences were

| `dt` | corrected ROATDCCD versus TDRCCSD | stale ROATDCCD versus TDRCCSD |
| ---: | ---: | ---: |
| 0.20 | 7.8066e-11 | 8.3519e-4 |
| 0.10 | 7.8110e-11 | 8.6112e-4 |
| 0.05 | 7.5452e-11 | 8.6767e-4 |

The corrected differences again lie below the `1e-10` ground-state amplitude
convergence tolerance.

Instrumentation also shows why the stale solver appears much cheaper:

| `dt` | stale RHS calls | corrected RHS calls | stale early returns | stale full transforms | corrected lab updates | corrected transforms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 40 | 377 | 15 | 51 | 51 | 403 |
| 0.10 | 52 | 404 | 2 | 101 | 101 | 455 |
| 0.05 | 102 | 603 | 2 | 201 | 201 | 704 |

The stale early returns freeze the transformed Hamiltonian during the crucial
first-step fixed-point solve, causing false convergence and the zero-response
trajectory. The low RHS count is therefore a symptom of the bug, not a useful
optimization. The corrected split cache behaves as intended: its lab-frame
update counts equal the stale solver's full-update counts, while it performs
the additional orbital transformations required for each changing trial
state.

The production runs and analysis are reproducible with
`dev/run_corrected_roatdccd_delta_rk4.py` and
`dev/compare_corrected_roatdccd_results.py`. The associated energy, dipole,
direct method-comparison, and three-way plots are stored in `dev/results/`.
The Gauss runs use `dev/run_he_delta_gauss.py`, and their combined analysis
uses `dev/analyze_he_delta_gauss.py`.
