#!/usr/bin/env python
"""Run He/cc-pVDZ delta-pulse propagation with the Gauss integrator."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from electromagnetic_fields.dipole_fields.delta_pulses import (
    DiscreteDeltaPulse,
)
from gauss_integrator import GaussIntegrator  # noqa: F401
from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction
from scipy.integrate import complex_ode

from corrected_roatdccd import CorrectedROATDCCD, StaleROATDCCD
from coupled_cluster.rccd import ROACCD
from coupled_cluster.rccsd import RCCSD, TDRCCSD

DEFAULT_DT = 0.2
DEFAULT_T_FINAL = 5.0
FIELD_STRENGTH = 1e-3
POLARIZATION_DIRECTION = 2
GROUND_STATE_TOLERANCE = 1e-10
DEFAULT_GAUSS_STAGES = 3
DEFAULT_GAUSS_MAX_ITERATIONS = 20
GAUSS_TOLERANCE = 1e-10
METHODS = ("tdrccsd", "stale_roatdccd", "corrected_roatdccd")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--t-final", type=float, default=DEFAULT_T_FINAL)
    parser.add_argument("--stages", type=int, default=DEFAULT_GAUSS_STAGES)
    parser.add_argument(
        "--maxit", type=int, default=DEFAULT_GAUSS_MAX_ITERATIONS
    )
    return parser.parse_args()


def timestep_label(dt):
    return f"{dt:g}".replace(".", "p")


def construct_initial_state_and_solver(method, system):
    if method == "tdrccsd":
        ground_state = RCCSD(system)
        ground_state.compute_ground_state(
            t_kwargs={"tol": GROUND_STATE_TOLERANCE},
            l_kwargs={"tol": GROUND_STATE_TOLERANCE},
        )
        solver_class = TDRCCSD
    else:
        ground_state = ROACCD(system)
        ground_state.compute_ground_state(tol=GROUND_STATE_TOLERANCE)
        solver_class = (
            StaleROATDCCD if method == "stale_roatdccd" else CorrectedROATDCCD
        )

    ground_state_energy = ground_state.compute_energy()
    initial_state = ground_state.get_amplitudes(get_t_0=True).asarray()
    return ground_state_energy, initial_state, solver_class


def main():
    args = parse_args()
    method = args.method
    dt = args.dt
    t_final = args.t_final
    gauss_stages = args.stages
    gauss_max_iterations = args.maxit
    if dt <= 0 or t_final <= 0:
        raise ValueError("dt and t-final must be positive")
    if gauss_stages <= 0:
        raise ValueError("stages must be positive")
    if gauss_max_iterations <= 0:
        raise ValueError("maxit must be positive")
    num_steps = int(round(t_final / dt))
    if not np.isclose(num_steps * dt, t_final):
        raise ValueError("t-final must be an integer multiple of dt")

    output_directory = Path(__file__).parent / "results"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_stem = output_directory / (
        f"{method}_he_ccpvdz_delta_gauss_s{gauss_stages}_eps1e-10_dt"
        + timestep_label(dt)
    )

    system = construct_pyscf_system_rhf(
        molecule="He 0.0 0.0 0.0",
        basis="cc-pvdz",
        add_spin=False,
        anti_symmetrize=False,
    )
    ground_state_energy, initial_state, solver_class = (
        construct_initial_state_and_solver(method, system)
    )

    polarization = np.zeros(3)
    polarization[POLARIZATION_DIRECTION] = 1.0
    field = DiscreteDeltaPulse(F_str=FIELD_STRENGTH, dt=dt)
    system.set_time_evolution_operator(
        DipoleFieldInteraction(
            field,
            polarization_vector=polarization,
            gauge="length",
        )
    )
    assert system.has_one_body_time_evolution_operator
    assert not system.has_two_body_time_evolution_operator

    solver = solver_class(system)
    integrator = complex_ode(solver).set_integrator(
        "GaussIntegrator",
        s=gauss_stages,
        eps=GAUSS_TOLERANCE,
        maxit=gauss_max_iterations,
    )
    integrator.set_initial_value(initial_state, 0.0)

    time_points = np.arange(num_steps + 1, dtype=float) * dt
    energy = np.zeros(num_steps + 1, dtype=np.complex128)
    dipole_z = np.zeros(num_steps + 1, dtype=np.complex128)
    field_values = np.asarray(field(time_points), dtype=float)

    energy[0] = solver.compute_energy(integrator.t, integrator.y)
    dipole_z[0] = solver.compute_one_body_expectation_value(
        integrator.t,
        integrator.y,
        system.dipole_moment[POLARIZATION_DIRECTION],
        make_hermitian=False,
    )

    for step in range(num_steps):
        integrator.integrate(time_points[step + 1])
        if not integrator.successful():
            raise RuntimeError(f"Gauss integration failed at step {step + 1}")
        energy[step + 1] = solver.compute_energy(integrator.t, integrator.y)
        dipole_z[step + 1] = solver.compute_one_body_expectation_value(
            integrator.t,
            integrator.y,
            system.dipole_moment[POLARIZATION_DIRECTION],
            make_hermitian=False,
        )

    tableau = GaussIntegrator(
        s=gauss_stages,
        eps=GAUSS_TOLERANCE,
        maxit=gauss_max_iterations,
    )
    effective_kick = dt * np.dot(tableau.b, field(dt * tableau.c))
    post_pulse_energy = energy[1:].real
    post_pulse_energy_range = np.ptp(post_pulse_energy)
    max_post_pulse_energy_deviation = np.max(
        np.abs(post_pulse_energy - post_pulse_energy[0])
    )
    scipy_integrator = integrator._integrator
    rhs_evaluations = scipy_integrator.rhs_evals

    result = {
        "time_points": time_points,
        "energy": energy,
        "dipole_z": dipole_z,
        "field": field_values,
        "ground_state_energy": ground_state_energy,
        "dt": dt,
        "t_final": t_final,
        "field_strength": FIELD_STRENGTH,
        "effective_gauss_kick": effective_kick,
        "post_pulse_energy_range": post_pulse_energy_range,
        "max_post_pulse_energy_deviation": max_post_pulse_energy_deviation,
        "gauss_stages": gauss_stages,
        "gauss_tolerance": GAUSS_TOLERANCE,
        "gauss_max_iterations": gauss_max_iterations,
        "rhs_evaluations": rhs_evaluations,
        "method": method,
    }
    if isinstance(solver, CorrectedROATDCCD):
        result["lab_hamiltonian_updates"] = solver.lab_hamiltonian_updates
        result["orbital_transform_updates"] = solver.orbital_transform_updates
    if isinstance(solver, StaleROATDCCD):
        result["hamiltonian_update_calls"] = solver.hamiltonian_update_calls
        result["stale_cache_hits"] = solver.stale_cache_hits
        result["stale_cache_misses"] = solver.stale_cache_misses
    np.savez(output_stem.with_suffix(".npz"), **result)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    axes[0].plot(time_points, dipole_z.real, marker="o", ms=3)
    axes[0].set_ylabel(r"$\mathrm{Re}\,\langle d_z(t)\rangle$")
    axes[0].grid(alpha=0.3)
    axes[1].plot(
        time_points,
        energy.real - ground_state_energy.real,
        marker="o",
        ms=3,
        label=r"$\mathrm{Re}[E(t)-E_\mathrm{gs}]$",
    )
    axes[1].set_xlabel("Time [a.u.]")
    axes[1].set_ylabel("Energy change [a.u.]")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    title = method.replace("_", " ")
    fig.suptitle(
        f"He/cc-pVDZ {title}, length-gauge delta pulse, "
        f"Gauss s={gauss_stages}"
    )
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=180)
    plt.close(fig)

    print(f"Method:                    {method}")
    print(f"Time step:                 {dt:.12e}")
    print(f"Gauss stages/maxit:        {gauss_stages}/{gauss_max_iterations}")
    print(f"Ground-state energy:       {ground_state_energy.real:.15f}")
    print(f"Nominal pulse strength:    {FIELD_STRENGTH:.12e}")
    print(f"Effective Gauss kick:      {float(effective_kick):.12e}")
    print(f"Maximum |dipole_z|:        {np.max(np.abs(dipole_z)):.12e}")
    print(f"Post-pulse energy range:   {post_pulse_energy_range:.12e}")
    print(
        "Max post-pulse deviation: " f"{max_post_pulse_energy_deviation:.12e}"
    )
    print(f"Maximum |Im(dipole_z)|:    {np.max(np.abs(dipole_z.imag)):.12e}")
    print(f"Maximum |Im(energy)|:      {np.max(np.abs(energy.imag)):.12e}")
    print(f"RHS evaluations:           {rhs_evaluations}")
    if isinstance(solver, CorrectedROATDCCD):
        print(f"Lab Hamiltonian updates:   {solver.lab_hamiltonian_updates}")
        print(f"Orbital transformations:   {solver.orbital_transform_updates}")
    if isinstance(solver, StaleROATDCCD):
        print(f"Hamiltonian update calls:  {solver.hamiltonian_update_calls}")
        print(f"Stale-cache early returns: {solver.stale_cache_hits}")
        print(f"Full transformations:      {solver.stale_cache_misses}")
    print(f"Saved data: {output_stem.with_suffix('.npz')}")
    print(f"Saved plot: {output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
