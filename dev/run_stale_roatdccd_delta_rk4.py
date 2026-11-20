#!/usr/bin/env python
"""Run the stale-cache ROATDCCD delta-pulse reference calculation."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rk4_integrator  # noqa: F401 - registers Rk4Integrator with SciPy
from electromagnetic_fields.dipole_fields.delta_pulses import (
    DiscreteDeltaPulse,
)
from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction
from scipy.integrate import complex_ode

from coupled_cluster.rccd import ROACCD, ROATDCCD

DEFAULT_DT = 0.2
DEFAULT_T_FINAL = 5.0
FIELD_STRENGTH = 1e-3
POLARIZATION_DIRECTION = 2
GROUND_STATE_TOLERANCE = 1e-10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--t-final", type=float, default=DEFAULT_T_FINAL)
    return parser.parse_args()


def timestep_label(dt):
    return f"{dt:g}".replace(".", "p")


def main():
    args = parse_args()
    dt = args.dt
    t_final = args.t_final
    if dt <= 0 or t_final <= 0:
        raise ValueError("dt and t-final must be positive")
    num_steps = int(round(t_final / dt))
    if not np.isclose(num_steps * dt, t_final):
        raise ValueError("t-final must be an integer multiple of dt")

    output_directory = Path(__file__).parent / "results"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_stem = output_directory / (
        "stale_roatdccd_he_ccpvdz_delta_rk4_dt" + timestep_label(dt)
    )

    system = construct_pyscf_system_rhf(
        molecule="He 0.0 0.0 0.0",
        basis="cc-pvdz",
        add_spin=False,
        anti_symmetrize=False,
    )

    roaccd = ROACCD(system)
    roaccd.compute_ground_state(tol=GROUND_STATE_TOLERANCE)
    ground_state_energy = roaccd.compute_energy()
    initial_state = roaccd.get_amplitudes(get_t_0=True).asarray()

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

    solver = ROATDCCD(system)
    integrator = complex_ode(solver).set_integrator("Rk4Integrator", dt=dt)
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
            raise RuntimeError(f"RK4 failed at step {step + 1}")

        # These calls intentionally preserve the old solver's time-only cache.
        energy[step + 1] = solver.compute_energy(integrator.t, integrator.y)
        dipole_z[step + 1] = solver.compute_one_body_expectation_value(
            integrator.t,
            integrator.y,
            system.dipole_moment[POLARIZATION_DIRECTION],
            make_hermitian=False,
        )

    effective_kick = (
        dt
        * (field(0.0) + 2 * field(0.5 * dt) + 2 * field(0.5 * dt) + field(dt))
        / 6
    )
    post_pulse_energy = energy[1:].real
    post_pulse_energy_range = np.ptp(post_pulse_energy)
    max_post_pulse_energy_deviation = np.max(
        np.abs(post_pulse_energy - post_pulse_energy[0])
    )

    np.savez(
        output_stem.with_suffix(".npz"),
        time_points=time_points,
        energy=energy,
        dipole_z=dipole_z,
        field=field_values,
        ground_state_energy=ground_state_energy,
        dt=dt,
        t_final=t_final,
        field_strength=FIELD_STRENGTH,
        effective_rk4_kick=effective_kick,
        post_pulse_energy_range=post_pulse_energy_range,
        max_post_pulse_energy_deviation=max_post_pulse_energy_deviation,
        stale_hamiltonian_cache=True,
    )

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
    fig.suptitle("He/cc-pVDZ stale ROATDCCD, length-gauge delta pulse, RK4")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=180)
    plt.close(fig)

    print(f"Time step:                 {dt:.12e}")
    print(f"Ground-state energy:       {ground_state_energy.real:.15f}")
    print(f"Nominal pulse strength:    {FIELD_STRENGTH:.12e}")
    print(f"Effective RK4 kick:        {float(effective_kick):.12e}")
    print(f"Maximum |dipole_z|:        {np.max(np.abs(dipole_z)):.12e}")
    print(f"Post-pulse energy range:   {post_pulse_energy_range:.12e}")
    print(
        "Max post-pulse deviation: " f"{max_post_pulse_energy_deviation:.12e}"
    )
    print(f"Maximum |Im(dipole_z)|:    {np.max(np.abs(dipole_z.imag)):.12e}")
    print(f"Maximum |Im(energy)|:      {np.max(np.abs(energy.imag)):.12e}")
    print(f"Saved data: {output_stem.with_suffix('.npz')}")
    print(f"Saved plot: {output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
