#!/usr/bin/env python
"""TDRCCSD baseline for the OATDCC Hamiltonian-cache investigation."""

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

from coupled_cluster.rccsd import RCCSD, TDRCCSD

DT = 0.2
T_FINAL = 5.0
FIELD_STRENGTH = 1e-3
POLARIZATION_DIRECTION = 2
GROUND_STATE_TOLERANCE = 1e-10


def main():
    output_directory = Path(__file__).parent / "results"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_stem = output_directory / "tdrccsd_he_ccpvdz_delta_rk4_dt0p2"

    system = construct_pyscf_system_rhf(
        molecule="He 0.0 0.0 0.0",
        basis="cc-pvdz",
        add_spin=False,
        anti_symmetrize=False,
    )

    rccsd = RCCSD(system)
    rccsd.compute_ground_state(
        t_kwargs={"tol": GROUND_STATE_TOLERANCE},
        l_kwargs={"tol": GROUND_STATE_TOLERANCE},
    )
    ground_state_energy = rccsd.compute_energy()
    initial_state = rccsd.get_amplitudes(get_t_0=True).asarray()

    polarization = np.zeros(3)
    polarization[POLARIZATION_DIRECTION] = 1.0
    field = DiscreteDeltaPulse(F_str=FIELD_STRENGTH, dt=DT)
    system.set_time_evolution_operator(
        DipoleFieldInteraction(
            field,
            polarization_vector=polarization,
            gauge="length",
        )
    )

    assert system.has_one_body_time_evolution_operator
    assert not system.has_two_body_time_evolution_operator

    solver = TDRCCSD(system)
    integrator = complex_ode(solver).set_integrator("Rk4Integrator", dt=DT)
    integrator.set_initial_value(initial_state, 0.0)

    num_steps = int(round(T_FINAL / DT))
    time_points = np.arange(num_steps + 1, dtype=float) * DT
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

        energy[step + 1] = solver.compute_energy(integrator.t, integrator.y)
        dipole_z[step + 1] = solver.compute_one_body_expectation_value(
            integrator.t,
            integrator.y,
            system.dipole_moment[POLARIZATION_DIRECTION],
            make_hermitian=False,
        )

    # With the installed right-sided pulse and a pulse starting at t=0, RK4
    # samples the field at k1, k2, and k3, but not k4.
    effective_kick = (
        DT
        * (field(0.0) + 2 * field(0.5 * DT) + 2 * field(0.5 * DT) + field(DT))
        / 6
    )

    np.savez(
        output_stem.with_suffix(".npz"),
        time_points=time_points,
        energy=energy,
        dipole_z=dipole_z,
        field=field_values,
        ground_state_energy=ground_state_energy,
        dt=DT,
        t_final=T_FINAL,
        field_strength=FIELD_STRENGTH,
        effective_rk4_kick=effective_kick,
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

    fig.suptitle("He/cc-pVDZ TDRCCSD, length-gauge discrete delta pulse, RK4")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=180)
    plt.close(fig)

    post_pulse_energy = energy[1:].real
    print(f"Ground-state energy:       {ground_state_energy.real:.15f}")
    print(f"Nominal pulse strength:    {FIELD_STRENGTH:.12e}")
    print(f"Effective RK4 kick:        {float(effective_kick):.12e}")
    print(f"Maximum |dipole_z|:        {np.max(np.abs(dipole_z)):.12e}")
    print("Post-pulse energy range:  " f"{np.ptp(post_pulse_energy):.12e}")
    print(f"Maximum |Im(dipole_z)|:    {np.max(np.abs(dipole_z.imag)):.12e}")
    print(f"Maximum |Im(energy)|:      {np.max(np.abs(energy.imag)):.12e}")
    print(f"Saved data: {output_stem.with_suffix('.npz')}")
    print(f"Saved plot: {output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
