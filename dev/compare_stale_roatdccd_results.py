#!/usr/bin/env python
"""Analyze stale-cache ROATDCCD and compare its dipole with TDRCCSD."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIME_STEPS = (0.2, 0.1, 0.05)


def timestep_label(dt):
    return f"{dt:g}".replace(".", "p")


def load_dataset(results_directory, prefix, dt):
    path = results_directory / (prefix + timestep_label(dt) + ".npz")
    data = np.load(path)
    return {
        "dt": dt,
        "time": data["time_points"],
        "energy": data["energy"].real,
        "dipole_z": data["dipole_z"].real,
    }


def main():
    results_directory = Path(__file__).parent / "results"
    stale_datasets = [
        load_dataset(
            results_directory,
            "stale_roatdccd_he_ccpvdz_delta_rk4_dt",
            dt,
        )
        for dt in TIME_STEPS
    ]
    tdrccsd_datasets = [
        load_dataset(
            results_directory,
            "tdrccsd_he_ccpvdz_delta_rk4_dt",
            dt,
        )
        for dt in TIME_STEPS
    ]

    for item in stale_datasets:
        post_pulse_energy = item["energy"][1:]
        item["energy_range"] = np.ptp(post_pulse_energy)
        item["max_energy_deviation"] = np.max(
            np.abs(post_pulse_energy - post_pulse_energy[0])
        )

    time_steps = np.array(TIME_STEPS)
    energy_ranges = np.array([item["energy_range"] for item in stale_datasets])
    fitted_order, log_prefactor = np.polyfit(
        np.log(time_steps), np.log(energy_ranges), 1
    )
    pairwise_orders = np.log(energy_ranges[:-1] / energy_ranges[1:]) / np.log(
        time_steps[:-1] / time_steps[1:]
    )

    energy_fig, energy_axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for item in stale_datasets:
        post_time = item["time"][1:]
        post_energy = item["energy"][1:]
        absolute_drift = np.abs(post_energy - post_energy[0])
        energy_axes[0].semilogy(
            post_time[1:],
            absolute_drift[1:],
            label=rf"$dt={item['dt']:g}$",
        )
    energy_axes[0].set_xlabel("Time [a.u.]")
    energy_axes[0].set_ylabel(r"$|E(t)-E(t_s)|$ [a.u.]")
    energy_axes[0].set_title("Post-pulse energy drift")
    energy_axes[0].grid(alpha=0.3, which="both")
    energy_axes[0].legend()

    order_grid = np.linspace(time_steps.min(), time_steps.max(), 200)
    fitted_curve = np.exp(log_prefactor) * order_grid**fitted_order
    energy_axes[1].loglog(
        time_steps,
        energy_ranges,
        "o",
        ms=7,
        label="Measured range",
    )
    energy_axes[1].loglog(
        order_grid,
        fitted_curve,
        "--",
        label=rf"Fit: $dt^{{{fitted_order:.2f}}}$",
    )
    energy_axes[1].set_xlabel(r"$dt$ [a.u.]")
    energy_axes[1].set_ylabel("Post-pulse energy range [a.u.]")
    energy_axes[1].set_title("Energy-conservation convergence")
    energy_axes[1].grid(alpha=0.3, which="both")
    energy_axes[1].legend()
    energy_fig.suptitle(
        "He/cc-pVDZ stale ROATDCCD with length-gauge delta pulse and RK4"
    )
    energy_fig.tight_layout()
    energy_path = (
        results_directory / "stale_roatdccd_energy_conservation_vs_dt.png"
    )
    energy_fig.savefig(energy_path, dpi=180)
    plt.close(energy_fig)

    dipole_fig, dipole_axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    for item in stale_datasets:
        dipole_axes[0].plot(
            item["time"],
            item["dipole_z"],
            marker="o",
            ms=2.5,
            markevery=max(1, round(0.2 / item["dt"])),
            label=rf"$dt={item['dt']:g}$",
        )
    dipole_axes[0].set_ylabel(r"$\mathrm{Re}\,\langle d_z(t)\rangle$")
    dipole_axes[0].set_title(
        "He/cc-pVDZ stale ROATDCCD dipole after length-gauge delta pulse"
    )
    dipole_axes[0].grid(alpha=0.3)
    dipole_axes[0].legend()

    reference = stale_datasets[-1]
    for item in stale_datasets[:-1]:
        reference_at_time = np.interp(
            item["time"], reference["time"], reference["dipole_z"]
        )
        dipole_axes[1].plot(
            item["time"],
            item["dipole_z"] - reference_at_time,
            marker="o",
            ms=2.5,
            label=rf"$dt={item['dt']:g}$",
        )
    dipole_axes[1].axhline(0, color="black", lw=0.8, alpha=0.5)
    dipole_axes[1].set_xlabel("Time [a.u.]")
    dipole_axes[1].set_ylabel(
        rf"$\langle d_z\rangle_{{dt}}-\langle d_z\rangle_{{{reference['dt']:g}}}$"
    )
    dipole_axes[1].set_title(r"Difference relative to $dt=0.05$")
    dipole_axes[1].grid(alpha=0.3)
    dipole_axes[1].legend()
    dipole_fig.tight_layout()
    dipole_path = results_directory / "stale_roatdccd_dipole_vs_dt.png"
    dipole_fig.savefig(dipole_path, dpi=180)
    plt.close(dipole_fig)

    method_fig, method_axes = plt.subplots(
        len(TIME_STEPS),
        2,
        figsize=(11.0, 9.0),
        sharex="col",
    )
    comparison_rows = []
    for row, (stale, tdrccsd) in enumerate(
        zip(stale_datasets, tdrccsd_datasets)
    ):
        np.testing.assert_allclose(stale["time"], tdrccsd["time"])
        difference = stale["dipole_z"] - tdrccsd["dipole_z"]
        max_difference = np.max(np.abs(difference))
        rms_difference = np.sqrt(np.mean(difference**2))
        comparison_rows.append((stale["dt"], max_difference, rms_difference))

        method_axes[row, 0].plot(
            stale["time"], tdrccsd["dipole_z"], label="TDRCCSD"
        )
        method_axes[row, 0].plot(
            stale["time"],
            stale["dipole_z"],
            "--",
            label="stale ROATDCCD",
        )
        method_axes[row, 0].set_ylabel(
            rf"$dt={stale['dt']:g}$" + "\n" + r"$\langle d_z\rangle$"
        )
        method_axes[row, 0].grid(alpha=0.3)

        method_axes[row, 1].plot(stale["time"], difference)
        method_axes[row, 1].axhline(0, color="black", lw=0.8, alpha=0.5)
        method_axes[row, 1].set_ylabel(
            r"$d_z^{\mathrm{ROATDCCD}}-d_z^{\mathrm{TDRCCSD}}$"
        )
        method_axes[row, 1].grid(alpha=0.3)

    method_axes[0, 0].set_title("Dipole trajectories")
    method_axes[0, 1].set_title("Method difference")
    method_axes[0, 0].legend()
    method_axes[-1, 0].set_xlabel("Time [a.u.]")
    method_axes[-1, 1].set_xlabel("Time [a.u.]")
    method_fig.suptitle("He/cc-pVDZ: TDRCCSD versus stale ROATDCCD dipole")
    method_fig.tight_layout()
    method_path = results_directory / "tdrccsd_vs_stale_roatdccd_dipole.png"
    method_fig.savefig(method_path, dpi=180)
    plt.close(method_fig)

    energy_table = np.column_stack(
        (
            time_steps,
            energy_ranges,
            [item["max_energy_deviation"] for item in stale_datasets],
        )
    )
    energy_table_path = (
        results_directory / "stale_roatdccd_energy_conservation_vs_dt.csv"
    )
    np.savetxt(
        energy_table_path,
        energy_table,
        delimiter=",",
        header="dt,post_pulse_energy_range,max_post_pulse_deviation",
        comments="",
    )
    comparison_table_path = (
        results_directory / "tdrccsd_vs_stale_roatdccd_dipole.csv"
    )
    np.savetxt(
        comparison_table_path,
        np.asarray(comparison_rows),
        delimiter=",",
        header="dt,max_abs_dipole_difference,rms_dipole_difference",
        comments="",
    )

    for item in stale_datasets:
        print(
            f"dt={item['dt']:.2f}: energy_range=" f"{item['energy_range']:.12e}"
        )
    for coarse, fine, order in zip(
        time_steps[:-1], time_steps[1:], pairwise_orders
    ):
        print(f"Energy order {coarse:g} -> {fine:g}: {order:.6f}")
    print(f"Three-point energy order: {fitted_order:.6f}")
    for dt, max_difference, rms_difference in comparison_rows:
        print(
            f"dt={dt:.2f}: max |dipole method difference|="
            f"{max_difference:.12e}, rms={rms_difference:.12e}"
        )
    print(f"Saved energy plot: {energy_path}")
    print(f"Saved dipole plot: {dipole_path}")
    print(f"Saved method comparison: {method_path}")


if __name__ == "__main__":
    main()
