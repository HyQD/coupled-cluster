#!/usr/bin/env python
"""Analyze corrected ROATDCCD and compare it with TDRCCSD and stale data."""

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


def load_method(results_directory, method):
    return [
        load_dataset(
            results_directory,
            method + "_he_ccpvdz_delta_rk4_dt",
            dt,
        )
        for dt in TIME_STEPS
    ]


def main():
    results_directory = Path(__file__).parent / "results"
    corrected_datasets = load_method(results_directory, "corrected_roatdccd")
    stale_datasets = load_method(results_directory, "stale_roatdccd")
    tdrccsd_datasets = load_method(results_directory, "tdrccsd")

    for item in corrected_datasets:
        post_pulse_energy = item["energy"][1:]
        item["energy_range"] = np.ptp(post_pulse_energy)
        item["max_energy_deviation"] = np.max(
            np.abs(post_pulse_energy - post_pulse_energy[0])
        )

    time_steps = np.array(TIME_STEPS)
    energy_ranges = np.array(
        [item["energy_range"] for item in corrected_datasets]
    )
    fitted_order, log_prefactor = np.polyfit(
        np.log(time_steps), np.log(energy_ranges), 1
    )
    pairwise_orders = np.log(energy_ranges[:-1] / energy_ranges[1:]) / np.log(
        time_steps[:-1] / time_steps[1:]
    )

    energy_fig, energy_axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for item in corrected_datasets:
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
        "He/cc-pVDZ corrected ROATDCCD with length-gauge delta pulse and RK4"
    )
    energy_fig.tight_layout()
    energy_path = (
        results_directory / "corrected_roatdccd_energy_conservation_vs_dt.png"
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
    for item in corrected_datasets:
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
        "He/cc-pVDZ corrected ROATDCCD dipole after length-gauge delta pulse"
    )
    dipole_axes[0].grid(alpha=0.3)
    dipole_axes[0].legend()

    reference = corrected_datasets[-1]
    for item in corrected_datasets[:-1]:
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
    dipole_path = results_directory / "corrected_roatdccd_dipole_vs_dt.png"
    dipole_fig.savefig(dipole_path, dpi=180)
    plt.close(dipole_fig)

    corrected_fig, corrected_axes = plt.subplots(
        len(TIME_STEPS),
        2,
        figsize=(11.0, 9.0),
        sharex="col",
    )
    for row, (corrected, tdrccsd) in enumerate(
        zip(corrected_datasets, tdrccsd_datasets)
    ):
        np.testing.assert_allclose(corrected["time"], tdrccsd["time"])
        difference = corrected["dipole_z"] - tdrccsd["dipole_z"]
        corrected_axes[row, 0].plot(
            corrected["time"], tdrccsd["dipole_z"], label="TDRCCSD"
        )
        corrected_axes[row, 0].plot(
            corrected["time"],
            corrected["dipole_z"],
            "--",
            label="corrected ROATDCCD",
        )
        corrected_axes[row, 0].set_ylabel(
            rf"$dt={corrected['dt']:g}$" + "\n" + r"$\langle d_z\rangle$"
        )
        corrected_axes[row, 0].grid(alpha=0.3)
        corrected_axes[row, 1].plot(corrected["time"], difference)
        corrected_axes[row, 1].axhline(0, color="black", lw=0.8, alpha=0.5)
        corrected_axes[row, 1].set_ylabel(
            r"$d_z^{\mathrm{ROATDCCD}}-d_z^{\mathrm{TDRCCSD}}$"
        )
        corrected_axes[row, 1].grid(alpha=0.3)
    corrected_axes[0, 0].set_title("Dipole trajectories")
    corrected_axes[0, 1].set_title("Method difference")
    corrected_axes[0, 0].legend()
    corrected_axes[-1, 0].set_xlabel("Time [a.u.]")
    corrected_axes[-1, 1].set_xlabel("Time [a.u.]")
    corrected_fig.suptitle("He/cc-pVDZ: TDRCCSD versus corrected ROATDCCD")
    corrected_fig.tight_layout()
    corrected_path = (
        results_directory / "tdrccsd_vs_corrected_roatdccd_dipole.png"
    )
    corrected_fig.savefig(corrected_path, dpi=180)
    plt.close(corrected_fig)

    comparison_fig, comparison_axes = plt.subplots(
        len(TIME_STEPS),
        2,
        figsize=(11.0, 9.0),
        sharex="col",
    )
    comparison_rows = []
    stale_rows = []
    for row, (corrected, stale, tdrccsd) in enumerate(
        zip(corrected_datasets, stale_datasets, tdrccsd_datasets)
    ):
        np.testing.assert_allclose(corrected["time"], tdrccsd["time"])
        np.testing.assert_allclose(stale["time"], tdrccsd["time"])
        corrected_difference = corrected["dipole_z"] - tdrccsd["dipole_z"]
        stale_difference = stale["dipole_z"] - tdrccsd["dipole_z"]
        corrected_max = np.max(np.abs(corrected_difference))
        corrected_rms = np.sqrt(np.mean(corrected_difference**2))
        stale_max = np.max(np.abs(stale_difference))
        stale_rms = np.sqrt(np.mean(stale_difference**2))
        comparison_rows.append((corrected["dt"], corrected_max, corrected_rms))
        stale_rows.append((corrected["dt"], stale_max, stale_rms))

        comparison_axes[row, 0].plot(
            corrected["time"], tdrccsd["dipole_z"], label="TDRCCSD"
        )
        comparison_axes[row, 0].plot(
            corrected["time"],
            corrected["dipole_z"],
            "--",
            label="corrected ROATDCCD",
        )
        comparison_axes[row, 0].plot(
            stale["time"],
            stale["dipole_z"],
            ":",
            label="stale ROATDCCD",
        )
        comparison_axes[row, 0].set_ylabel(
            rf"$dt={corrected['dt']:g}$" + "\n" + r"$\langle d_z\rangle$"
        )
        comparison_axes[row, 0].grid(alpha=0.3)

        comparison_axes[row, 1].plot(
            corrected["time"],
            corrected_difference,
            label="corrected minus TDRCCSD",
        )
        comparison_axes[row, 1].plot(
            stale["time"],
            stale_difference,
            ":",
            label="stale minus TDRCCSD",
        )
        comparison_axes[row, 1].axhline(0, color="black", lw=0.8, alpha=0.5)
        comparison_axes[row, 1].set_ylabel(r"$\Delta d_z$")
        comparison_axes[row, 1].grid(alpha=0.3)

    comparison_axes[0, 0].set_title("Dipole trajectories")
    comparison_axes[0, 1].set_title("Difference from TDRCCSD")
    comparison_axes[0, 0].legend()
    comparison_axes[0, 1].legend()
    comparison_axes[-1, 0].set_xlabel("Time [a.u.]")
    comparison_axes[-1, 1].set_xlabel("Time [a.u.]")
    comparison_fig.suptitle(
        "He/cc-pVDZ: TDRCCSD versus corrected and stale ROATDCCD"
    )
    comparison_fig.tight_layout()
    comparison_path = (
        results_directory / "tdrccsd_vs_roatdccd_dipole_three_way.png"
    )
    comparison_fig.savefig(comparison_path, dpi=180)
    plt.close(comparison_fig)

    energy_table = np.column_stack(
        (
            time_steps,
            energy_ranges,
            [item["max_energy_deviation"] for item in corrected_datasets],
        )
    )
    energy_table_path = (
        results_directory / "corrected_roatdccd_energy_conservation_vs_dt.csv"
    )
    np.savetxt(
        energy_table_path,
        energy_table,
        delimiter=",",
        header="dt,post_pulse_energy_range,max_post_pulse_deviation",
        comments="",
    )
    comparison_table_path = (
        results_directory / "tdrccsd_vs_corrected_roatdccd_dipole.csv"
    )
    np.savetxt(
        comparison_table_path,
        np.asarray(comparison_rows),
        delimiter=",",
        header="dt,max_abs_dipole_difference,rms_dipole_difference",
        comments="",
    )

    for item in corrected_datasets:
        print(
            f"dt={item['dt']:.2f}: energy_range=" f"{item['energy_range']:.12e}"
        )
    for coarse, fine, order in zip(
        time_steps[:-1], time_steps[1:], pairwise_orders
    ):
        print(f"Energy order {coarse:g} -> {fine:g}: {order:.6f}")
    print(f"Three-point energy order: {fitted_order:.6f}")
    for corrected_row, stale_row in zip(comparison_rows, stale_rows):
        dt, corrected_max, corrected_rms = corrected_row
        _, stale_max, stale_rms = stale_row
        print(
            f"dt={dt:.2f}: corrected/TDR max={corrected_max:.12e}, "
            f"rms={corrected_rms:.12e}; stale/TDR max={stale_max:.12e}, "
            f"rms={stale_rms:.12e}"
        )
    print(f"Saved energy plot: {energy_path}")
    print(f"Saved dipole plot: {dipole_path}")
    print(f"Saved corrected comparison: {corrected_path}")
    print(f"Saved three-way comparison: {comparison_path}")


if __name__ == "__main__":
    main()
