#!/usr/bin/env python
"""Analyze the He/cc-pVDZ three-stage Gauss propagation results."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIME_STEPS = (0.2, 0.1, 0.05)
METHODS = ("tdrccsd", "stale_roatdccd", "corrected_roatdccd")
METHOD_LABELS = {
    "tdrccsd": "TDRCCSD",
    "stale_roatdccd": "stale ROATDCCD",
    "corrected_roatdccd": "corrected ROATDCCD",
}
GAUSS_STAGES = 3
GAUSS_TOLERANCE_LABEL = "1e-10"


def timestep_label(dt):
    return f"{dt:g}".replace(".", "p")


def load_dataset(results_directory, method, dt):
    stem = (
        f"{method}_he_ccpvdz_delta_gauss_s{GAUSS_STAGES}_"
        f"eps{GAUSS_TOLERANCE_LABEL}_dt{timestep_label(dt)}"
    )
    data = np.load(results_directory / f"{stem}.npz")
    post_energy = data["energy"].real[1:]
    return {
        "method": method,
        "dt": dt,
        "time": data["time_points"],
        "energy": data["energy"].real,
        "dipole_z": data["dipole_z"].real,
        "effective_kick": float(data["effective_gauss_kick"]),
        "energy_range": np.ptp(post_energy),
        "max_energy_deviation": np.max(np.abs(post_energy - post_energy[0])),
        "rhs_evaluations": int(data["rhs_evaluations"]),
    }


def main():
    results_directory = Path(__file__).parent / "results"
    datasets = {
        method: [
            load_dataset(results_directory, method, dt) for dt in TIME_STEPS
        ]
        for method in METHODS
    }
    time_steps = np.asarray(TIME_STEPS)

    energy_fig, energy_axes = plt.subplots(
        len(METHODS), 2, figsize=(11.0, 11.0)
    )
    for row, method in enumerate(METHODS):
        method_data = datasets[method]
        for item in method_data:
            post_time = item["time"][1:]
            post_energy = item["energy"][1:]
            absolute_drift = np.abs(post_energy - post_energy[0])
            nonzero = absolute_drift > 0
            energy_axes[row, 0].semilogy(
                post_time[nonzero],
                absolute_drift[nonzero],
                label=rf"$dt={item['dt']:g}$",
            )
        energy_ranges = [item["energy_range"] for item in method_data]
        energy_axes[row, 1].loglog(time_steps, energy_ranges, "o-", ms=6)
        energy_axes[row, 0].set_ylabel(
            METHOD_LABELS[method] + "\n" + r"$|E(t)-E(t_s)|$ [a.u.]"
        )
        energy_axes[row, 1].set_ylabel("Energy range [a.u.]")
        energy_axes[row, 0].grid(alpha=0.3, which="both")
        energy_axes[row, 1].grid(alpha=0.3, which="both")
        energy_axes[row, 0].legend()
    energy_axes[0, 0].set_title("Post-pulse energy drift")
    energy_axes[0, 1].set_title("Energy range versus time step")
    energy_axes[-1, 0].set_xlabel("Time [a.u.]")
    energy_axes[-1, 1].set_xlabel(r"$dt$ [a.u.]")
    energy_fig.suptitle(
        r"He/cc-pVDZ, length-gauge delta pulse, Gauss $s=3$, "
        r"$\epsilon=10^{-10}$"
    )
    energy_fig.tight_layout()
    energy_path = (
        results_directory / "gauss_s3_eps1e-10_energy_conservation.png"
    )
    energy_fig.savefig(energy_path, dpi=180)
    plt.close(energy_fig)

    dipole_fig, dipole_axes = plt.subplots(
        len(METHODS), 2, figsize=(11.0, 11.0), sharex="col"
    )
    for row, method in enumerate(METHODS):
        method_data = datasets[method]
        reference = method_data[-1]
        for item in method_data:
            dipole_axes[row, 0].plot(
                item["time"],
                item["dipole_z"],
                marker="o",
                ms=2.2,
                markevery=max(1, round(0.2 / item["dt"])),
                label=rf"$dt={item['dt']:g}$",
            )
        for item in method_data[:-1]:
            reference_at_time = np.interp(
                item["time"], reference["time"], reference["dipole_z"]
            )
            dipole_axes[row, 1].plot(
                item["time"],
                item["dipole_z"] - reference_at_time,
                marker="o",
                ms=2.2,
                label=rf"$dt={item['dt']:g}$",
            )
        dipole_axes[row, 0].set_ylabel(
            METHOD_LABELS[method] + "\n" + r"$\langle d_z\rangle$"
        )
        dipole_axes[row, 1].set_ylabel(r"$d_z(dt)-d_z(dt=0.05)$")
        dipole_axes[row, 1].axhline(0, color="black", lw=0.8, alpha=0.5)
        dipole_axes[row, 0].grid(alpha=0.3)
        dipole_axes[row, 1].grid(alpha=0.3)
        dipole_axes[row, 0].legend()
        dipole_axes[row, 1].legend()
    dipole_axes[0, 0].set_title("Dipole trajectories")
    dipole_axes[0, 1].set_title(r"Difference relative to $dt=0.05$")
    dipole_axes[-1, 0].set_xlabel("Time [a.u.]")
    dipole_axes[-1, 1].set_xlabel("Time [a.u.]")
    dipole_fig.suptitle(r"He/cc-pVDZ dipole, Gauss $s=3$, $\epsilon=10^{-10}$")
    dipole_fig.tight_layout()
    dipole_path = results_directory / "gauss_s3_eps1e-10_dipole_vs_dt.png"
    dipole_fig.savefig(dipole_path, dpi=180)
    plt.close(dipole_fig)

    method_fig, method_axes = plt.subplots(
        len(TIME_STEPS), 2, figsize=(11.0, 9.0), sharex="col"
    )
    comparison_rows = []
    for row, dt in enumerate(TIME_STEPS):
        tdrccsd = datasets["tdrccsd"][row]
        stale = datasets["stale_roatdccd"][row]
        corrected = datasets["corrected_roatdccd"][row]
        np.testing.assert_allclose(tdrccsd["time"], stale["time"])
        np.testing.assert_allclose(tdrccsd["time"], corrected["time"])
        stale_difference = stale["dipole_z"] - tdrccsd["dipole_z"]
        corrected_difference = corrected["dipole_z"] - tdrccsd["dipole_z"]
        cache_difference = stale["dipole_z"] - corrected["dipole_z"]
        comparison_rows.append(
            (
                dt,
                np.max(np.abs(stale_difference)),
                np.sqrt(np.mean(stale_difference**2)),
                np.max(np.abs(corrected_difference)),
                np.sqrt(np.mean(corrected_difference**2)),
                np.max(np.abs(cache_difference)),
            )
        )

        method_axes[row, 0].plot(
            tdrccsd["time"], tdrccsd["dipole_z"], label="TDRCCSD"
        )
        method_axes[row, 0].plot(
            stale["time"], stale["dipole_z"], ":", label="stale ROATDCCD"
        )
        method_axes[row, 0].plot(
            corrected["time"],
            corrected["dipole_z"],
            "--",
            label="corrected ROATDCCD",
        )
        method_axes[row, 0].set_ylabel(
            rf"$dt={dt:g}$" + "\n" + r"$\langle d_z\rangle$"
        )
        method_axes[row, 0].grid(alpha=0.3)
        method_axes[row, 1].plot(
            stale["time"], stale_difference, ":", label="stale minus TDRCCSD"
        )
        method_axes[row, 1].plot(
            corrected["time"],
            corrected_difference,
            "--",
            label="corrected minus TDRCCSD",
        )
        method_axes[row, 1].axhline(0, color="black", lw=0.8, alpha=0.5)
        method_axes[row, 1].set_ylabel(r"$\Delta d_z$")
        method_axes[row, 1].grid(alpha=0.3)
    method_axes[0, 0].set_title("Dipole trajectories")
    method_axes[0, 1].set_title("Difference from TDRCCSD")
    method_axes[0, 0].legend()
    method_axes[0, 1].legend()
    method_axes[-1, 0].set_xlabel("Time [a.u.]")
    method_axes[-1, 1].set_xlabel("Time [a.u.]")
    method_fig.suptitle(
        r"He/cc-pVDZ method comparison, Gauss $s=3$, $\epsilon=10^{-10}$"
    )
    method_fig.tight_layout()
    method_path = results_directory / "gauss_s3_eps1e-10_method_dipoles.png"
    method_fig.savefig(method_path, dpi=180)
    plt.close(method_fig)

    energy_rows = []
    for method in METHODS:
        for item in datasets[method]:
            energy_rows.append(
                (
                    METHODS.index(method),
                    item["dt"],
                    item["energy_range"],
                    item["max_energy_deviation"],
                    item["rhs_evaluations"],
                )
            )
    energy_table_path = results_directory / "gauss_s3_eps1e-10_energy.csv"
    np.savetxt(
        energy_table_path,
        np.asarray(energy_rows),
        delimiter=",",
        header=(
            "method_index (0=tdrccsd;1=stale;2=corrected),dt,"
            "post_pulse_energy_range,max_post_pulse_deviation,rhs_evaluations"
        ),
        comments="",
    )
    comparison_table_path = results_directory / "gauss_s3_eps1e-10_dipoles.csv"
    np.savetxt(
        comparison_table_path,
        np.asarray(comparison_rows),
        delimiter=",",
        header=(
            "dt,stale_tdr_max,stale_tdr_rms,corrected_tdr_max,"
            "corrected_tdr_rms,stale_corrected_max"
        ),
        comments="",
    )

    for method in METHODS:
        print(METHOD_LABELS[method])
        for item in datasets[method]:
            print(
                f"  dt={item['dt']:.2f}: energy_range="
                f"{item['energy_range']:.12e}, rhs={item['rhs_evaluations']}"
            )
    for row in comparison_rows:
        dt, stale_max, stale_rms, corrected_max, corrected_rms, cache_max = row
        print(
            f"dt={dt:.2f}: stale/TDR max={stale_max:.12e}, "
            f"rms={stale_rms:.12e}; corrected/TDR max="
            f"{corrected_max:.12e}, rms={corrected_rms:.12e}; "
            f"stale/corrected max={cache_max:.12e}"
        )
    print(f"Saved energy plot: {energy_path}")
    print(f"Saved dipole plot: {dipole_path}")
    print(f"Saved method comparison: {method_path}")


if __name__ == "__main__":
    main()
