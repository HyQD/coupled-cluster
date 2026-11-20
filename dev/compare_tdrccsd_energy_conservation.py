#!/usr/bin/env python
"""Compare TDRCCSD post-pulse energy conservation across RK4 time steps."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIME_STEPS = (0.2, 0.1, 0.05)


def timestep_label(dt):
    return f"{dt:g}".replace(".", "p")


def main():
    results_directory = Path(__file__).parent / "results"
    datasets = []

    for dt in TIME_STEPS:
        path = results_directory / (
            "tdrccsd_he_ccpvdz_delta_rk4_dt" + timestep_label(dt) + ".npz"
        )
        data = np.load(path)
        time = data["time_points"]
        energy = data["energy"].real
        post_pulse_energy = energy[1:]
        energy_range = np.ptp(post_pulse_energy)
        max_deviation = np.max(np.abs(post_pulse_energy - post_pulse_energy[0]))
        datasets.append(
            {
                "dt": dt,
                "time": time,
                "energy": energy,
                "range": energy_range,
                "max_deviation": max_deviation,
            }
        )

    time_steps = np.array([item["dt"] for item in datasets])
    energy_ranges = np.array([item["range"] for item in datasets])
    max_deviations = np.array([item["max_deviation"] for item in datasets])
    fitted_order, log_prefactor = np.polyfit(
        np.log(time_steps), np.log(energy_ranges), 1
    )
    pairwise_orders = np.log(energy_ranges[:-1] / energy_ranges[1:]) / np.log(
        time_steps[:-1] / time_steps[1:]
    )

    table = np.column_stack((time_steps, energy_ranges, max_deviations))
    table_path = results_directory / "tdrccsd_energy_conservation_vs_dt.csv"
    np.savetxt(
        table_path,
        table,
        delimiter=",",
        header="dt,post_pulse_energy_range,max_post_pulse_deviation",
        comments="",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for item in datasets:
        post_time = item["time"][1:]
        post_energy = item["energy"][1:]
        axes[0].plot(
            post_time,
            post_energy - post_energy[0],
            label=rf"$dt={item['dt']:g}$",
        )
    axes[0].set_xlabel("Time [a.u.]")
    axes[0].set_ylabel(r"$E(t)-E(dt)$ [a.u.]")
    axes[0].set_title("Post-pulse energy drift")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    order_grid = np.linspace(time_steps.min(), time_steps.max(), 200)
    fitted_curve = np.exp(log_prefactor) * order_grid**fitted_order
    axes[1].loglog(
        time_steps,
        energy_ranges,
        "o",
        ms=7,
        label="Measured range",
    )
    axes[1].loglog(
        order_grid,
        fitted_curve,
        "--",
        label=rf"Fit: $dt^{{{fitted_order:.2f}}}$",
    )
    axes[1].set_xlabel(r"$dt$ [a.u.]")
    axes[1].set_ylabel("Post-pulse energy range [a.u.]")
    axes[1].set_title("Energy-conservation convergence")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()

    fig.suptitle("He/cc-pVDZ TDRCCSD with length-gauge delta pulse and RK4")
    fig.tight_layout()
    figure_path = results_directory / "tdrccsd_energy_conservation_vs_dt.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    for item in datasets:
        print(
            f"dt={item['dt']:.2f}: range={item['range']:.12e}, "
            f"max_deviation={item['max_deviation']:.12e}"
        )
    for coarse, fine, order in zip(
        time_steps[:-1], time_steps[1:], pairwise_orders
    ):
        print(f"Observed order {coarse:g} -> {fine:g}: {order:.6f}")
    print(f"Three-point fitted order: {fitted_order:.6f}")
    print(f"Saved table: {table_path}")
    print(f"Saved plot: {figure_path}")


if __name__ == "__main__":
    main()
