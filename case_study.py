"""Inspect selected RF sweep points with longer trajectories.

Use this script after a coarse sweep to look more carefully at individual
voltage/frequency pairs.  It solves the normalized 1 V FEM base field once,
then reuses that base field for several RF cases.

This remains a simplified prototype.  A case that survives for 5 ms is not a
validated stable trap; it is only a useful candidate for longer simulations and
better physical modeling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import DEFAULT_CONFIG, SimulationConfig, with_config_overrides
from fem_solver import solve_laplace
from field_interpolation import make_E_at_position
from metrics import build_result_row
from particle_dynamics import ParticleResult, simulate_particle
from voltage_protocols import make_rf_case_config
from visualization import (
    plot_coordinates_vs_time,
    plot_speed_vs_time,
    plot_trajectory_3d,
)


# Selected RF cases to inspect.  Frequencies are written in Hz for readability;
# the dynamics code uses angular frequency in rad/s.
CASES = [
    {"rf_voltage": 10.0, "rf_frequency_hz": 5.0e3},
    {"rf_voltage": 10.0, "rf_frequency_hz": 1.5e4},
    {"rf_voltage": 10.0, "rf_frequency_hz": 3.0e4},
    {"rf_voltage": 20.0, "rf_frequency_hz": 2.0e4},
    {"rf_voltage": 20.0, "rf_frequency_hz": 5.0e4},
    {"rf_voltage": 40.0, "rf_frequency_hz": 1.0e5},
]

# First longer inspection time.  Increase this after the first pass to check
# whether "survived" cases are only temporary.
CASE_SIMULATION_TIME = (0.0, 5.0e-3)


def make_case_label(rf_voltage: float, rf_frequency_hz: float) -> str:
    """Return a filesystem-friendly label for one RF case."""

    voltage_text = f"{rf_voltage:g}".replace(".", "p")
    frequency_text = f"{rf_frequency_hz:g}".replace(".", "p")
    return f"rf_{voltage_text}V_{frequency_text}Hz"


def plot_radius_vs_time(particle_result: ParticleResult, ax=None):
    """Plot radial distance r(t) = sqrt(x^2 + y^2 + z^2)."""

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    radius = np.linalg.norm(particle_result.positions, axis=1)
    ax.plot(particle_result.t, radius)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("r [m]")
    ax.set_title("Radial distance vs time")
    ax.grid(True, alpha=0.3)
    return ax


def save_case_plots(
    particle_result: ParticleResult,
    rf_voltage: float,
    rf_frequency_hz: float,
    output_dir: str | Path,
):
    """Save trajectory, coordinate, radius, and speed plots for one case."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = make_case_label(rf_voltage, rf_frequency_hz)

    fig1 = plt.figure(figsize=(6, 5))
    ax1 = fig1.add_subplot(111, projection="3d")
    plot_trajectory_3d(particle_result, ax=ax1)
    ax1.set_title(f"Trajectory: {rf_voltage:g} V, {rf_frequency_hz:g} Hz")
    fig1.savefig(output_dir / f"{label}_trajectory_3d.png", dpi=180, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    plot_coordinates_vs_time(particle_result, ax=ax2)
    ax2.set_title(f"Coordinates: {rf_voltage:g} V, {rf_frequency_hz:g} Hz")
    fig2.savefig(output_dir / f"{label}_coordinates_vs_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    plot_radius_vs_time(particle_result, ax=ax3)
    ax3.set_title(f"Radius: {rf_voltage:g} V, {rf_frequency_hz:g} Hz")
    fig3.savefig(output_dir / f"{label}_radius_vs_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(7, 4))
    plot_speed_vs_time(particle_result, ax=ax4)
    ax4.set_title(f"Speed: {rf_voltage:g} V, {rf_frequency_hz:g} Hz")
    fig4.savefig(output_dir / f"{label}_speed_vs_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig4)


def run_case(
    E_base,
    base_config: SimulationConfig,
    rf_voltage: float,
    rf_frequency_hz: float,
    output_dir: str | Path,
) -> dict:
    """Run one selected RF case and save its diagnostic plots."""

    rf_angular_frequency = 2.0 * np.pi * rf_frequency_hz
    config = make_rf_case_config(
        base_config,
        rf_voltage=float(rf_voltage),
        rf_angular_frequency=float(rf_angular_frequency),
    )

    particle_result = simulate_particle(E_base, config)
    save_case_plots(particle_result, rf_voltage, rf_frequency_hz, output_dir)

    row = build_result_row(
        particle_result,
        config,
        rf_voltage,
        rf_angular_frequency,
        config.confinement_radius_threshold,
    )
    row["exit_time"] = particle_result.exit_time
    return row


def print_case_summary(row: dict):
    """Print the important diagnostics for one case."""

    print()
    print(f"Case: {row['rf_voltage']:g} V, {row['rf_frequency_hz']:g} Hz")
    print("-" * 44)
    print(f"status:       {row['status']}")
    print(f"final_time:   {row['final_time']:.3e} s")
    print(f"max_radius:   {row['max_radius']:.3e} m")
    print(f"final_radius: {row['final_radius']:.3e} m")
    print(f"final_speed:  {row['final_speed']:.3e} m/s")
    print(f"RF periods:   {row['simulated_rf_periods']:.1f}")

    if row["status"] == "escaped":
        if row["exit_time"] is None:
            print("exit_time:    not available")
        else:
            print(f"exit_time:    {row['exit_time']:.3e} s")


def main():
    """Run all selected case studies."""

    case_config = with_config_overrides(
        DEFAULT_CONFIG,
        mesh_cells=(5, 5, 5),
        simulation_time=CASE_SIMULATION_TIME,
        time_step=1.0e-5,
        max_time_step=1.0e-5,
    )
    output_dir = Path(case_config.output_dir) / "case_studies"

    print("Solving normalized 1 V FEM base field for case studies...")
    fem_result = solve_laplace(case_config)
    E_base = make_E_at_position(fem_result)

    for case in CASES:
        row = run_case(
            E_base,
            case_config,
            rf_voltage=case["rf_voltage"],
            rf_frequency_hz=case["rf_frequency_hz"],
            output_dir=output_dir,
        )
        print_case_summary(row)

    print(f"\nSaved case-study plots to: {output_dir}")


if __name__ == "__main__":
    main()
