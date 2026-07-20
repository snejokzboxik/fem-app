"""RF parameter sweep for the charged-particle trap prototype.

This is still a simplified educational model, not a validated physical model.
The FEM solve is done once for a normalized 1 V electrostatic boundary
condition.  Then each RF parameter pair uses

    E(t, r) = (dc_voltage + rf_voltage*cos(Omega*t)) * E_base(r)

which relies on the linearity of Laplace's equation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import DEFAULT_CONFIG, SimulationConfig, with_config_overrides
from fem_solver import solve_laplace
from field_interpolation import make_E_at_position
from mathieu_analysis import (
    compute_effective_mathieu_parameters_from_curvature,
    estimate_potential_curvature_near_center,
    plot_mathieu_stability_diagram,
)
from metrics import CSV_COLUMNS, build_result_row
from particle_dynamics import simulate_particle
from voltage_protocols import make_rf_case_config


# ---------------------------------------------------------------------------
# Easy-to-edit sweep settings
# ---------------------------------------------------------------------------

# The main RF sweep excludes 0 V so the map focuses on driven cases.  Add 0.0
# manually as a control case when you want a baseline trajectory without RF
# drive.  These placeholder ranges explore kHz to 100 kHz behavior.
RF_VOLTAGES = np.array([2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0])
RF_FREQUENCIES_HZ = np.array([5.0e3, 1.0e4, 1.5e4, 2.0e4, 3.0e4, 5.0e4, 1.0e5])

# Increase this when you want to test whether "survived" cases remain inside
# the domain for longer times.  Short simulations can make marginal cases look
# better than they really are.
SWEEP_SIMULATION_TIME = (0.0, 5.0e-3)

# Use the threshold from config.py by default.  You can override it here for a
# sweep without changing the main simulation settings.
CONFINEMENT_RADIUS_THRESHOLD = DEFAULT_CONFIG.confinement_radius_threshold

# Set this to True to automatically run a second, finer sweep around the first
# coarse transition region found in the survived/escaped map.
RUN_FINE_SWEEP_NEAR_TRANSITION = False
FINE_SWEEP_POINTS = (7, 7)  # (voltage points, frequency points)


def print_model_warning():
    """Print the main physical caveat for this sweep."""

    print(
        "Warning: this is still a simplified RF model with placeholder particle "
        "parameters and a placeholder electrode geometry. The sweep is only "
        "for exploring numerical behavior."
    )


def run_parameter_sweep(
    rf_voltages: np.ndarray,
    rf_angular_frequencies: np.ndarray,
    base_config: SimulationConfig = DEFAULT_CONFIG,
    simulation_time: tuple[float, float] | None = None,
    confinement_radius_threshold: float | None = None,
):
    """Run particle simulations over RF voltage and angular frequency.

    Returns
    -------
    tuple[list[dict], numpy.ndarray, numpy.ndarray]
        Results plus two maps with shape ``(n_frequencies, n_voltages)``:

        * survival_map: 1 for survived or confined, 0 for escaped
        * confinement_map: 1 for confined, 0 for not confined
    """

    rf_voltages = np.asarray(rf_voltages, dtype=float)
    rf_angular_frequencies = np.asarray(rf_angular_frequencies, dtype=float)

    if simulation_time is not None:
        base_config = with_config_overrides(base_config, simulation_time=simulation_time)

    if confinement_radius_threshold is None:
        confinement_radius_threshold = base_config.confinement_radius_threshold

    print_model_warning()
    print("Solving normalized 1 V FEM base field for the sweep...")
    fem_result = solve_laplace(base_config)
    E_base = make_E_at_position(fem_result)
    mathieu_curvature = None
    mathieu_warning_printed = False

    try:
        mathieu_curvature = estimate_potential_curvature_near_center(fem_result)
        print(
            "Estimated local FEM curvature for effective Mathieu parameters: "
            f"kx={mathieu_curvature['kx']:.3e}, "
            f"ky={mathieu_curvature['ky']:.3e}, "
            f"kz={mathieu_curvature['kz']:.3e} 1/m^2"
        )
    except Exception as exc:
        print(
            "Warning: could not estimate FEM curvature for Mathieu columns; "
            f"leaving them as NaN. Reason: {exc}"
        )

    results = []
    survival_map = np.zeros(
        (len(rf_angular_frequencies), len(rf_voltages)),
        dtype=int,
    )
    confinement_map = np.zeros_like(survival_map)

    for i_freq, angular_frequency in enumerate(rf_angular_frequencies):
        for j_volt, rf_voltage in enumerate(rf_voltages):
            config = make_rf_case_config(
                base_config,
                rf_voltage=float(rf_voltage),
                rf_angular_frequency=float(angular_frequency),
            )

            particle_result = simulate_particle(E_base, config)
            row = build_result_row(
                particle_result,
                config,
                float(rf_voltage),
                float(angular_frequency),
                confinement_radius_threshold,
            )
            if mathieu_curvature is not None:
                try:
                    row.update(
                        compute_effective_mathieu_parameters_from_curvature(
                            mathieu_curvature,
                            particle_charge=config.particle_charge,
                            particle_mass=config.particle_mass,
                            dc_voltage=config.dc_voltage,
                            rf_voltage=float(rf_voltage),
                            omega=float(angular_frequency),
                        )
                    )
                except Exception as exc:
                    if not mathieu_warning_printed:
                        print(
                            "Warning: could not compute one or more effective "
                            "Mathieu parameter rows; leaving failed rows as NaN. "
                            f"Reason: {exc}"
                        )
                        mathieu_warning_printed = True

            survival_map[i_freq, j_volt] = 0 if row["status"] == "escaped" else 1
            confinement_map[i_freq, j_volt] = 1 if row["status"] == "confined" else 0
            results.append(row)

    return results, survival_map, confinement_map


def save_results_csv(results: list[dict], output_path: str | Path):
    """Save sweep results with trajectory metrics to a CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)


def save_binary_map(
    map_values: np.ndarray,
    rf_voltages: np.ndarray,
    rf_angular_frequencies: np.ndarray,
    output_path: str | Path,
    title: str,
    zero_label: str,
    one_label: str,
):
    """Save one clearly labeled binary map."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_binary_map(
        map_values,
        rf_voltages,
        rf_angular_frequencies,
        title=title,
        zero_label=zero_label,
        one_label=one_label,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_binary_map(
    map_values: np.ndarray,
    rf_voltages: np.ndarray,
    rf_angular_frequencies: np.ndarray,
    title: str,
    zero_label: str,
    one_label: str,
):
    """Return a matplotlib figure for one clearly labeled binary map."""

    frequencies_hz = rf_angular_frequencies / (2.0 * np.pi)

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(
        map_values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            rf_voltages[0],
            rf_voltages[-1],
            frequencies_hz[0],
            frequencies_hz[-1],
        ],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("RF voltage amplitude [V]")
    ax.set_ylabel("RF frequency [Hz]")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1])
    colorbar.ax.set_yticklabels([zero_label, one_label])
    return fig


def save_sweep_maps(
    survival_map: np.ndarray,
    confinement_map: np.ndarray,
    rf_voltages: np.ndarray,
    rf_angular_frequencies: np.ndarray,
    output_dir: str | Path,
    filename_prefix: str = "rf",
):
    """Save survived/escaped and confined/not-confined maps."""

    output_dir = Path(output_dir)

    # This keeps the original map concept: escaped vs not escaped.
    save_binary_map(
        survival_map,
        rf_voltages,
        rf_angular_frequencies,
        output_dir / f"{filename_prefix}_stability_map.png",
        title="RF sweep survival map",
        zero_label="escaped",
        one_label="survived",
    )

    save_binary_map(
        confinement_map,
        rf_voltages,
        rf_angular_frequencies,
        output_dir / f"{filename_prefix}_confinement_map.png",
        title="RF sweep confinement map",
        zero_label="not confined",
        one_label="confined",
    )


def save_stability_map(
    stability_map: np.ndarray,
    rf_voltages: np.ndarray,
    rf_angular_frequencies: np.ndarray,
    output_path: str | Path,
):
    """Backward-compatible helper for the original escaped/survived map."""

    save_binary_map(
        stability_map,
        rf_voltages,
        rf_angular_frequencies,
        output_path,
        title="RF sweep survival map",
        zero_label="escaped",
        one_label="survived",
    )


def find_transition_region(
    rf_voltages: np.ndarray,
    rf_angular_frequencies: np.ndarray,
    survival_map: np.ndarray,
):
    """Find a coarse bounding box where survived/escaped values change.

    This is only a convenience helper for planning a finer sweep.  It looks for
    neighboring cells with different survived/escaped values and returns the
    voltage and angular-frequency ranges around those changes.
    """

    transition_voltages = []
    transition_frequencies = []

    for i_freq in range(survival_map.shape[0]):
        for j_volt in range(survival_map.shape[1] - 1):
            if survival_map[i_freq, j_volt] != survival_map[i_freq, j_volt + 1]:
                transition_voltages.extend(
                    [rf_voltages[j_volt], rf_voltages[j_volt + 1]]
                )
                transition_frequencies.append(rf_angular_frequencies[i_freq])

    for i_freq in range(survival_map.shape[0] - 1):
        for j_volt in range(survival_map.shape[1]):
            if survival_map[i_freq, j_volt] != survival_map[i_freq + 1, j_volt]:
                transition_voltages.append(rf_voltages[j_volt])
                transition_frequencies.extend(
                    [rf_angular_frequencies[i_freq], rf_angular_frequencies[i_freq + 1]]
                )

    if not transition_voltages or not transition_frequencies:
        return None

    return (
        float(np.min(transition_voltages)),
        float(np.max(transition_voltages)),
        float(np.min(transition_frequencies)),
        float(np.max(transition_frequencies)),
    )


def make_refined_arrays_near_transition(
    rf_voltages: np.ndarray,
    rf_angular_frequencies: np.ndarray,
    survival_map: np.ndarray,
    voltage_points: int = 7,
    frequency_points: int = 7,
):
    """Create finer arrays around the first coarse transition region."""

    region = find_transition_region(rf_voltages, rf_angular_frequencies, survival_map)
    if region is None:
        return None, None

    v_min, v_max, w_min, w_max = region

    if np.isclose(v_min, v_max):
        voltage_step = (
            np.min(np.abs(np.diff(rf_voltages))) if len(rf_voltages) > 1 else 1.0
        )
        v_min -= 0.5 * voltage_step
        v_max += 0.5 * voltage_step

    if np.isclose(w_min, w_max):
        freq_step = (
            np.min(np.abs(np.diff(rf_angular_frequencies)))
            if len(rf_angular_frequencies) > 1
            else 2.0 * np.pi * 100.0
        )
        w_min -= 0.5 * freq_step
        w_max += 0.5 * freq_step

    refined_voltages = np.linspace(v_min, v_max, voltage_points)
    refined_angular_frequencies = np.linspace(w_min, w_max, frequency_points)
    return refined_voltages, refined_angular_frequencies


def print_results_table(results: list[dict]):
    """Print voltage, frequency, status, final time, and radius metrics."""

    print()
    print(
        "RF voltage [V] | RF frequency [Hz] | status   | "
        "final time [s] | periods | max r [m] | final r [m]"
    )
    print("-" * 104)
    for item in results:
        print(
            f"{item['rf_voltage']:14.3g} | "
            f"{item['rf_frequency_hz']:17.3g} | "
            f"{item['status']:<8} | "
            f"{item['final_time']:.3e}     | "
            f"{item['simulated_rf_periods']:7.1f} | "
            f"{item['max_radius']:.3e} | "
            f"{item['final_radius']:.3e}"
        )


def print_summary(results: list[dict], top_n: int = 5):
    """Print compact sweep counts and the best confined cases."""

    escaped_count = sum(item["status"] == "escaped" for item in results)
    confined_count = sum(item["status"] == "confined" for item in results)
    survived_only_count = sum(item["status"] == "survived" for item in results)
    survived_total = survived_only_count + confined_count

    print()
    print("Sweep summary")
    print("-" * 40)
    print(f"Escaped cases: {escaped_count}")
    print(f"Survived cases including confined: {survived_total}")
    print(f"Survived but not confined: {survived_only_count}")
    print(f"Confined cases: {confined_count}")

    if survived_only_count:
        print(
            "Survived-but-not-confined cases are temporary-survival candidates; "
            "rerun them with a longer simulation time."
        )

    confined = [item for item in results if item["status"] == "confined"]
    confined.sort(key=lambda item: item["max_radius"])

    if not confined:
        print("No confined cases found in this sweep.")
        return

    print()
    print(f"Best confined cases by smallest max_radius, top {min(top_n, len(confined))}")
    print("RF voltage [V] | RF frequency [Hz] | max r [m] | final speed [m/s]")
    print("-" * 76)
    for item in confined[:top_n]:
        print(
            f"{item['rf_voltage']:14.3g} | "
            f"{item['rf_frequency_hz']:17.3g} | "
            f"{item['max_radius']:.3e} | "
            f"{item['final_speed']:.3e}"
        )


def run_named_sweep(
    name: str,
    rf_voltages: np.ndarray,
    rf_angular_frequencies: np.ndarray,
    base_config: SimulationConfig,
    output_dir: Path,
):
    """Run one sweep, save CSV/maps, and print a summary."""

    results, survival_map, confinement_map = run_parameter_sweep(
        rf_voltages=rf_voltages,
        rf_angular_frequencies=rf_angular_frequencies,
        base_config=base_config,
        simulation_time=base_config.simulation_time,
        confinement_radius_threshold=base_config.confinement_radius_threshold,
    )

    save_results_csv(results, output_dir / f"{name}_sweep_results.csv")
    save_sweep_maps(
        survival_map,
        confinement_map,
        rf_voltages,
        rf_angular_frequencies,
        output_dir,
        filename_prefix=name,
    )
    mathieu_fig, _mathieu_ax = plot_mathieu_stability_diagram(points=results)
    mathieu_fig.savefig(
        output_dir / f"{name}_mathieu_stability_diagram.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(mathieu_fig)
    print_results_table(results)
    print_summary(results)

    return results, survival_map, confinement_map


def main():
    """Run a coarse sweep and optionally a finer transition sweep."""

    coarse_config = with_config_overrides(
        DEFAULT_CONFIG,
        mesh_cells=(5, 5, 5),
        simulation_time=SWEEP_SIMULATION_TIME,
        time_step=5.0e-6,
        max_time_step=5.0e-6,
        confinement_radius_threshold=CONFINEMENT_RADIUS_THRESHOLD,
    )

    rf_voltages = np.asarray(RF_VOLTAGES, dtype=float)
    rf_angular_frequencies = 2.0 * np.pi * np.asarray(RF_FREQUENCIES_HZ, dtype=float)
    output_dir = Path(coarse_config.output_dir)

    _results, survival_map, _confinement_map = run_named_sweep(
        name="rf",
        rf_voltages=rf_voltages,
        rf_angular_frequencies=rf_angular_frequencies,
        base_config=coarse_config,
        output_dir=output_dir,
    )

    print(f"\nSaved CSV to: {output_dir / 'rf_sweep_results.csv'}")
    print(f"Saved survival map to: {output_dir / 'rf_stability_map.png'}")
    print(f"Saved confinement map to: {output_dir / 'rf_confinement_map.png'}")
    print(
        "Saved Mathieu stability diagram to: "
        f"{output_dir / 'rf_mathieu_stability_diagram.png'}"
    )

    if not RUN_FINE_SWEEP_NEAR_TRANSITION:
        return

    refined_voltages, refined_angular_frequencies = make_refined_arrays_near_transition(
        rf_voltages,
        rf_angular_frequencies,
        survival_map,
        voltage_points=FINE_SWEEP_POINTS[0],
        frequency_points=FINE_SWEEP_POINTS[1],
    )

    if refined_voltages is None:
        print("\nNo coarse transition found; skipping fine sweep.")
        return

    print("\nRunning finer sweep near the coarse transition region...")
    run_named_sweep(
        name="rf_fine",
        rf_voltages=refined_voltages,
        rf_angular_frequencies=refined_angular_frequencies,
        base_config=coarse_config,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
