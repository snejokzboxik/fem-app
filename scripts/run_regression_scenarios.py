"""Headless regression scenarios for the Streamlit trap prototype.

The runner intentionally does not start Streamlit and does not use browser
automation.  It exercises the same lower-level helpers that the UI relies on:
surface geometry sources, Poisson-kernel field construction, field import,
interpolation, and a very short particle-dynamics pass.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import pi
from pathlib import Path
import sys
import time
import traceback
from typing import Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG, SimulationConfig, with_config_overrides
from experiment_config import validate_experiment_config
from experiment_presets import built_in_experiment_presets
from field_data import FieldGrid, load_field_grid_npz
from field_interpolation import make_E_at_position_from_field_grid
from metrics import compute_trajectory_metrics
from particle_dynamics import simulate_particle
from surface_geometry_sources import (
    ROLE_DC,
    ROLE_GND,
    ROLE_RF,
    ElectrodeAssignment,
    ElectrodeRegionDefinition,
    TwoChannelElectricField,
    build_voltage_maps_from_assignments,
    canvas_design_from_dict,
    canvas_design_to_voltage_maps,
    label_canvas_electrodes,
    rasterize_function_regions,
)
from surface_superposition import (
    SurfaceMaskConfig,
    compute_surface_field_grid,
    detect_electrode_components,
    load_binary_electrode_mask,
)


SCENARIO_NAMES = (
    "function_rectangles_rf",
    "image_mask_four_rectangles",
    "canvas_sample_design",
    "imported_demo_field",
    "rf_dc_separated_surface",
    "quick_start_preset",
)


def scenario_names() -> tuple[str, ...]:
    """Return the stable list of available regression scenarios."""

    return SCENARIO_NAMES


def _json_safe(value):
    """Convert NumPy values into plain JSON-compatible Python values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def serialize_result(result: dict) -> dict:
    """Return one scenario result as a JSON-safe dictionary."""

    return _json_safe(result)


def write_summary(
    results: list[dict],
    output_dir: str | Path = PROJECT_ROOT / "results" / "regression",
) -> dict:
    """Write JSON and CSV regression summaries and return the summary object."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_results = [serialize_result(result) for result in results]
    failures = [result for result in safe_results if result["status"] == "FAIL"]
    summary = {
        "status": "failed" if failures else "ok",
        "scenario_count": len(safe_results),
        "passed": sum(1 for result in safe_results if result["status"] == "PASS"),
        "failed": len(failures),
        "skipped": sum(1 for result in safe_results if result["status"] == "SKIP"),
        "results": safe_results,
    }

    json_path = output_path / "regression_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output_path / "regression_summary.csv"
    fieldnames = [
        "name",
        "status",
        "duration_s",
        "field_shape",
        "max_abs_field",
        "time_points",
        "max_radius_m",
        "final_speed",
        "left_domain",
        "notes",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in safe_results:
            metrics = result.get("metrics", {})
            row = {
                "name": result.get("name", ""),
                "status": result.get("status", ""),
                "duration_s": result.get("duration_s", ""),
                "field_shape": metrics.get("field_shape", ""),
                "max_abs_field": metrics.get("max_abs_field", ""),
                "time_points": metrics.get("time_points", ""),
                "max_radius_m": metrics.get("max_radius_m", ""),
                "final_speed": metrics.get("final_speed", ""),
                "left_domain": metrics.get("left_domain", ""),
                "notes": result.get("notes", ""),
                "error": result.get("error", ""),
            }
            writer.writerow(row)

    summary["json_path"] = str(json_path)
    summary["csv_path"] = str(csv_path)
    return summary


def _make_result(
    name: str,
    status: str,
    start_time: float,
    *,
    metrics: dict | None = None,
    notes: str = "",
    error: str = "",
) -> dict:
    """Create one compact scenario-result dictionary."""

    return {
        "name": name,
        "status": status,
        "duration_s": round(time.perf_counter() - start_time, 6),
        "metrics": metrics or {},
        "notes": notes,
        "error": error,
    }


def _assert_finite_array(label: str, values) -> None:
    """Fail the scenario if an array contains NaN or infinity."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"{label} is empty.")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{label} contains non-finite values.")


def _surface_config(
    *,
    x_size_m: float = 1.0e-3,
    y_size_m: float = 1.0e-3,
    z_max_m: float = 4.0e-4,
    max_mask_size: int = 64,
    source_description: str,
) -> SurfaceMaskConfig:
    """Small Poisson-kernel grid used by fast regression scenarios."""

    return SurfaceMaskConfig(
        x_size_m=x_size_m,
        y_size_m=y_size_m,
        z_max_m=z_max_m,
        min_z_m=2.0e-5,
        nx=7,
        ny=7,
        nz=5,
        grid_mode_xy="uniform",
        grid_mode_z="uniform",
        edge_refinement_enabled=False,
        downsample_mask_for_computation=True,
        max_computational_mask_size=max_mask_size,
        max_active_pixels_for_direct_sum=10000,
        chunk_size_points=128,
        active_pixel_chunk_size=256,
        source_description=source_description,
    )


def _short_dynamics_config(
    *,
    domain_size: tuple[float, float, float] = (1.0e-3, 1.0e-3, 8.0e-4),
    initial_position: tuple[float, float, float] = (0.0, 0.0, 1.2e-4),
    use_time_dependent_voltage: bool = False,
) -> SimulationConfig:
    """Return conservative fast dynamics settings for smoke-level integration."""

    return with_config_overrides(
        DEFAULT_CONFIG,
        domain_size=domain_size,
        particle_mass=1.0e-15,
        particle_charge=1.0e-21,
        damping_coefficient=0.0,
        initial_position=initial_position,
        initial_velocity=(0.0, 0.0, 0.0),
        voltage_amplitude=1.0,
        use_time_dependent_voltage=use_time_dependent_voltage,
        dc_voltage=0.0,
        rf_voltage=1.0,
        rf_angular_frequency=2.0 * pi * 3.0e4,
        simulation_time=(0.0, 2.0e-5),
        time_step=2.0e-6,
        max_time_step=2.0e-6,
        ode_rtol=1.0e-6,
        ode_atol=1.0e-12,
        confinement_radius_threshold=3.5e-4,
        output_dir="results/regression",
    )


def _field_metrics(field_grid: FieldGrid) -> dict:
    """Validate and summarize a computed or imported field grid."""

    _assert_finite_array("electric_field_grid", field_grid.electric_field_grid)
    metrics = {
        "field_shape": list(np.asarray(field_grid.electric_field_grid).shape),
        "max_abs_field": float(np.max(np.abs(field_grid.electric_field_grid))),
        "x_points": int(len(field_grid.x_grid)),
        "y_points": int(len(field_grid.y_grid)),
        "z_points": int(len(field_grid.z_grid)),
    }
    if field_grid.potential_grid is not None:
        _assert_finite_array("potential_grid", field_grid.potential_grid)
        metrics["potential_shape"] = list(np.asarray(field_grid.potential_grid).shape)
    return metrics


def _dynamics_metrics(E_function: Callable, config) -> dict:
    """Run a short particle simulation and return trajectory metrics."""

    result = simulate_particle(E_function, config)
    _assert_finite_array("trajectory positions", result.positions)
    _assert_finite_array("trajectory velocities", result.velocities)
    _assert_finite_array("trajectory speed", result.speed)
    trajectory_metrics = compute_trajectory_metrics(result)
    return {
        "time_points": int(len(result.t)),
        "left_domain": bool(result.left_domain),
        "exit_time": result.exit_time,
        **trajectory_metrics,
    }


def _surface_metrics(
    voltage_map: np.ndarray,
    mask: np.ndarray,
    config: SurfaceMaskConfig,
    *,
    use_time_dependent_voltage: bool = False,
    field_function: Callable | None = None,
) -> dict:
    """Build a surface field and run a short dynamics pass."""

    field_grid = compute_surface_field_grid(voltage_map, config, mask=mask)
    metrics = _field_metrics(field_grid)
    E_function = field_function or make_E_at_position_from_field_grid(field_grid)
    sample_z = min(1.2e-4, 0.75 * config.z_max_m)
    sample_z = max(sample_z, 2.0 * config.min_z_m)
    sample_field = (
        E_function(0.0, (0.0, 0.0, sample_z))
        if getattr(E_function, "expects_time", False)
        else E_function((0.0, 0.0, sample_z))
    )
    _assert_finite_array("sample electric field", sample_field)
    dynamics_config = _short_dynamics_config(
        domain_size=(config.x_size_m, config.y_size_m, 2.0 * config.z_max_m),
        initial_position=(0.0, 0.0, sample_z),
        use_time_dependent_voltage=use_time_dependent_voltage,
    )
    metrics.update(_dynamics_metrics(E_function, dynamics_config))
    metrics["active_pixels"] = int(np.count_nonzero(voltage_map))
    return metrics


def _ensure_qa_assets() -> Path:
    """Generate QA assets if the checked-in files are absent."""

    qa_dir = PROJECT_ROOT / "examples" / "qa_assets"
    required = [
        qa_dir / "mask_four_rectangles.png",
        qa_dir / "sample_canvas_design.json",
    ]
    if not all(path.exists() for path in required):
        from examples.qa_assets.generate_qa_assets import generate_qa_assets

        generate_qa_assets(qa_dir)
    return qa_dir


def _ensure_example_field() -> Path:
    """Generate the tiny imported-field demo if it is absent."""

    field_path = PROJECT_ROOT / "examples" / "example_field_grid.npz"
    if not field_path.exists():
        from examples.generate_example_field_npz import main as generate_example_field

        generate_example_field()
    return field_path


def _run_function_definitions(
    name: str,
    definitions: list[ElectrodeRegionDefinition],
    *,
    x_size_m: float = 1.0e-3,
    y_size_m: float = 1.0e-3,
    mask_size: int = 64,
) -> dict:
    """Shared path for function-defined surface-electrode scenarios."""

    geometry = rasterize_function_regions(
        definitions,
        x_size_m=x_size_m,
        y_size_m=y_size_m,
        nx_mask=mask_size,
        ny_mask=mask_size,
    )
    rf_map, dc_map = build_voltage_maps_from_assignments(
        geometry.region_labels,
        geometry.assignments,
    )
    voltage_map = rf_map + dc_map
    if not np.any(voltage_map):
        raise ValueError(f"{name}: no active electrode pixels after rasterization.")
    config = _surface_config(
        x_size_m=x_size_m,
        y_size_m=y_size_m,
        max_mask_size=mask_size,
        source_description=name,
    )
    return _surface_metrics(voltage_map, voltage_map != 0.0, config)


def scenario_function_rectangles_rf() -> dict:
    """Four function-defined RF rectangles."""

    definitions = [
        ElectrodeRegionDefinition(
            "RF left",
            "abs(x + 250e-6) < 70e-6 and abs(y) < 280e-6",
            ROLE_RF,
        ),
        ElectrodeRegionDefinition(
            "RF right",
            "abs(x - 250e-6) < 70e-6 and abs(y) < 280e-6",
            ROLE_RF,
        ),
        ElectrodeRegionDefinition(
            "RF top",
            "abs(y - 250e-6) < 70e-6 and abs(x) < 280e-6",
            ROLE_RF,
        ),
        ElectrodeRegionDefinition(
            "RF bottom",
            "abs(y + 250e-6) < 70e-6 and abs(x) < 280e-6",
            ROLE_RF,
        ),
    ]
    return _run_function_definitions("function_rectangles_rf", definitions)


def scenario_image_mask_four_rectangles() -> dict:
    """Existing QA PNG mask with four separated rectangular electrodes."""

    qa_dir = _ensure_qa_assets()
    mask = load_binary_electrode_mask(qa_dir / "mask_four_rectangles.png")
    labels, number_of_components = detect_electrode_components(mask)
    if number_of_components < 4:
        raise ValueError(
            "mask_four_rectangles.png should contain at least four components."
        )
    assignments = [
        ElectrodeAssignment(
            region_id=region_id,
            name=f"image electrode {region_id}",
            role=ROLE_RF if region_id <= 2 else ROLE_GND,
        )
        for region_id in range(1, number_of_components + 1)
    ]
    rf_map, dc_map = build_voltage_maps_from_assignments(labels, assignments)
    voltage_map = rf_map + dc_map
    config = _surface_config(
        max_mask_size=64,
        source_description="image_mask_four_rectangles",
    )
    metrics = _surface_metrics(voltage_map, mask, config)
    metrics["components"] = int(number_of_components)
    return metrics


def scenario_canvas_sample_design() -> dict:
    """Existing QA canvas JSON, decoded without Streamlit."""

    qa_dir = _ensure_qa_assets()
    data = json.loads(
        (qa_dir / "sample_canvas_design.json").read_text(encoding="utf-8")
    )
    design = canvas_design_from_dict(data)
    labels, number_of_components, cleaned_mask = label_canvas_electrodes(
        design["binary_mask"],
        min_component_area=1,
    )
    if number_of_components < 3:
        raise ValueError("sample_canvas_design.json has too few components.")
    rf_map, dc_map = canvas_design_to_voltage_maps(labels, design["assignments"])
    voltage_map = rf_map + dc_map
    config = _surface_config(
        x_size_m=design["x_size_m"],
        y_size_m=design["y_size_m"],
        max_mask_size=64,
        source_description="canvas_sample_design",
    )
    metrics = _surface_metrics(voltage_map, cleaned_mask, config)
    metrics["components"] = int(number_of_components)
    return metrics


def scenario_imported_demo_field() -> dict:
    """Load the documented tiny FieldGrid example and run interpolation/dynamics."""

    field_grid = load_field_grid_npz(_ensure_example_field())
    metrics = _field_metrics(field_grid)
    E_function = make_E_at_position_from_field_grid(field_grid)
    _assert_finite_array("imported sample field", E_function((0.0, 0.0, 0.0)))
    domain_size = (
        float(np.ptp(field_grid.x_grid)),
        float(np.ptp(field_grid.y_grid)),
        float(np.ptp(field_grid.z_grid)),
    )
    dynamics_config = _short_dynamics_config(
        domain_size=domain_size,
        initial_position=(0.0, 0.0, 0.0),
    )
    metrics.update(_dynamics_metrics(E_function, dynamics_config))
    return metrics


def scenario_rf_dc_separated_surface() -> dict:
    """Separate RF and DC maps, then combine them through TwoChannelElectricField."""

    labels = np.zeros((64, 64), dtype=int)
    labels[18:46, 8:16] = 1
    labels[18:46, 48:56] = 2
    labels[8:16, 24:40] = 3
    labels[48:56, 24:40] = 4
    assignments = [
        ElectrodeAssignment(1, "RF left", ROLE_RF),
        ElectrodeAssignment(2, "RF right", ROLE_RF),
        ElectrodeAssignment(3, "DC top", ROLE_DC, 0.25),
        ElectrodeAssignment(4, "GND bottom", ROLE_GND),
    ]
    rf_map, dc_map = build_voltage_maps_from_assignments(labels, assignments)
    mask = labels > 0
    config = _surface_config(
        max_mask_size=64,
        source_description="rf_dc_separated_surface",
    )
    rf_field_grid = compute_surface_field_grid(rf_map, config, mask=mask)
    dc_field_grid = compute_surface_field_grid(dc_map, config, mask=mask)
    metrics = {
        "rf": _field_metrics(rf_field_grid),
        "dc": _field_metrics(dc_field_grid),
    }
    E_function = TwoChannelElectricField(
        make_E_at_position_from_field_grid(rf_field_grid),
        make_E_at_position_from_field_grid(dc_field_grid),
        rf_amplitude=1.0,
        rf_angular_frequency=2.0 * pi * 3.0e4,
    )
    combined_metrics = _surface_metrics(
        rf_map + dc_map,
        mask,
        config,
        use_time_dependent_voltage=True,
        field_function=E_function,
    )
    metrics.update(combined_metrics)
    metrics["active_rf_pixels"] = int(np.count_nonzero(rf_map))
    metrics["active_dc_pixels"] = int(np.count_nonzero(dc_map))
    return metrics


def scenario_quick_start_preset() -> dict:
    """Run the built-in quick-start preset through the headless function path."""

    presets = built_in_experiment_presets()
    title = next((candidate for candidate in presets if "4 RF" in candidate), None)
    if title is None:
        raise KeyError("Quick-start preset with '4 RF' in title was not found.")
    preset = presets[title]
    report = validate_experiment_config(preset)
    if not report["valid"]:
        raise ValueError("; ".join(report["errors"]))
    geometry = preset["geometry_config"]
    definitions = [
        ElectrodeRegionDefinition(
            item["name"],
            item["expression"],
            item.get("role", ROLE_GND),
            float(item.get("voltage", 0.0)),
            float(item.get("rf_phase", 0.0)),
        )
        for item in geometry.get("function_definitions", [])
    ]
    if not definitions:
        raise ValueError("Quick-start preset does not contain function definitions.")
    mask_size = min(64, int(geometry.get("max_computational_mask_size", 64)))
    metrics = _run_function_definitions(
        "quick_start_preset",
        definitions,
        x_size_m=float(geometry.get("x_size_m", 1.0e-3)),
        y_size_m=float(geometry.get("y_size_m", 1.0e-3)),
        mask_size=mask_size,
    )
    metrics["preset_title"] = title
    metrics["validation_warnings"] = report.get("warnings", [])
    return metrics


SCENARIOS: dict[str, Callable[[], dict]] = {
    "function_rectangles_rf": scenario_function_rectangles_rf,
    "image_mask_four_rectangles": scenario_image_mask_four_rectangles,
    "canvas_sample_design": scenario_canvas_sample_design,
    "imported_demo_field": scenario_imported_demo_field,
    "rf_dc_separated_surface": scenario_rf_dc_separated_surface,
    "quick_start_preset": scenario_quick_start_preset,
}


def run_scenarios(names: tuple[str, ...] = SCENARIO_NAMES) -> list[dict]:
    """Run scenarios by name and return PASS/FAIL dictionaries."""

    results = []
    for name in names:
        start_time = time.perf_counter()
        print(f"Сценарий: {name} ...", flush=True)
        try:
            metrics = SCENARIOS[name]()
        except Exception as exc:
            print(f"  ОШИБКА: {exc}", flush=True)
            results.append(
                _make_result(
                    name,
                    "FAIL",
                    start_time,
                    error=f"{type(exc).__name__}: {exc}",
                    notes=traceback.format_exc(limit=5),
                )
            )
        else:
            print("  OK", flush=True)
            results.append(_make_result(name, "PASS", start_time, metrics=metrics))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Headless regression scenarios for charged_particle_trap.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available scenario names and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = build_arg_parser().parse_args(argv)
    if args.list:
        print("Доступные headless regression scenarios:")
        for name in scenario_names():
            print(f"- {name}")
        return 0

    print("Запуск headless regression scenarios.")
    print("Streamlit и браузер не запускаются.")
    results = run_scenarios()
    summary = write_summary(results)

    print(f"Итог: {summary['status']}")
    print(f"PASS: {summary['passed']}, FAIL: {summary['failed']}, SKIP: {summary['skipped']}")
    print(f"JSON: {summary['json_path']}")
    print(f"CSV: {summary['csv_path']}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
