"""Basic smoke tests for the educational prototype."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import (
    DEFAULT_CONFIG,
    GeometryConfig,
    MeshConfig,
    ParticleConfig,
    SimulationConfig,
    SolverConfig,
    VoltageConfig,
    with_config_overrides,
)
from drag_models import (
    EnvironmentConfig,
    classify_drag_regime,
    compute_damping_gamma,
    cunningham_slip_correction,
    epstein_drag_gamma,
    gas_mean_free_path,
    knudsen_number,
    particle_mass_from_radius_density,
    stokes_cunningham_drag_gamma,
    stokes_drag_gamma,
)
from field_data import (
    FieldGrid,
    compute_field_from_potential_grid,
    load_field_grid_npz,
    load_potential_grid_npz,
    save_fem_field_to_npz,
)
from fem_solver import solve_laplace
from field_interpolation import make_E_at_position, make_E_at_position_from_field_grid
from field_validation import (
    compare_field_grids,
    compare_potential_grids,
    estimate_symmetry_checks,
    validate_field_grid as validate_field_grid_report,
)
from experiment_config import (
    ExperimentConfig,
    apply_experiment_config_to_session_state,
    collect_current_experiment_config_from_ui,
    experiment_config_from_dict,
    load_experiment_config_from_json,
    save_experiment_config_to_json,
    validate_experiment_config,
)
from experiment_presets import built_in_experiment_presets, validate_all_presets
from mathieu_analysis import (
    compute_ideal_mathieu_parameters,
    estimate_potential_curvature_near_center,
    frequency_hz_to_omega,
    omega_to_frequency_hz,
    plot_mathieu_stability_diagram,
)
from metrics import (
    CSV_COLUMNS,
    build_result_row,
    classify_localization_status,
    compute_trajectory_metrics,
    localization_status_label,
)
from particle_dynamics import ParticleResult, simulate_particle, voltage_scale
from report_export import (
    build_markdown_report,
    export_experiment_zip,
)
from surface_superposition import (
    SurfaceMaskConfig,
    build_voltage_map_from_components,
    compute_field_from_potential_grid_nonuniform,
    compute_potential_from_surface_voltage_map,
    compute_surface_field_grid,
    detect_electrode_components,
    detect_electrode_edges,
    downsample_binary_mask_for_computation,
    load_binary_electrode_mask,
    make_edge_aware_xy_grid,
    make_mask_pixel_coordinates,
    make_rectilinear_grid_1d,
    make_surface_observation_grid,
    make_z_grid,
    prepare_surface_voltage_map_for_computation,
)
from surface_geometry_sources import (
    ROLE_CUSTOM,
    ROLE_DC,
    ROLE_GND,
    ROLE_RF,
    ElectrodeAssignment,
    ElectrodeRegionDefinition,
    TwoChannelElectricField,
    build_voltage_maps_from_assignments,
    canvas_design_from_dict,
    canvas_design_to_dict,
    canvas_design_to_voltage_maps,
    canvas_image_to_binary_mask,
    clean_binary_mask,
    compute_pseudopotential_from_rf_field_grid,
    evaluate_region_expression_safe,
    label_canvas_electrodes,
    rasterize_function_regions,
)
from sweep import run_parameter_sweep
from voltage_protocols import (
    make_rf_case_config,
    rf_period_from_frequency,
    rf_resolved_time_step,
)
from case_study import CASES, make_case_label
from app import (
    DEMO_PRESETS,
    compute_field_grid_diagnostics,
    dashboard_mode_to_action,
    mathieu_parameters_outside_standard_view,
    parse_comma_separated_floats,
    preset_value,
    run_parameter_search_with_field,
    status_badge_html,
)
from examples.qa_assets.generate_qa_assets import ASSET_FILENAMES, generate_qa_assets
from scripts.smoke_check import smoke_check


def make_synthetic_field_grid(offset: float = 0.0) -> FieldGrid:
    """Create a small deterministic FieldGrid for validation tests."""

    x_grid = np.linspace(-1.0, 1.0, 4)
    y_grid = np.linspace(-1.0, 1.0, 4)
    z_grid = np.linspace(-1.0, 1.0, 4)
    x_mesh, y_mesh, z_mesh = np.meshgrid(
        x_grid,
        y_grid,
        z_grid,
        indexing="ij",
    )
    potential_grid = x_mesh**2 - y_mesh**2 + 0.1 * z_mesh**2
    electric_field_grid = compute_field_from_potential_grid(
        potential_grid,
        x_grid,
        y_grid,
        z_grid,
    )
    electric_field_grid = electric_field_grid + offset
    return FieldGrid(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        electric_field_grid=electric_field_grid,
        potential_grid=potential_grid,
    )


def make_test_surface_mask() -> np.ndarray:
    """Return a small mask with two separated rectangular electrodes."""

    mask = np.zeros((12, 12), dtype=bool)
    mask[3:8, 2:5] = True
    mask[3:8, 7:10] = True
    return mask


def make_parabolic_test_mask() -> np.ndarray:
    """Return a small curved mask to exercise arbitrary image geometry."""

    yy, xx = np.mgrid[0:24, 0:24]
    x = (xx - 12) / 12
    y = (12 - yy) / 12
    return (y > 0.15 + 0.45 * x**2) & (np.abs(x) < 0.85)


def make_high_resolution_surface_mask() -> np.ndarray:
    """Return a large mask whose computational version should be downsampled."""

    mask = np.zeros((256, 512), dtype=bool)
    mask[64:192, 64:448] = True
    return mask


def make_synthetic_canvas_image() -> np.ndarray:
    """Return a white RGBA canvas with two dark electrode strokes and noise."""

    image = np.full((32, 32, 4), 255, dtype=np.uint8)
    image[5:13, 5:13, :3] = 0
    image[18:28, 20:29, :3] = 0
    image[1, 1, :3] = 0
    return image


def test_safe_expression_accepts_rectangle_circle_ring_and_parabola():
    x = np.linspace(-5.0e-4, 5.0e-4, 60)
    y = np.linspace(-5.0e-4, 5.0e-4, 60)
    x_grid, y_grid = np.meshgrid(x, y)

    expressions = [
        "abs(x) < 200e-6 and abs(y - 250e-6) < 80e-6",
        "(x**2 + y**2) < (250e-6)**2",
        "((x**2 + y**2) > (150e-6)**2) and ((x**2 + y**2) < (250e-6)**2)",
        "y > 0.02*(x/1e-4)**2 + 50e-6 and y < 0.02*(x/1e-4)**2 + 250e-6",
    ]

    for expression in expressions:
        mask = evaluate_region_expression_safe(expression, x_grid, y_grid)
        assert mask.dtype == bool
        assert mask.shape == x_grid.shape
        assert np.any(mask)


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os").system("echo unsafe")',
        'open("file")',
        "eval('1 + 1')",
        "x.__class__",
    ],
)
def test_safe_expression_rejects_unsafe_input(expression):
    x = np.zeros((3, 3))
    y = np.zeros((3, 3))

    with pytest.raises(ValueError):
        evaluate_region_expression_safe(expression, x, y)


def test_function_region_rasterization_and_overlap_are_deterministic():
    definitions = [
        ElectrodeRegionDefinition(
            "first",
            "abs(x) < 150e-6 and abs(y) < 150e-6",
            ROLE_RF,
        ),
        ElectrodeRegionDefinition(
            "second",
            "abs(x) < 80e-6 and abs(y) < 80e-6",
            ROLE_DC,
            2.0,
        ),
    ]

    raster = rasterize_function_regions(
        definitions,
        x_size_m=1.0e-3,
        y_size_m=1.0e-3,
        nx_mask=41,
        ny_mask=41,
    )

    assert np.any(raster.region_labels > 0)
    assert raster.overlap_pixels > 0
    center_label = raster.region_labels[20, 20]
    assert center_label == 2


def test_electrode_assignments_create_rf_dc_ground_and_custom_maps():
    labels = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ]
    )
    assignments = [
        ElectrodeAssignment(1, "rf", ROLE_RF),
        ElectrodeAssignment(2, "dc", ROLE_DC, 1.5),
        ElectrodeAssignment(3, "ground", ROLE_GND),
        ElectrodeAssignment(4, "custom", ROLE_CUSTOM, -2.0),
    ]

    rf_map, dc_map = build_voltage_maps_from_assignments(labels, assignments)

    assert np.all(rf_map[labels == 1] == 1.0)
    assert np.all(dc_map[labels == 2] == 1.5)
    assert np.all(dc_map[labels == 3] == 0.0)
    assert np.all(dc_map[labels == 4] == -2.0)
    assert np.all(rf_map[labels != 1] == 0.0)


def test_two_channel_field_combines_rf_and_dc_with_time():
    def E_rf(_position):
        return np.array([1.0, 0.0, 0.0])

    def E_dc(_position):
        return np.array([0.0, 2.0, 0.0])

    field = TwoChannelElectricField(
        E_rf,
        E_dc,
        rf_amplitude=10.0,
        rf_angular_frequency=0.0,
    )

    assert field.expects_time is True
    assert np.allclose(field(0.0, np.zeros(3)), [10.0, 2.0, 0.0])


def test_pseudopotential_helper_uses_rf_field_only():
    field_grid = make_synthetic_field_grid()
    field_grid.electric_field_grid = np.ones_like(field_grid.electric_field_grid)
    pseudo = compute_pseudopotential_from_rf_field_grid(
        field_grid,
        particle_charge=2.0,
        particle_mass=4.0,
        rf_voltage=3.0,
        rf_angular_frequency=5.0,
    )

    expected = (2.0**2 * (3.0 * np.sqrt(3.0)) ** 2) / (4.0 * 4.0 * 5.0**2)
    assert pseudo.shape == field_grid.electric_field_grid.shape[:-1]
    assert np.allclose(pseudo, expected)


def test_canvas_image_to_binary_mask_detects_dark_drawn_regions():
    image = make_synthetic_canvas_image()

    mask = canvas_image_to_binary_mask(image)

    assert mask.dtype == bool
    assert mask.shape == image.shape[:2]
    assert mask[6, 6]
    assert mask[20, 22]
    assert not mask[0, 0]


def test_canvas_cleaning_removes_tiny_noise_and_labels_components():
    image = make_synthetic_canvas_image()
    mask = canvas_image_to_binary_mask(image)

    cleaned = clean_binary_mask(mask, min_component_area=4)
    labels, number_of_components, cleaned_from_label = label_canvas_electrodes(
        mask,
        min_component_area=4,
    )

    assert not cleaned[1, 1]
    assert number_of_components == 2
    assert np.array_equal(cleaned, cleaned_from_label)
    assert set(np.unique(labels)) == {0, 1, 2}


def test_canvas_assignment_maps_have_correct_shapes_and_roles():
    image = make_synthetic_canvas_image()
    mask = clean_binary_mask(canvas_image_to_binary_mask(image), min_component_area=4)
    labels, number_of_components, _cleaned = label_canvas_electrodes(mask, min_component_area=4)
    assignments = [
        ElectrodeAssignment(1, "rf", ROLE_RF),
        ElectrodeAssignment(2, "custom", ROLE_CUSTOM, -1.25),
    ]

    rf_map, dc_map = canvas_design_to_voltage_maps(labels, assignments)

    assert number_of_components == 2
    assert rf_map.shape == labels.shape
    assert dc_map.shape == labels.shape
    assert np.all(rf_map[labels == 1] == 1.0)
    assert np.all(dc_map[labels == 2] == -1.25)
    assert np.all(dc_map[labels == 0] == 0.0)


def test_canvas_design_json_roundtrip_preserves_mask_and_assignments():
    image = make_synthetic_canvas_image()
    mask = clean_binary_mask(canvas_image_to_binary_mask(image), min_component_area=4)
    assignments = [
        ElectrodeAssignment(1, "rf rail", ROLE_RF),
        ElectrodeAssignment(2, "dc pad", ROLE_DC, 0.75),
    ]

    design = canvas_design_to_dict(
        x_size_m=1.0e-3,
        y_size_m=2.0e-3,
        canvas_resolution_px=32,
        binary_mask=mask,
        assignments=assignments,
        rf_amplitude=10.0,
        rf_frequency_hz=30.0e3,
        notes="unit test",
    )
    loaded = canvas_design_from_dict(design)

    assert np.array_equal(loaded["binary_mask"], mask)
    assert loaded["x_size_m"] == 1.0e-3
    assert loaded["y_size_m"] == 2.0e-3
    assert loaded["canvas_resolution_px"] == 32
    assert loaded["assignments"][0].role == ROLE_RF
    assert loaded["assignments"][1].voltage == 0.75


def test_experiment_presets_are_valid_json_serializable():
    presets = built_in_experiment_presets()
    reports = validate_all_presets()

    assert "Быстрый старт: 4 RF электрода" in presets
    for title, preset in presets.items():
        assert reports[title]["valid"], reports[title]["errors"]
        json_text = __import__("json").dumps(preset, ensure_ascii=False)
        assert title in json_text


def test_rf_dc_preset_contains_rf_and_dc_like_roles():
    preset = built_in_experiment_presets()["RF + DC compensation"]
    roles = {assignment["role"] for assignment in preset["electrode_assignments"]}

    assert ROLE_RF in roles
    assert ROLE_DC in roles or ROLE_CUSTOM in roles
    assert ROLE_GND in roles


def test_experiment_config_save_load_roundtrip(tmp_path):
    config = ExperimentConfig(
        title="Roundtrip",
        workflow_mode="Проверить локализацию",
        geometry_source="Электроды функциями",
        rf_config={"rf_voltage": 10.0, "rf_frequency_hz": 30000.0},
    )
    path = tmp_path / "experiment_config.json"

    save_experiment_config_to_json(config, path)
    loaded = load_experiment_config_from_json(path)

    assert loaded.title == "Roundtrip"
    assert loaded.rf_config["rf_voltage"] == 10.0


def test_experiment_config_embeds_canvas_design_snapshot():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 6:10] = True
    surface_config = SurfaceMaskConfig(x_size_m=1.0e-3, y_size_m=1.0e-3)
    assignment = ElectrodeAssignment(1, "RF rail", ROLE_RF)
    experiment = collect_current_experiment_config_from_ui(
        config=DEFAULT_CONFIG,
        geometry_source="Нарисовать электроды",
        surface_options={
            "source_kind": "canvas",
            "config": surface_config,
            "number_of_components": 1,
            "uses_rf_dc_separation": True,
            "canvas_binary_mask": mask,
            "assignments": [assignment],
        },
    )

    geometry = experiment.to_dict()["geometry_config"]
    loaded_design = canvas_design_from_dict(geometry["canvas_design"])

    assert loaded_design["binary_mask"].shape == mask.shape
    assert loaded_design["assignments"][0].role == ROLE_RF


def test_experiment_config_uses_internal_workflow_keys_for_streamlit_state():
    class FakeStreamlitState:
        def __init__(self):
            self._data = {}

        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            self._data[key] = value

        def get(self, key, default=None):
            return self._data.get(key, default)

        def __contains__(self, key):
            return key in self._data

    state = FakeStreamlitState()
    experiment = ExperimentConfig(
        title="Workflow apply",
        workflow_mode="Проверить локализацию",
        geometry_source="Электроды функциями",
    )

    report = apply_experiment_config_to_session_state(experiment, state)

    assert report["valid"]
    assert state["app_workflow_mode"] == "Проверить локализацию"
    assert state["pending_workflow_mode"] == "Проверить локализацию"
    assert "workflow_mode" not in state


def test_invalid_experiment_config_reports_readable_error():
    report = validate_experiment_config({"geometry_config": []})

    assert not report["valid"]
    assert any("version" in error for error in report["errors"])
    assert any("geometry_config" in error for error in report["errors"])


def test_markdown_report_contains_key_sections():
    preset = experiment_config_from_dict(
        built_in_experiment_presets()["Быстрый старт: 4 RF электрода"]
    )
    report = build_markdown_report(
        experiment_config=preset,
        metrics={"status": "localized_like", "max_radius": 1.0e-4},
        field_diagnostics={"min_field_position": (0.0, 0.0, 1.0e-4)},
        environment_report={"gamma_kg_s": 0.0},
    )

    assert "# Быстрый старт: 4 RF электрода" in report
    assert "## Electrode Roles" in report
    assert "## Results" in report
    assert "Pseudopotential" in report or "pseudopotential" in report


def test_experiment_zip_contains_expected_files(tmp_path):
    preset = experiment_config_from_dict(
        built_in_experiment_presets()["Быстрый старт: 4 RF электрода"]
    )
    report = build_markdown_report(experiment_config=preset)
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text("status,max_radius\nok,1e-4\n", encoding="utf-8")
    zip_path = export_experiment_zip(
        tmp_path / "experiment.zip",
        experiment_config=preset,
        report_md=report,
        report_html="<html></html>",
        extra_files={"metrics.csv": metrics_path},
    )

    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "experiment_config.json" in names
    assert "report.md" in names
    assert "report.html" in names
    assert "metrics.csv" in names
    assert "README_EXPERIMENT.txt" in names


def test_qa_documentation_files_exist_and_are_russian_facing():
    project_root = Path(__file__).resolve().parents[1]
    qa_files = [
        "docs/qa/README_FOR_TESTER.md",
        "docs/qa/QUICK_START_FOR_TESTER.md",
        "docs/qa/TEST_PLAN_FULL.md",
        "docs/qa/BUG_REPORT_TEMPLATE.md",
        "docs/qa/TEST_REPORT_TEMPLATE.md",
        "docs/qa/KNOWN_LIMITATIONS.md",
        "examples/qa_assets/README_TEST_ASSETS.md",
    ]

    for relative_path in qa_files:
        text = (project_root / relative_path).read_text(encoding="utf-8")
        assert any("\u0400" <= character <= "\u04ff" for character in text)


def test_windows_launcher_files_exist_and_are_documented():
    project_root = Path(__file__).resolve().parents[1]
    launcher_files = [
        "launcher.py",
        "run_app_windows.bat",
        "run_app_windows.ps1",
        "scripts/build_windows_launcher.ps1",
        "docs/LAUNCH_WINDOWS.md",
    ]

    for relative_path in launcher_files:
        path = project_root / relative_path
        assert path.exists(), relative_path
        assert path.stat().st_size > 0

    (project_root / "run_app_windows.bat").read_text(encoding="ascii")
    launch_docs = (project_root / "docs/LAUNCH_WINDOWS.md").read_text(encoding="utf-8")
    assert "run_app_windows.bat" in launch_docs
    assert "ChargedTrapLauncher.exe" in launch_docs
    assert "launcher" in launch_docs
    assert any("\u0400" <= character <= "\u04ff" for character in launch_docs)


def test_manual_qa_checklist_has_many_cases():
    project_root = Path(__file__).resolve().parents[1]
    checklist_path = project_root / "docs/qa/manual_test_checklist.csv"

    with checklist_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) >= 40
    assert {"id", "priority", "area", "scenario", "steps", "expected_result"}.issubset(
        rows[0]
    )
    assert any(row["area"] == "Электроды функциями" for row in rows)
    assert any("Unsafe" in row["scenario"] for row in rows)


def test_qa_asset_generator_creates_expected_files(tmp_path):
    generated_paths = generate_qa_assets(tmp_path)

    generated_names = {path.name for path in generated_paths}
    assert set(ASSET_FILENAMES).issubset(generated_names)
    for filename in ASSET_FILENAMES:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0

    image = Image.open(tmp_path / "mask_four_rectangles.png")
    assert image.size == (512, 512)


def test_smoke_check_runs_without_heavy_simulation():
    report = smoke_check(generate_missing_assets=False)

    assert report["status"] == "ok"
    assert report["qa_docs_checked"] >= 7
    assert report["qa_assets_checked"] >= 7
    assert report["launcher_files_checked"] >= 4
    assert report["regression_files_checked"] >= 1
    assert "surface_superposition" in report["imported_modules"]


def test_regression_scenario_runner_files_exist_and_list_scenarios():
    project_root = Path(__file__).resolve().parents[1]
    runner_path = project_root / "scripts/run_regression_scenarios.py"

    assert runner_path.exists()
    assert runner_path.stat().st_size > 0

    completed = subprocess.run(
        [sys.executable, str(runner_path), "--list"],
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "function_rectangles_rf" in completed.stdout
    assert "quick_start_preset" in completed.stdout


def test_regression_summary_serialization_helper(tmp_path):
    from scripts.run_regression_scenarios import serialize_result, write_summary

    result = serialize_result(
        {
            "name": "demo",
            "status": "PASS",
            "duration_s": np.float64(0.1),
            "metrics": {
                "field_shape": np.array([2, 2, 2, 3]),
                "max_abs_field": np.float64(1.25),
                "time_points": np.int64(3),
            },
        }
    )
    assert result["metrics"]["field_shape"] == [2, 2, 2, 3]

    summary = write_summary([result], output_dir=tmp_path)
    json_path = Path(summary["json_path"])
    csv_path = Path(summary["csv_path"])

    assert summary["status"] == "ok"
    assert json_path.exists()
    assert csv_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["results"][0]["name"] == "demo"


def test_config_values_are_loaded():
    assert isinstance(DEFAULT_CONFIG, SimulationConfig)
    assert isinstance(DEFAULT_CONFIG.geometry, GeometryConfig)
    assert isinstance(DEFAULT_CONFIG.mesh, MeshConfig)
    assert isinstance(DEFAULT_CONFIG.particle, ParticleConfig)
    assert isinstance(DEFAULT_CONFIG.voltage, VoltageConfig)
    assert isinstance(DEFAULT_CONFIG.solver, SolverConfig)
    assert DEFAULT_CONFIG.domain_size[0] > 0.0
    assert DEFAULT_CONFIG.particle_mass > 0.0


def test_config_flat_fields_are_backward_compatible():
    assert DEFAULT_CONFIG.domain_size == DEFAULT_CONFIG.geometry.domain_size
    assert DEFAULT_CONFIG.mesh_cells == DEFAULT_CONFIG.mesh.mesh_cells
    assert DEFAULT_CONFIG.rf_voltage == DEFAULT_CONFIG.voltage.rf_voltage
    assert DEFAULT_CONFIG.initial_position == DEFAULT_CONFIG.particle.initial_position
    assert DEFAULT_CONFIG.simulation_time == DEFAULT_CONFIG.solver.simulation_time
    assert DEFAULT_CONFIG.output_dir == DEFAULT_CONFIG.output.output_dir


def test_particle_mass_can_be_computed_from_radius_and_density():
    particle = ParticleConfig(
        particle_mass=None,
        particle_radius=1.0e-6,
        particle_density=1000.0,
    )
    config = SimulationConfig(particle=particle)

    expected_mass = 1000.0 * (4.0 / 3.0) * np.pi * (1.0e-6) ** 3
    assert np.isclose(particle.sphere_mass(), expected_mass)
    assert np.isclose(config.particle_mass, expected_mass)


def test_drag_particle_mass_from_radius_density_is_positive():
    mass = particle_mass_from_radius_density(1.0e-6, 2200.0)

    assert np.isfinite(mass)
    assert mass > 0.0


def test_drag_models_return_finite_positive_values():
    mean_free_path = gas_mean_free_path(293.15, 101325.0)
    kn = knudsen_number(mean_free_path, 1.0e-6)
    stokes_gamma = stokes_drag_gamma(1.0e-6, 1.8e-5)
    slip = cunningham_slip_correction(kn)
    cunningham_gamma = stokes_cunningham_drag_gamma(1.0e-6, 1.8e-5, kn)
    epstein_gamma = epstein_drag_gamma(1.0e-6, 1.2, 450.0)

    assert np.isfinite(mean_free_path)
    assert mean_free_path > 0.0
    assert kn > 0.0
    assert slip >= 1.0
    assert stokes_gamma > 0.0
    assert cunningham_gamma > 0.0
    assert epstein_gamma > 0.0


def test_damping_gamma_modes_and_regime_classification():
    vacuum = compute_damping_gamma(
        1.0e-6,
        1.0e-18,
        EnvironmentConfig(environment_mode="vacuum"),
    )
    custom = compute_damping_gamma(
        1.0e-6,
        1.0e-18,
        EnvironmentConfig(environment_mode="custom", custom_gamma_kg_s=2.0e-15),
    )

    assert vacuum["gamma_kg_s"] == 0.0
    assert np.isclose(custom["gamma_kg_s"], 2.0e-15)
    assert classify_drag_regime(1.0e-3) == "stokes"
    assert classify_drag_regime(1.0) == "stokes_cunningham"
    assert classify_drag_regime(20.0) == "epstein"


def test_config_flat_overrides_update_nested_sections():
    config = with_config_overrides(
        DEFAULT_CONFIG,
        mesh_cells=(3, 4, 5),
        rf_voltage=12.0,
        simulation_time=(0.0, 1.0e-6),
    )

    assert config.mesh.mesh_cells == (3, 4, 5)
    assert config.mesh_cells == (3, 4, 5)
    assert config.voltage.rf_voltage == 12.0
    assert config.rf_voltage == 12.0
    assert config.solver.simulation_time == (0.0, 1.0e-6)


def test_fem_solver_runs_on_coarse_mesh():
    config = with_config_overrides(DEFAULT_CONFIG, mesh_cells=(3, 3, 3))
    fem_result = solve_laplace(config)

    assert fem_result.base_voltage == 1.0
    assert fem_result.potential.ndim == 1
    assert fem_result.potential.size == fem_result.basis.N
    assert fem_result.electric_field_grid.shape[-1] == 3


def test_voltage_scale_static_and_time_dependent_modes():
    static_config = with_config_overrides(
        DEFAULT_CONFIG,
        voltage_amplitude=12.0,
        use_time_dependent_voltage=False,
    )
    assert voltage_scale(0.0, static_config) == 12.0

    rf_config = with_config_overrides(
        DEFAULT_CONFIG,
        use_time_dependent_voltage=True,
        dc_voltage=3.0,
        rf_voltage=5.0,
        rf_angular_frequency=2.0,
    )
    assert voltage_scale(0.0, rf_config) == 8.0


def test_E_at_position_returns_three_component_vector():
    config = with_config_overrides(DEFAULT_CONFIG, mesh_cells=(3, 3, 3))
    fem_result = solve_laplace(config)
    E = make_E_at_position(fem_result)

    field = E((0.0, 0.0, 0.0))
    assert isinstance(field, np.ndarray)
    assert field.shape == (3,)


def test_save_and_load_fem_field_grid_npz(tmp_path):
    config = with_config_overrides(DEFAULT_CONFIG, mesh_cells=(3, 3, 3))
    fem_result = solve_laplace(config)
    output_path = tmp_path / "fem_field_grid.npz"

    save_fem_field_to_npz(fem_result, output_path)
    field_grid = load_field_grid_npz(output_path)

    assert field_grid.electric_field_grid.shape == fem_result.electric_field_grid.shape
    assert field_grid.potential_grid.shape == fem_result.potential_grid.shape
    assert field_grid.source_description is not None


def test_load_potential_grid_npz_computes_field(tmp_path):
    x_grid = np.linspace(-1.0, 1.0, 5)
    y_grid = np.linspace(-1.0, 1.0, 5)
    z_grid = np.linspace(-1.0, 1.0, 5)
    x_mesh, y_mesh, z_mesh = np.meshgrid(
        x_grid,
        y_grid,
        z_grid,
        indexing="ij",
    )
    potential_grid = x_mesh**2 + 2.0 * y_mesh + 0.5 * z_mesh
    output_path = tmp_path / "potential_grid.npz"
    np.savez_compressed(
        output_path,
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        potential_grid=potential_grid,
    )

    field_grid = load_potential_grid_npz(output_path)
    direct_field = compute_field_from_potential_grid(
        potential_grid,
        x_grid,
        y_grid,
        z_grid,
    )
    center_field = field_grid.electric_field_grid[2, 2, 2]

    assert field_grid.potential_grid.shape == potential_grid.shape
    assert np.allclose(direct_field, field_grid.electric_field_grid)
    assert np.allclose(center_field, np.array([0.0, -2.0, -0.5]))


def test_field_grid_interpolator_returns_three_component_vector():
    x_grid = np.linspace(-1.0, 1.0, 3)
    y_grid = np.linspace(-1.0, 1.0, 3)
    z_grid = np.linspace(-1.0, 1.0, 3)
    electric_field_grid = np.zeros((3, 3, 3, 3))
    electric_field_grid[..., 0] = 1.0
    electric_field_grid[..., 1] = 2.0
    electric_field_grid[..., 2] = 3.0
    field_grid = FieldGrid(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        electric_field_grid=electric_field_grid,
    )

    E = make_E_at_position_from_field_grid(field_grid)
    field = E((0.0, 0.0, 0.0))

    assert field.shape == (3,)
    assert np.allclose(field, np.array([1.0, 2.0, 3.0]))


def test_field_validation_accepts_good_synthetic_field():
    field_grid = make_synthetic_field_grid()
    report = validate_field_grid_report(field_grid)
    symmetry = estimate_symmetry_checks(field_grid)

    assert report["valid"]
    assert not report["errors"]
    assert report["grid_shape"] == (4, 4, 4)
    assert np.isfinite(symmetry["kx"])


def test_field_validation_rejects_bad_field_shape():
    field_grid = make_synthetic_field_grid()
    bad_field_grid = FieldGrid(
        x_grid=field_grid.x_grid,
        y_grid=field_grid.y_grid,
        z_grid=field_grid.z_grid,
        electric_field_grid=np.zeros((4, 4, 4)),
        potential_grid=field_grid.potential_grid,
    )

    report = validate_field_grid_report(bad_field_grid)

    assert not report["valid"]
    assert any("electric_field_grid shape" in error for error in report["errors"])


def test_compare_identical_fields_gives_near_zero_error():
    reference = make_synthetic_field_grid()
    candidate = make_synthetic_field_grid()

    field_metrics = compare_field_grids(reference, candidate)
    potential_metrics = compare_potential_grids(reference, candidate)

    assert np.isclose(field_metrics["mean_abs_error"], 0.0)
    assert np.isclose(field_metrics["max_abs_error"], 0.0)
    assert np.isclose(field_metrics["relative_l2_error"], 0.0)
    assert np.isclose(potential_metrics["relative_l2_error"], 0.0)


def test_compare_slightly_different_fields_gives_nonzero_error():
    reference = make_synthetic_field_grid()
    candidate = make_synthetic_field_grid(offset=0.01)

    field_metrics = compare_field_grids(reference, candidate)

    assert field_metrics["mean_abs_error"] > 0.0
    assert field_metrics["max_abs_error"] > 0.0
    assert field_metrics["relative_l2_error"] > 0.0
    assert field_metrics["component_errors"]["Ex"]["mean_abs_error"] > 0.0


def test_loading_generated_binary_mask_works(tmp_path):
    image_array = np.zeros((8, 8), dtype=np.uint8)
    image_array[2:6, 3:5] = 255
    image_path = tmp_path / "mask.png"
    Image.fromarray(image_array, mode="L").save(image_path)

    mask = load_binary_electrode_mask(image_path, threshold=0.5)

    assert mask.dtype == bool
    assert mask.sum() == 8


def test_surface_connected_components_and_voltage_map():
    mask = make_test_surface_mask()
    labels, number_of_components = detect_electrode_components(mask)
    voltage_map = build_voltage_map_from_components(labels, {1: 1.0, 2: -1.0})

    assert number_of_components == 2
    assert np.isclose(voltage_map[mask].max(), 1.0)
    assert np.isclose(voltage_map[mask].min(), -1.0)
    assert np.all(voltage_map[~mask] == 0.0)


def test_parabolic_surface_mask_has_active_pixels():
    mask = make_parabolic_test_mask()
    labels, number_of_components = detect_electrode_components(mask)

    assert mask.sum() > 0
    assert number_of_components >= 1
    assert labels.shape == mask.shape


def test_high_resolution_mask_is_downsampled_for_computation():
    mask = make_high_resolution_surface_mask()

    computational_mask, metadata = downsample_binary_mask_for_computation(
        mask,
        max_size=64,
    )

    assert computational_mask.dtype == bool
    assert computational_mask.shape == (32, 64)
    assert metadata["original_shape"] == (256, 512)
    assert metadata["computational_shape"] == (32, 64)
    assert metadata["active_pixels_after"] < metadata["active_pixels_before"]
    assert np.isclose(metadata["scale"], 64 / 512)


def test_surface_pixel_area_uses_computational_mask_resolution():
    mask = make_high_resolution_surface_mask()
    voltage_map = np.where(mask, 1.0, 0.0)
    config = SurfaceMaskConfig(
        x_size_m=2.0e-3,
        y_size_m=1.0e-3,
        max_computational_mask_size=64,
    )

    computational_voltage_map, computational_mask, metadata = (
        prepare_surface_voltage_map_for_computation(
            voltage_map,
            config,
            mask=mask,
        )
    )
    _x_pixels, _y_pixels, pixel_area = make_mask_pixel_coordinates(
        computational_mask,
        config,
    )

    expected_area = (
        config.x_size_m
        / metadata["computational_shape"][1]
        * config.y_size_m
        / metadata["computational_shape"][0]
    )
    assert computational_voltage_map.shape == computational_mask.shape
    assert np.isclose(pixel_area, expected_area)


def test_electrode_edge_detection_for_rectangle():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    edge_mask, edge_coordinates = detect_electrode_edges(mask)

    assert edge_mask.sum() == 12
    assert len(edge_coordinates) == 12
    assert not edge_mask[3, 3]


def test_surface_uniform_and_center_clustered_grids_are_strictly_increasing():
    uniform = make_rectilinear_grid_1d(1.0e-3, 9, mode="uniform")
    clustered = make_rectilinear_grid_1d(
        1.0e-3,
        21,
        mode="center_clustered_tanh",
        cluster_strength=2.5,
    )
    center = len(clustered) // 2

    assert np.all(np.diff(uniform) > 0.0)
    assert np.all(np.diff(clustered) > 0.0)
    center_spacing = clustered[center + 1] - clustered[center]
    edge_spacing = clustered[-1] - clustered[-2]
    assert center_spacing < edge_spacing


def test_surface_edge_aware_grid_is_strict_and_refined_near_edges():
    mask = make_test_surface_mask()
    config = SurfaceMaskConfig(
        x_size_m=1.0e-3,
        y_size_m=1.0e-3,
        nx=9,
        ny=9,
        grid_mode_xy="edge_aware",
        max_edge_grid_points=8,
        edge_refinement_points_per_edge=5,
        edge_refinement_radius_m=5.0e-5,
        min_grid_spacing_m=1.0e-6,
    )
    x_grid, y_grid = make_edge_aware_xy_grid(mask, config)

    assert np.all(np.diff(x_grid) > 0.0)
    assert np.all(np.diff(y_grid) > 0.0)
    assert len(x_grid) <= 9 + 9 + config.max_edge_grid_points * 5
    assert np.min(np.diff(x_grid)) < np.median(np.diff(x_grid))


def test_surface_z_grid_never_includes_zero():
    z_grid = make_z_grid(
        z_max_m=1.0e-3,
        min_z_m=2.0e-5,
        nz=12,
        mode="near_surface_clustered",
    )

    assert np.all(np.diff(z_grid) > 0.0)
    assert np.all(z_grid > 0.0)


def test_surface_direct_sum_default_active_pixel_limit_is_qa_friendly():
    config = SurfaceMaskConfig()

    assert config.max_active_pixels_for_direct_sum >= 10000
    assert config.active_pixel_chunk_size > 0


def test_surface_potential_and_field_shapes_are_finite():
    mask = make_test_surface_mask()
    labels, _number_of_components = detect_electrode_components(mask)
    voltage_map = build_voltage_map_from_components(labels, {1: 1.0, 2: -1.0})
    config = SurfaceMaskConfig(
        x_size_m=1.0e-3,
        y_size_m=1.0e-3,
        z_max_m=3.0e-4,
        min_z_m=2.0e-5,
        nx=7,
        ny=7,
        nz=5,
        grid_mode_xy="uniform",
        grid_mode_z="uniform",
        max_active_pixels_for_direct_sum=100,
        chunk_size_points=20,
    )
    potential_grid, x_grid, y_grid, z_grid = compute_potential_from_surface_voltage_map(
        voltage_map,
        config,
        mask=mask,
    )
    field_grid = compute_surface_field_grid(voltage_map, config, mask=mask)

    assert potential_grid.shape == (7, 7, 5)
    assert field_grid.electric_field_grid.shape == (7, 7, 5, 3)
    assert np.all(np.isfinite(potential_grid))
    assert np.all(np.isfinite(field_grid.electric_field_grid))
    assert np.array_equal(field_grid.x_grid, x_grid)
    assert np.array_equal(field_grid.y_grid, y_grid)
    assert np.array_equal(field_grid.z_grid, z_grid)


def test_surface_potential_active_pixel_chunking_matches_large_chunk_reference():
    voltage_map = np.zeros((6, 6), dtype=float)
    voltage_map[1:3, 1:3] = 1.0
    voltage_map[3:5, 3:5] = -0.5
    mask = voltage_map != 0.0
    config = SurfaceMaskConfig(
        x_size_m=1.0e-3,
        y_size_m=1.0e-3,
        z_max_m=2.0e-4,
        min_z_m=2.0e-5,
        nx=4,
        ny=4,
        nz=3,
        grid_mode_xy="uniform",
        grid_mode_z="uniform",
        downsample_mask_for_computation=False,
        max_active_pixels_for_direct_sum=100,
        chunk_size_points=3,
    )

    chunked, *_ = compute_potential_from_surface_voltage_map(
        voltage_map,
        config,
        mask=mask,
        active_pixel_chunk_size=2,
    )
    reference, *_ = compute_potential_from_surface_voltage_map(
        voltage_map,
        config,
        mask=mask,
        active_pixel_chunk_size=100,
    )

    assert np.allclose(chunked, reference, rtol=1.0e-12, atol=1.0e-18)


def test_surface_field_computation_downsamples_high_resolution_mask():
    mask = make_high_resolution_surface_mask()
    voltage_map = np.where(mask, 1.0, 0.0)
    config = SurfaceMaskConfig(
        x_size_m=1.0e-3,
        y_size_m=1.0e-3,
        z_max_m=2.0e-4,
        min_z_m=2.0e-5,
        nx=5,
        ny=5,
        nz=3,
        grid_mode_xy="uniform",
        grid_mode_z="uniform",
        max_computational_mask_size=32,
        max_active_pixels_for_direct_sum=2000,
        chunk_size_points=20,
    )

    field_grid = compute_surface_field_grid(voltage_map, config, mask=mask)
    metadata = field_grid.surface_computation_metadata

    assert metadata["original_shape"] == (256, 512)
    assert metadata["computational_shape"] == (16, 32)
    assert metadata["active_pixels_after"] < metadata["active_pixels_before"]
    assert field_grid.electric_field_grid.shape == (5, 5, 3, 3)
    assert np.all(np.isfinite(field_grid.electric_field_grid))


def test_surface_direct_sum_active_pixel_limit_is_enforced():
    voltage_map = np.ones((6, 6), dtype=float)
    config = SurfaceMaskConfig(
        nx=3,
        ny=3,
        nz=3,
        max_active_pixels_for_direct_sum=10,
    )

    with pytest.raises(ValueError, match="Слишком много активных"):
        compute_potential_from_surface_voltage_map(voltage_map, config)


def test_surface_nonuniform_gradient_sign():
    x_grid = np.array([-1.0, -0.2, 0.0, 0.4, 1.0])
    y_grid = np.array([-1.0, -0.5, 0.3, 1.0])
    z_grid = np.array([0.1, 0.2, 0.6, 1.1])
    x_mesh, y_mesh, z_mesh = np.meshgrid(
        x_grid,
        y_grid,
        z_grid,
        indexing="ij",
    )
    potential = x_mesh + 2.0 * y_mesh - 0.5 * z_mesh
    electric_field = compute_field_from_potential_grid_nonuniform(
        potential,
        x_grid,
        y_grid,
        z_grid,
    )

    assert np.allclose(electric_field[..., 0], -1.0)
    assert np.allclose(electric_field[..., 1], -2.0)
    assert np.allclose(electric_field[..., 2], 0.5)


def test_surface_fieldgrid_interpolation_works():
    mask = make_test_surface_mask()
    labels, _number_of_components = detect_electrode_components(mask)
    voltage_map = build_voltage_map_from_components(labels, {1: 1.0, 2: -1.0})
    config = SurfaceMaskConfig(
        x_size_m=1.0e-3,
        y_size_m=1.0e-3,
        z_max_m=3.0e-4,
        min_z_m=2.0e-5,
        nx=7,
        ny=7,
        nz=5,
        max_active_pixels_for_direct_sum=100,
    )
    field_grid = compute_surface_field_grid(voltage_map, config, mask=mask)
    E = make_E_at_position_from_field_grid(field_grid)
    field = E((0.0, 0.0, 1.0e-4))

    assert field.shape == (3,)
    assert np.all(np.isfinite(field))


def test_ode_solver_returns_correct_trajectory_shape():
    config = with_config_overrides(
        DEFAULT_CONFIG,
        simulation_time=(0.0, 1.0e-5),
        time_step=1.0e-6,
        max_time_step=1.0e-6,
    )

    def zero_field(_position):
        return np.zeros(3)

    result = simulate_particle(zero_field, config)

    assert result.states.shape[1] == 6
    assert result.positions.shape[1] == 3
    assert result.velocities.shape[1] == 3
    assert result.t.shape[0] == result.positions.shape[0]


def test_parameter_sweep_returns_status_map():
    config = with_config_overrides(
        DEFAULT_CONFIG,
        mesh_cells=(3, 3, 3),
        simulation_time=(0.0, 2.0e-6),
        time_step=1.0e-6,
        max_time_step=1.0e-6,
    )

    results, survival_map, confinement_map = run_parameter_sweep(
        rf_voltages=np.array([0.0]),
        rf_angular_frequencies=np.array([1.0e3]),
        base_config=config,
    )

    assert len(results) == 1
    assert results[0]["status"] in {"escaped", "survived", "confined"}
    assert survival_map.shape == (1, 1)
    assert confinement_map.shape == (1, 1)
    for column in CSV_COLUMNS:
        assert column in results[0]


def test_trajectory_metrics_are_computed():
    config = with_config_overrides(
        DEFAULT_CONFIG,
        simulation_time=(0.0, 1.0e-6),
        time_step=1.0e-6,
        max_time_step=1.0e-6,
    )

    def zero_field(_position):
        return np.zeros(3)

    result = simulate_particle(zero_field, config)
    metrics = compute_trajectory_metrics(result)

    assert metrics["max_radius"] >= metrics["final_radius"]
    assert metrics["final_speed"] >= 0.0


def test_localization_status_classifier_handles_core_cases():
    config = with_config_overrides(
        DEFAULT_CONFIG,
        simulation_time=(0.0, 1.0e-3),
        time_step=1.0e-3,
        confinement_radius_threshold=1.0e-4,
    )
    localized = ParticleResult(
        t=np.array([0.0, 1.0e-3]),
        states=np.zeros((2, 6)),
        positions=np.array([[0.0, 0.0, 0.0], [5.0e-5, 0.0, 0.0]]),
        velocities=np.zeros((2, 3)),
        speed=np.zeros(2),
        left_domain=False,
        exit_time=None,
    )
    escaped = ParticleResult(
        t=np.array([0.0, 5.0e-4]),
        states=np.zeros((2, 6)),
        positions=np.array([[0.0, 0.0, 0.0], [6.0e-4, 0.0, 0.0]]),
        velocities=np.zeros((2, 3)),
        speed=np.zeros(2),
        left_domain=True,
        exit_time=5.0e-4,
    )
    unclear = ParticleResult(
        t=np.array([0.0, 1.0e-3]),
        states=np.zeros((2, 6)),
        positions=np.array([[0.0, 0.0, 0.0], [2.0e-4, 0.0, 0.0]]),
        velocities=np.zeros((2, 3)),
        speed=np.zeros(2),
        left_domain=False,
        exit_time=None,
    )

    assert classify_localization_status(localized, config, 1.0e-4) == "localized_like"
    assert classify_localization_status(escaped, config, 1.0e-4) == "escaped"
    assert classify_localization_status(unclear, config, 1.0e-4) == "unclear"
    assert localization_status_label("localized_like") == "Похоже на локализацию"


def test_parameter_search_with_existing_field_returns_expected_shape():
    config = with_config_overrides(
        DEFAULT_CONFIG,
        simulation_time=(0.0, 1.0e-6),
        time_step=1.0e-6,
        max_time_step=1.0e-6,
    )

    def zero_field(_position):
        return np.zeros(3)

    rows, status_map, metric_maps = run_parameter_search_with_field(
        zero_field,
        config,
        rf_voltages=np.array([5.0, 10.0]),
        rf_frequencies_hz=np.array([1.0e3, 2.0e3]),
        max_simulations=4,
    )

    assert len(rows) == 4
    assert status_map.shape == (2, 2)
    assert metric_maps["max_radius"].shape == (2, 2)


def test_result_row_contains_rf_period_metrics():
    config = with_config_overrides(
        DEFAULT_CONFIG,
        simulation_time=(0.0, 1.0e-6),
        time_step=1.0e-6,
        max_time_step=1.0e-6,
    )

    def zero_field(_position):
        return np.zeros(3)

    result = simulate_particle(zero_field, config)
    row = build_result_row(
        result,
        config,
        rf_voltage=10.0,
        rf_angular_frequency=2.0 * np.pi * 1.0e5,
        confinement_radius_threshold=config.confinement_radius_threshold,
    )

    assert np.isclose(row["rf_period"], 1.0e-5)
    assert np.isclose(row["simulated_rf_periods"], 0.1)
    for column in CSV_COLUMNS:
        assert column in row


def test_case_study_cases_are_defined():
    assert len(CASES) >= 3

    label = make_case_label(10.0, 500.0)
    assert "10V" in label
    assert "500Hz" in label


def test_rf_case_uses_period_resolved_time_step():
    config = with_config_overrides(
        DEFAULT_CONFIG,
        max_time_step=1.0e-5,
        time_step=1.0e-5,
    )
    period = rf_period_from_frequency(rf_frequency_hz=1.0e5)
    step = rf_resolved_time_step(config, rf_frequency_hz=1.0e5)
    case_config = make_rf_case_config(
        config,
        rf_voltage=10.0,
        rf_angular_frequency=2.0 * np.pi * 1.0e5,
    )

    assert np.isclose(period, 1.0e-5)
    assert np.isclose(step, 2.0e-7)
    assert np.isclose(case_config.max_time_step, 2.0e-7)
    assert np.isclose(case_config.time_step, 2.0e-7)


def test_app_helpers_import_without_running_streamlit():
    values = parse_comma_separated_floats("5, 10, 20")

    assert np.allclose(values, np.array([5.0, 10.0, 20.0]))


def test_dashboard_mode_mapping_and_status_badge_helpers():
    assert dashboard_mode_to_action("Проверить локализацию") == "check"
    assert dashboard_mode_to_action("Подобрать RF-параметры") == "search"
    assert dashboard_mode_to_action("Посмотреть поле / экспорт") == "field_only"

    badge = status_badge_html("localized_like")

    assert "Похоже на локализацию" in badge
    assert "status-badge" in badge


def test_app_field_grid_diagnostics_include_rf_pseudopotential():
    field_grid = make_synthetic_field_grid()
    config = with_config_overrides(
        DEFAULT_CONFIG,
        use_time_dependent_voltage=True,
        rf_voltage=10.0,
        rf_angular_frequency=2.0 * np.pi * 1.0e4,
    )

    diagnostics = compute_field_grid_diagnostics(field_grid, config)

    assert diagnostics["grid_shape"] == (4, 4, 4)
    assert np.isfinite(diagnostics["min_field_magnitude"])
    assert diagnostics["pseudopotential"] is not None
    assert np.isfinite(diagnostics["pseudopotential"]["min_value_J"])


def test_demo_presets_define_expected_ui_defaults():
    assert "Custom" in DEMO_PRESETS
    assert "High-frequency RF confined test" in DEMO_PRESETS
    assert "Mathieu RF demo" in DEMO_PRESETS

    quick_sweep = DEMO_PRESETS["Quick sweep demo"]
    assert quick_sweep["mode"] == "RF"
    assert "sweep_voltages" in quick_sweep
    assert "sweep_frequencies" in quick_sweep

    mathieu_demo = DEMO_PRESETS["Mathieu RF demo"]
    assert mathieu_demo["mode"] == "RF"
    assert mathieu_demo["dc_voltage"] == 0.0
    assert mathieu_demo["rf_voltage"] == 20.0
    assert mathieu_demo["rf_frequency_hz"] == 3.0e4

    assert preset_value(DEMO_PRESETS["Custom"], "rf_voltage", 12.0) == 12.0


def test_frequency_and_omega_conversions_are_consistent():
    omega = frequency_hz_to_omega(1.0e3)

    assert np.isclose(omega, 2.0 * np.pi * 1.0e3)
    assert np.isclose(omega_to_frequency_hz(omega), 1.0e3)


def test_ideal_mathieu_parameters_return_expected_keys():
    parameters = compute_ideal_mathieu_parameters(
        particle_charge=1.0e-16,
        particle_mass=1.0e-18,
        dc_voltage=1.0,
        rf_voltage=2.0,
        r0=5.0e-4,
        omega=2.0 * np.pi * 1.0e5,
    )

    assert set(parameters) == {
        "mathieu_a_x",
        "mathieu_q_x",
        "mathieu_a_y",
        "mathieu_q_y",
    }
    assert np.isclose(parameters["mathieu_a_y"], -parameters["mathieu_a_x"])
    assert np.isclose(parameters["mathieu_q_y"], -parameters["mathieu_q_x"])


def test_fem_curvature_estimation_runs_on_coarse_mesh():
    config = with_config_overrides(DEFAULT_CONFIG, mesh_cells=(3, 3, 3))
    fem_result = solve_laplace(config)
    curvature = estimate_potential_curvature_near_center(fem_result)

    assert {"kx", "ky", "kz", "fit_radius", "number_of_points_used"}.issubset(
        curvature
    )
    assert curvature["number_of_points_used"] >= 7
    assert np.isfinite(curvature["kx"])


def test_mathieu_stability_diagram_returns_figure_and_axes():
    fig, ax = plot_mathieu_stability_diagram(
        points={"mathieu_a": 0.0, "mathieu_q": 0.1, "status": "current"},
        grid_size=(21, 17),
        steps_per_period=24,
    )

    assert fig is ax.figure
    assert ax.get_xlabel() == "q_mathieu"
    assert ax.get_ylabel() == "a_mathieu"
    assert ax.get_title() == "Mathieu stability diagram, first region"
    plt.close(fig)


def test_mathieu_diagram_keeps_standard_limits_for_huge_point_by_default():
    fig, ax = plot_mathieu_stability_diagram(
        points={"mathieu_a": 1500.0, "mathieu_q": 0.0, "status": "current"},
        grid_size=(21, 17),
        steps_per_period=24,
    )

    assert np.allclose(ax.get_xlim(), (0.0, 1.2))
    assert np.allclose(ax.get_ylim(), (-0.5, 0.5))
    plt.close(fig)


def test_mathieu_diagram_auto_zoom_can_include_huge_point():
    fig, ax = plot_mathieu_stability_diagram(
        points={"mathieu_a": 1500.0, "mathieu_q": 0.0, "status": "current"},
        grid_size=(21, 17),
        steps_per_period=24,
        auto_zoom=True,
    )

    assert ax.get_ylim()[1] > 1500.0
    plt.close(fig)


def test_mathieu_ui_outside_standard_view_helper():
    inside = {
        "mathieu_a_x": 0.0,
        "mathieu_q_x": 0.2,
        "mathieu_a_y": 0.0,
        "mathieu_q_y": -0.2,
    }
    outside = {
        "mathieu_a_x": 1500.0,
        "mathieu_q_x": 0.0,
        "mathieu_a_y": -1500.0,
        "mathieu_q_y": 0.0,
    }

    assert not mathieu_parameters_outside_standard_view(inside)
    assert mathieu_parameters_outside_standard_view(outside)
