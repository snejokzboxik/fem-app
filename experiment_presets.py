"""Built-in experiment presets for the Streamlit dashboard."""

from __future__ import annotations

import copy

import numpy as np

from experiment_config import EXPERIMENT_CONFIG_VERSION, validate_experiment_config
from surface_geometry_sources import (
    ROLE_CUSTOM,
    ROLE_DC,
    ROLE_GND,
    ROLE_RF,
    ElectrodeAssignment,
    canvas_design_to_dict,
)


def _function_definition(name, expression, role, voltage=0.0) -> dict:
    """Return a JSON-safe function electrode definition."""

    return {
        "name": name,
        "expression": expression,
        "role": role,
        "voltage": float(voltage),
        "rf_phase": 0.0,
    }


def _base_experiment(title: str, description: str, geometry_source: str) -> dict:
    """Return common lightweight defaults used by presets."""

    return {
        "version": EXPERIMENT_CONFIG_VERSION,
        "title": title,
        "description": description,
        "workflow_mode": "Проверить локализацию",
        "geometry_source": geometry_source,
        "geometry_config": {
            "x_size_m": 1.0e-3,
            "y_size_m": 1.0e-3,
            "z_max_m": 8.0e-4,
            "min_z_m": 2.0e-5,
            "nx": 21,
            "ny": 21,
            "nz": 13,
            "grid_mode_xy": "edge_aware",
            "grid_mode_z": "near_surface_clustered",
            "max_computational_mask_size": 128,
            "max_active_pixels_for_direct_sum": 10000,
        },
        "electrode_assignments": [],
        "rf_config": {
            "use_time_dependent_voltage": True,
            "rf_voltage": 20.0,
            "rf_frequency_hz": 3.0e4,
        },
        "dc_config": {
            "dc_voltage": 0.0,
            "voltage_amplitude": 1.0,
        },
        "particle_config": {
            "particle_charge": 1.0e-16,
            "particle_mass": 1.0e-18,
            "particle_radius": 1.0e-6,
            "particle_density": 2200.0,
        },
        "environment_config": {
            "mode_label": "Вакуум: трение выключено",
            "regime": "vacuum",
            "gamma_kg_s": 0.0,
        },
        "dynamics_config": {
            "initial_position": [0.0, 0.0, 1.2e-4],
            "initial_velocity": [0.0, 0.0, 0.0],
            "simulation_time": [0.0, 1.5e-3],
            "time_step": 2.0e-6,
            "max_time_step": 2.0e-6,
            "confinement_radius_threshold": 3.5e-4,
        },
        "field_grid_config": {},
        "diagnostics_config": {
            "expected_thing_to_look_at": "Посмотрите RF-null и псевдопотенциал.",
        },
        "parameter_search_config": {
            "sweep_voltages": "5,10,20",
            "sweep_frequencies": "5000,30000,100000",
            "max_simulations": 9,
        },
        "notes": "Preset parameters are placeholders for numerical exploration.",
    }


def _with_function_geometry(base: dict, definitions: list[dict]) -> dict:
    """Attach function-defined geometry and assignments to a preset."""

    preset = copy.deepcopy(base)
    preset["geometry_config"]["source_kind"] = "function"
    preset["geometry_config"]["function_definitions"] = definitions
    preset["electrode_assignments"] = [
        {
            "region_id": index,
            "name": definition["name"],
            "role": definition["role"],
            "voltage": definition.get("voltage", 0.0),
            "rf_phase": definition.get("rf_phase", 0.0),
        }
        for index, definition in enumerate(definitions, start=1)
    ]
    return preset


def _canvas_demo_design() -> dict:
    """Return a tiny encoded canvas design for the canvas preset."""

    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 12:20] = True
    mask[16:48, 44:52] = True
    mask[10:18, 26:38] = True
    assignments = [
        ElectrodeAssignment(1, "RF left", ROLE_RF),
        ElectrodeAssignment(2, "RF right", ROLE_RF),
        ElectrodeAssignment(3, "DC top", ROLE_DC, 0.5),
    ]
    return canvas_design_to_dict(
        x_size_m=1.0e-3,
        y_size_m=1.0e-3,
        canvas_resolution_px=64,
        binary_mask=mask,
        assignments=assignments,
        rf_amplitude=20.0,
        rf_frequency_hz=3.0e4,
        notes="Generated canvas demo design.",
    )


def built_in_experiment_presets() -> dict[str, dict]:
    """Return all built-in experiment presets by title."""

    quick_start = _with_function_geometry(
        _base_experiment(
            "Быстрый старт: 4 RF электрода",
            "Four rectangular RF rails with short simulation settings.",
            "Электроды функциями",
        ),
        [
            _function_definition(
                "RF left",
                "abs(x + 250e-6) < 70e-6 and abs(y) < 280e-6",
                ROLE_RF,
            ),
            _function_definition(
                "RF right",
                "abs(x - 250e-6) < 70e-6 and abs(y) < 280e-6",
                ROLE_RF,
            ),
            _function_definition(
                "RF top",
                "abs(y - 250e-6) < 70e-6 and abs(x) < 280e-6",
                ROLE_RF,
            ),
            _function_definition(
                "RF bottom",
                "abs(y + 250e-6) < 70e-6 and abs(x) < 280e-6",
                ROLE_RF,
            ),
        ],
    )
    quick_start["geometry_config"]["max_computational_mask_size"] = 96
    quick_start["diagnostics_config"]["expected_thing_to_look_at"] = (
        "Быстрый демо-расчёт использует грубую сетку, чтобы запускаться быстро."
    )

    rf_dc = _with_function_geometry(
        _base_experiment(
            "RF + DC compensation",
            "RF rails plus small DC compensation pads.",
            "Электроды функциями",
        ),
        [
            _function_definition("RF left", "abs(x + 230e-6) < 60e-6 and abs(y) < 320e-6", ROLE_RF),
            _function_definition("RF right", "abs(x - 230e-6) < 60e-6 and abs(y) < 320e-6", ROLE_RF),
            _function_definition("DC top", "abs(y - 290e-6) < 55e-6 and abs(x) < 170e-6", ROLE_DC, 0.5),
            _function_definition("DC bottom", "abs(y + 290e-6) < 55e-6 and abs(x) < 170e-6", ROLE_CUSTOM, -0.5),
            _function_definition("GND center", "(x**2 + y**2) < (55e-6)**2", ROLE_GND),
        ],
    )
    rf_dc["diagnostics_config"]["expected_thing_to_look_at"] = "Проверьте DC-смещение и траекторию."

    parabolic = _with_function_geometry(
        _base_experiment(
            "Параболические электроды",
            "Curved function-defined RF/DC electrodes for rasterization demos.",
            "Электроды функциями",
        ),
        [
            _function_definition(
                "RF parabola top",
                "y > 0.02*(x/1e-4)**2 + 80e-6 and y < 0.02*(x/1e-4)**2 + 210e-6",
                ROLE_RF,
            ),
            _function_definition(
                "RF parabola bottom",
                "y < -0.02*(x/1e-4)**2 - 80e-6 and y > -0.02*(x/1e-4)**2 - 210e-6",
                ROLE_RF,
            ),
        ],
    )

    ring = _with_function_geometry(
        _base_experiment(
            "Кольцевой электрод",
            "Ring-like RF electrode with a small DC center pad.",
            "Электроды функциями",
        ),
        [
            _function_definition(
                "RF ring",
                "(x**2 + y**2) > (150e-6)**2 and (x**2 + y**2) < (280e-6)**2",
                ROLE_RF,
            ),
            _function_definition(
                "DC center",
                "(x**2 + y**2) < (80e-6)**2",
                ROLE_DC,
                0.25,
            ),
        ],
    )

    canvas = _base_experiment(
        "Canvas demo",
        "Generated canvas-style mask with RF rails and one DC pad.",
        "Нарисовать электроды",
    )
    canvas["geometry_config"]["source_kind"] = "canvas"
    canvas["geometry_config"]["canvas_design"] = _canvas_demo_design()
    canvas["electrode_assignments"] = canvas["geometry_config"]["canvas_design"]["assignments"]
    canvas["diagnostics_config"]["expected_thing_to_look_at"] = "Проверьте компоненты canvas и RF/DC карты."

    imported = _base_experiment(
        "Imported field demo",
        "Uses examples/example_field_grid.npz when available.",
        "Загрузить поле .npz",
    )
    imported["field_grid_config"] = {
        "field_grid_path": "examples/example_field_grid.npz",
        "scaling_note": "Set voltage scale carefully if the imported field is already physical.",
    }
    imported["diagnostics_config"]["expected_thing_to_look_at"] = "Проверьте импортированное поле и масштабирование."

    presets = {
        quick_start["title"]: quick_start,
        rf_dc["title"]: rf_dc,
        parabolic["title"]: parabolic,
        ring["title"]: ring,
        canvas["title"]: canvas,
        imported["title"]: imported,
    }
    return presets


def get_experiment_preset(title: str) -> dict:
    """Return a deep copy of one preset."""

    return copy.deepcopy(built_in_experiment_presets()[title])


def validate_all_presets() -> dict[str, dict]:
    """Validate all built-in presets."""

    return {
        title: validate_experiment_config(preset)
        for title, preset in built_in_experiment_presets().items()
    }
