"""Experiment configuration helpers for reproducible Streamlit runs.

The goal of this module is not to replace ``config.py``.  It captures the
larger UI-level experiment: geometry source, electrode assignments, particle
settings, voltage settings, dynamics settings, and optional scan settings.
Everything returned here is JSON-serializable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


EXPERIMENT_CONFIG_VERSION = 1


@dataclass
class ExperimentConfig:
    """A JSON-serializable snapshot of one UI experiment."""

    version: int = EXPERIMENT_CONFIG_VERSION
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    title: str = "Untitled charged-particle trap experiment"
    description: str = ""
    workflow_mode: str = ""
    geometry_source: str = ""
    geometry_config: dict[str, Any] = field(default_factory=dict)
    electrode_assignments: list[dict[str, Any]] = field(default_factory=list)
    rf_config: dict[str, Any] = field(default_factory=dict)
    dc_config: dict[str, Any] = field(default_factory=dict)
    particle_config: dict[str, Any] = field(default_factory=dict)
    environment_config: dict[str, Any] = field(default_factory=dict)
    dynamics_config: dict[str, Any] = field(default_factory=dict)
    field_grid_config: dict[str, Any] = field(default_factory=dict)
    diagnostics_config: dict[str, Any] = field(default_factory=dict)
    parameter_search_config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary safe for JSON export."""

        return _json_safe(asdict(self))


def _json_safe(value):
    """Recursively convert common numeric/container values to JSON-safe data."""

    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is available in this project.
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__") and value.__class__.__name__ == "ElectrodeAssignment":
        return _json_safe(value.__dict__)
    return value


def _assignment_to_dict(assignment) -> dict[str, Any]:
    """Serialize an electrode assignment-like object."""

    if isinstance(assignment, dict):
        return _json_safe(assignment)
    return {
        "region_id": int(getattr(assignment, "region_id", 0)),
        "name": str(getattr(assignment, "name", "")),
        "role": str(getattr(assignment, "role", "GND")),
        "voltage": float(getattr(assignment, "voltage", 0.0)),
        "rf_phase": float(getattr(assignment, "rf_phase", 0.0)),
    }


def _surface_geometry_config(surface_options: dict | None) -> dict[str, Any]:
    """Collect the lightweight geometry config from surface options."""

    if not surface_options:
        return {}
    config = surface_options.get("config")
    geometry_config: dict[str, Any] = {
        "source_kind": surface_options.get("source_kind"),
        "number_of_components": surface_options.get("number_of_components", 0),
        "uses_rf_dc_separation": bool(surface_options.get("uses_rf_dc_separation")),
    }
    if config is not None:
        for name in (
            "x_size_m",
            "y_size_m",
            "z_max_m",
            "min_z_m",
            "nx",
            "ny",
            "nz",
            "grid_mode_xy",
            "grid_mode_z",
            "max_computational_mask_size",
            "max_active_pixels_for_direct_sum",
        ):
            if hasattr(config, name):
                geometry_config[name] = getattr(config, name)
    if "downsample_metadata" in surface_options:
        geometry_config["downsample_metadata"] = surface_options.get(
            "downsample_metadata"
        )
    if "canvas_binary_mask" in surface_options:
        geometry_config["has_canvas_mask"] = True
        try:
            from surface_geometry_sources import canvas_design_to_dict

            mask = surface_options.get("canvas_binary_mask")
            assignments = surface_options.get("assignments", [])
            if mask is not None and config is not None:
                geometry_config["canvas_design"] = canvas_design_to_dict(
                    x_size_m=float(getattr(config, "x_size_m")),
                    y_size_m=float(getattr(config, "y_size_m")),
                    canvas_resolution_px=int(max(mask.shape)),
                    binary_mask=mask,
                    assignments=assignments,
                    notes="Captured from Streamlit canvas geometry.",
                )
        except Exception as exc:
            geometry_config["canvas_design_warning"] = (
                f"Canvas design could not be serialized: {exc}"
            )
    return _json_safe(geometry_config)


def collect_current_experiment_config_from_ui(
    *,
    config=None,
    workflow_mode: str = "",
    geometry_source: str = "",
    surface_options: dict | None = None,
    environment_report: dict | None = None,
    parameter_search_config: dict | None = None,
    session_state: dict | None = None,
    title: str = "Streamlit charged-particle trap experiment",
    description: str = "",
    notes: str = "",
) -> ExperimentConfig:
    """Build an experiment snapshot from current UI data.

    The function accepts plain dictionaries so it remains easy to test without
    importing Streamlit.
    """

    state = session_state or {}
    assignments = [
        _assignment_to_dict(assignment)
        for assignment in (surface_options or {}).get("assignments", [])
    ]
    geometry_config = _surface_geometry_config(surface_options)
    if "function_electrode_defs" in state:
        geometry_config["function_definitions"] = _json_safe(
            state["function_electrode_defs"]
        )
    if "canvas_confirmed_mask" in state:
        geometry_config["has_canvas_confirmed_mask"] = True
    if "canvas_assignments" in state:
        geometry_config["canvas_assignments"] = _json_safe(
            state["canvas_assignments"]
        )

    rf_frequency = (
        float(config.rf_angular_frequency) / (2.0 * 3.141592653589793)
        if config is not None
        else state.get("dash_rf_frequency")
    )
    experiment = ExperimentConfig(
        title=title,
        description=description,
        workflow_mode=workflow_mode
        or str(state.get("app_workflow_mode", state.get("workflow_mode", ""))),
        geometry_source=geometry_source or str(state.get("dash_source_label", "")),
        geometry_config=geometry_config,
        electrode_assignments=assignments,
        rf_config={
            "use_time_dependent_voltage": getattr(
                config,
                "use_time_dependent_voltage",
                state.get("dash_voltage_mode", "RF") == "RF",
            ),
            "rf_voltage": getattr(config, "rf_voltage", state.get("dash_rf_voltage")),
            "rf_frequency_hz": rf_frequency,
            "rf_angular_frequency": getattr(
                config,
                "rf_angular_frequency",
                None,
            ),
        },
        dc_config={
            "dc_voltage": getattr(config, "dc_voltage", state.get("dash_dc_voltage")),
            "voltage_amplitude": getattr(
                config,
                "voltage_amplitude",
                state.get("dash_voltage_amplitude"),
            ),
        },
        particle_config={
            "particle_mass": getattr(config, "particle_mass", state.get("dash_particle_mass")),
            "particle_charge": getattr(config, "particle_charge", state.get("dash_particle_charge")),
            "particle_radius": getattr(config, "particle_radius", state.get("dash_particle_radius")),
            "particle_density": getattr(config, "particle_density", state.get("dash_particle_density")),
        },
        environment_config=_json_safe(environment_report or {}),
        dynamics_config={
            "initial_position": getattr(config, "initial_position", None),
            "initial_velocity": getattr(config, "initial_velocity", None),
            "simulation_time": getattr(config, "simulation_time", None),
            "time_step": getattr(config, "time_step", state.get("dash_time_step")),
            "max_time_step": getattr(config, "max_time_step", None),
            "confinement_radius_threshold": getattr(
                config,
                "confinement_radius_threshold",
                state.get("dash_confinement_radius"),
            ),
        },
        field_grid_config={
            "domain_size": getattr(config, "domain_size", None),
            "mesh_cells": getattr(config, "mesh_cells", None),
            "output_dir": getattr(config, "output_dir", None),
        },
        diagnostics_config={
            "plot_quality": state.get("plot_quality"),
            "mathieu_note": "Surface geometries use local effective estimates.",
        },
        parameter_search_config=_json_safe(parameter_search_config or {}),
        notes=notes,
    )
    return experiment


def validate_experiment_config(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a loaded experiment config and return a readable report."""

    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, dict):
        return {
            "valid": False,
            "errors": ["Experiment config must be a JSON object."],
            "warnings": [],
        }
    version = data.get("version")
    if version is None:
        errors.append("Missing required field: version.")
    elif int(version) > EXPERIMENT_CONFIG_VERSION:
        warnings.append(
            f"Config version {version} is newer than supported version "
            f"{EXPERIMENT_CONFIG_VERSION}."
        )
    for field_name in ("title", "workflow_mode", "geometry_source"):
        if field_name not in data:
            warnings.append(f"Missing optional field: {field_name}.")
    for section in (
        "geometry_config",
        "rf_config",
        "dc_config",
        "particle_config",
        "dynamics_config",
    ):
        if section in data and not isinstance(data[section], dict):
            errors.append(f"Section {section} must be an object.")
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def experiment_config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Create an ExperimentConfig dataclass from a tolerant dictionary."""

    report = validate_experiment_config(data)
    if not report["valid"]:
        raise ValueError("; ".join(report["errors"]))
    defaults = ExperimentConfig()
    merged = defaults.to_dict()
    merged.update(data)
    allowed = set(ExperimentConfig.__dataclass_fields__.keys())
    return ExperimentConfig(**{key: merged[key] for key in allowed})


def save_experiment_config_to_json(config: ExperimentConfig | dict, path) -> None:
    """Save an experiment config to JSON."""

    data = config.to_dict() if isinstance(config, ExperimentConfig) else _json_safe(config)
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_experiment_config_from_json(path) -> ExperimentConfig:
    """Load and validate an experiment config JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return experiment_config_from_dict(data)


def apply_experiment_config_to_session_state(
    experiment: ExperimentConfig | dict,
    session_state,
) -> dict[str, Any]:
    """Apply known experiment fields to a Streamlit-like session_state mapping."""

    data = experiment.to_dict() if isinstance(experiment, ExperimentConfig) else experiment
    report = validate_experiment_config(data)
    if not report["valid"]:
        return report

    def set_if_present(key: str, value):
        if value is not None:
            session_state[key] = value

    workflow_mode = data.get("workflow_mode")
    if workflow_mode is not None:
        session_state["app_workflow_mode"] = workflow_mode
        session_state["pending_workflow_mode"] = workflow_mode
        if isinstance(session_state, dict):
            session_state["workflow_mode"] = workflow_mode
    set_if_present("dash_source_label", data.get("geometry_source"))

    rf = data.get("rf_config", {})
    dc = data.get("dc_config", {})
    set_if_present("dash_voltage_mode", "RF" if rf.get("use_time_dependent_voltage", True) else "Статическое")
    set_if_present("dash_rf_voltage", rf.get("rf_voltage"))
    set_if_present("dash_rf_frequency", rf.get("rf_frequency_hz"))
    set_if_present("dash_dc_voltage", dc.get("dc_voltage"))
    set_if_present("dash_voltage_amplitude", dc.get("voltage_amplitude"))

    particle = data.get("particle_config", {})
    for source_key, state_key in (
        ("particle_mass", "dash_particle_mass"),
        ("particle_charge", "dash_particle_charge"),
        ("particle_radius", "dash_particle_radius"),
        ("particle_density", "dash_particle_density"),
    ):
        set_if_present(state_key, particle.get(source_key))

    dynamics = data.get("dynamics_config", {})
    position = dynamics.get("initial_position") or []
    velocity = dynamics.get("initial_velocity") or []
    for index, key in enumerate(("dash_x0", "dash_y0", "dash_z0")):
        if index < len(position):
            session_state[key] = position[index]
    for index, key in enumerate(("dash_vx0", "dash_vy0", "dash_vz0")):
        if index < len(velocity):
            session_state[key] = velocity[index]
    simulation_time = dynamics.get("simulation_time") or []
    if len(simulation_time) >= 2:
        session_state["dash_sim_time"] = float(simulation_time[1]) - float(simulation_time[0])
    set_if_present("dash_time_step", dynamics.get("time_step"))
    set_if_present(
        "dash_confinement_radius",
        dynamics.get("confinement_radius_threshold"),
    )

    geometry = data.get("geometry_config", {})
    for state_key, geometry_key in (
        ("function_x_size", "x_size_m"),
        ("function_y_size", "y_size_m"),
        ("function_z_max", "z_max_m"),
        ("function_min_z", "min_z_m"),
        ("function_surface_nx", "nx"),
        ("function_surface_ny", "ny"),
        ("function_surface_nz", "nz"),
        ("function_grid_mode_xy", "grid_mode_xy"),
        ("function_grid_mode_z", "grid_mode_z"),
        ("function_max_comp_mask", "max_computational_mask_size"),
        ("function_max_active", "max_active_pixels_for_direct_sum"),
    ):
        set_if_present(state_key, geometry.get(geometry_key))
    if "function_definitions" in geometry:
        session_state["function_electrode_defs"] = geometry["function_definitions"]
    if "canvas_assignments" in geometry:
        session_state["canvas_assignments"] = geometry["canvas_assignments"]

    search = data.get("parameter_search_config", {})
    set_if_present("dash_sweep_voltages", search.get("sweep_voltages"))
    set_if_present("dash_sweep_frequencies", search.get("sweep_frequencies"))
    set_if_present("dash_max_simulations", search.get("max_simulations"))
    return report
