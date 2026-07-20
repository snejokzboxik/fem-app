"""Trajectory metrics and classification shared by sweeps and case studies."""

from __future__ import annotations

import numpy as np

from config import SimulationConfig
from mathieu_analysis import MATHIEU_PARAMETER_KEYS, nan_mathieu_parameters
from particle_dynamics import ParticleResult
from voltage_protocols import rf_period_from_frequency


CSV_COLUMNS = [
    "rf_voltage",
    "rf_angular_frequency",
    "rf_frequency_hz",
    "rf_period",
    "simulated_rf_periods",
    *MATHIEU_PARAMETER_KEYS,
    "status",
    "final_time",
    "max_radius",
    "final_radius",
    "max_abs_x",
    "max_abs_y",
    "max_abs_z",
    "final_x",
    "final_y",
    "final_z",
    "final_speed",
]


def compute_trajectory_metrics(particle_result: ParticleResult) -> dict:
    """Compute simple size and final-state metrics for one trajectory."""

    positions = particle_result.positions
    radii = np.linalg.norm(positions, axis=1)
    max_abs_coordinates = np.max(np.abs(positions), axis=0)
    final_position = positions[-1]

    return {
        "max_radius": float(np.max(radii)),
        "final_radius": float(radii[-1]),
        "max_abs_x": float(max_abs_coordinates[0]),
        "max_abs_y": float(max_abs_coordinates[1]),
        "max_abs_z": float(max_abs_coordinates[2]),
        "final_x": float(final_position[0]),
        "final_y": float(final_position[1]),
        "final_z": float(final_position[2]),
        "final_speed": float(particle_result.speed[-1]),
    }


def classify_particle_result(
    particle_result: ParticleResult,
    config: SimulationConfig,
    confinement_radius_threshold: float,
) -> str:
    """Classify one trajectory as escaped, survived, or confined."""

    reached_final_time = (
        particle_result.t[-1] >= config.simulation_time[1] - 0.5 * config.time_step
    )

    # A solver stop before the requested final time is not a valid survival
    # result.  In this simple three-label sweep, count it with escaped cases.
    if particle_result.left_domain or not reached_final_time:
        return "escaped"

    metrics = compute_trajectory_metrics(particle_result)
    if metrics["max_radius"] <= confinement_radius_threshold:
        return "confined"

    return "survived"


def classify_localization_status(
    particle_result: ParticleResult,
    config: SimulationConfig,
    confinement_radius_threshold: float,
) -> str:
    """Classify a trajectory for the Russian localization workflow."""

    if particle_result.left_domain:
        return "escaped"

    reached_final_time = (
        particle_result.t[-1] >= config.simulation_time[1] - 0.5 * config.time_step
    )
    if not reached_final_time:
        return "unclear"

    metrics = compute_trajectory_metrics(particle_result)
    if metrics["max_radius"] <= confinement_radius_threshold:
        return "localized_like"

    return "unclear"


def localization_status_label(status: str) -> str:
    """Return a Russian UI label for one localization status code."""

    labels = {
        "localized_like": "Похоже на локализацию",
        "escaped": "Частица вылетела",
        "unclear": "Неясно",
    }
    return labels.get(status, status)


def build_result_row(
    particle_result: ParticleResult,
    config: SimulationConfig,
    rf_voltage: float,
    rf_angular_frequency: float,
    confinement_radius_threshold: float,
) -> dict:
    """Build one CSV-ready result dictionary."""

    status = classify_particle_result(
        particle_result,
        config,
        confinement_radius_threshold,
    )
    metrics = compute_trajectory_metrics(particle_result)
    rf_frequency_hz = float(rf_angular_frequency / (2.0 * np.pi))
    rf_period = rf_period_from_frequency(rf_frequency_hz)
    final_time = (
        particle_result.exit_time
        if particle_result.left_domain
        else float(particle_result.t[-1])
    )
    simulated_rf_periods = final_time / rf_period if np.isfinite(rf_period) else 0.0

    return {
        "rf_voltage": float(rf_voltage),
        "rf_angular_frequency": float(rf_angular_frequency),
        "rf_frequency_hz": rf_frequency_hz,
        "rf_period": float(rf_period),
        "simulated_rf_periods": float(simulated_rf_periods),
        **nan_mathieu_parameters(),
        "status": status,
        "final_time": final_time,
        **metrics,
    }
