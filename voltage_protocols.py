"""Voltage-protocol helpers shared by sweeps and case studies."""

from __future__ import annotations

import numpy as np

from config import SimulationConfig, with_config_overrides


def rf_period_from_frequency(rf_frequency_hz: float) -> float:
    """Return the RF period in seconds."""

    if rf_frequency_hz <= 0.0:
        return np.inf
    return 1.0 / rf_frequency_hz


def rf_resolved_time_step(base_config: SimulationConfig, rf_frequency_hz: float) -> float:
    """Choose an ODE step limit small enough to resolve the RF oscillation."""

    rf_period = rf_period_from_frequency(rf_frequency_hz)
    if not np.isfinite(rf_period):
        return base_config.max_time_step
    return min(base_config.max_time_step, rf_period / 50.0)


def make_rf_case_config(
    base_config: SimulationConfig,
    rf_voltage: float,
    rf_angular_frequency: float,
) -> SimulationConfig:
    """Return a config for one RF case with RF-resolved time stepping."""

    rf_frequency_hz = rf_angular_frequency / (2.0 * np.pi)
    resolved_step = rf_resolved_time_step(base_config, rf_frequency_hz)

    return with_config_overrides(
        base_config,
        use_time_dependent_voltage=True,
        rf_voltage=float(rf_voltage),
        rf_angular_frequency=float(rf_angular_frequency),
        max_time_step=resolved_step,
        # Keep output samples fine enough for metrics and plots too.  The ODE
        # solver still controls accuracy via max_time_step and tolerances.
        time_step=min(base_config.time_step, resolved_step),
    )
