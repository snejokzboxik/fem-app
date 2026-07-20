"""Approximate gas-drag helpers for the Streamlit learning UI.

These formulas are deliberately simple.  They are useful for distinguishing
vacuum, continuum air drag, slip-corrected drag, and a free-molecular
Epstein-like regime, but they are not a final gas-dynamics model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np


BOLTZMANN_CONSTANT = 1.380649e-23
AIR_MOLECULAR_DIAMETER_M = 3.7e-10


@dataclass(frozen=True)
class EnvironmentConfig:
    """Gas and damping settings used to estimate linear damping gamma."""

    environment_mode: str = "custom"
    pressure_pa: float = 101325.0
    temperature_k: float = 293.15
    gas_viscosity_pa_s: float = 1.8e-5
    gas_molecular_mass_kg: float = 4.81e-26
    accommodation_coefficient: float = 1.0
    custom_gamma_kg_s: float = 1.0e-15


def particle_mass_from_radius_density(radius_m: float, density_kg_m3: float) -> float:
    """Return mass of a spherical particle."""

    _require_positive("radius_m", radius_m)
    _require_positive("density_kg_m3", density_kg_m3)
    return float((4.0 / 3.0) * pi * radius_m**3 * density_kg_m3)


def gas_mean_free_path(
    temperature_k: float,
    pressure_pa: float,
    molecular_diameter_m: float = AIR_MOLECULAR_DIAMETER_M,
) -> float:
    """Estimate gas mean free path from hard-sphere kinetic theory."""

    _require_positive("temperature_k", temperature_k)
    _require_positive("pressure_pa", pressure_pa)
    _require_positive("molecular_diameter_m", molecular_diameter_m)
    denominator = sqrt(2.0) * pi * molecular_diameter_m**2 * pressure_pa
    return float(BOLTZMANN_CONSTANT * temperature_k / denominator)


def knudsen_number(mean_free_path_m: float, particle_radius_m: float) -> float:
    """Return Kn = mean_free_path / particle_radius."""

    _require_positive("mean_free_path_m", mean_free_path_m)
    _require_positive("particle_radius_m", particle_radius_m)
    return float(mean_free_path_m / particle_radius_m)


def stokes_drag_gamma(radius_m: float, viscosity_pa_s: float) -> float:
    """Return continuum Stokes drag coefficient gamma = 6*pi*eta*r."""

    _require_positive("radius_m", radius_m)
    _require_positive("viscosity_pa_s", viscosity_pa_s)
    return float(6.0 * pi * viscosity_pa_s * radius_m)


def cunningham_slip_correction(kn: float) -> float:
    """Return a common Cunningham slip correction factor."""

    if kn < 0.0:
        raise ValueError("kn must be non-negative.")
    if kn == 0.0:
        return 1.0
    return float(1.0 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn)))


def stokes_cunningham_drag_gamma(
    radius_m: float,
    viscosity_pa_s: float,
    kn: float,
) -> float:
    """Return Stokes drag reduced by the Cunningham slip correction."""

    return float(stokes_drag_gamma(radius_m, viscosity_pa_s) / cunningham_slip_correction(kn))


def epstein_drag_gamma(
    radius_m: float,
    gas_density_kg_m3: float,
    thermal_speed_m_s: float,
    accommodation_coefficient: float = 1.0,
) -> float:
    """Return a simple Epstein-like free-molecular linear drag coefficient."""

    _require_positive("radius_m", radius_m)
    _require_positive("gas_density_kg_m3", gas_density_kg_m3)
    _require_positive("thermal_speed_m_s", thermal_speed_m_s)
    _require_positive("accommodation_coefficient", accommodation_coefficient)
    return float(
        (4.0 / 3.0)
        * pi
        * radius_m**2
        * gas_density_kg_m3
        * thermal_speed_m_s
        * accommodation_coefficient
    )


def classify_drag_regime(kn: float) -> str:
    """Classify drag by Knudsen number."""

    if kn < 0.0:
        raise ValueError("kn must be non-negative.")
    if kn < 0.01:
        return "stokes"
    if kn < 10.0:
        return "stokes_cunningham"
    return "epstein"


def compute_damping_gamma(
    particle_radius_m: float,
    particle_mass_kg: float,
    env_config: EnvironmentConfig,
) -> dict:
    """Estimate linear damping gamma and return diagnostics.

    The particle mass is accepted so the caller can compute the damping time
    m/gamma in one place.
    """

    _require_positive("particle_mass_kg", particle_mass_kg)

    mode = env_config.environment_mode
    if mode == "vacuum":
        return _gamma_report(
            gamma_kg_s=0.0,
            regime="vacuum",
            mean_free_path_m=np.inf,
            kn=np.inf,
            particle_mass_kg=particle_mass_kg,
        )

    if mode == "custom":
        gamma = max(float(env_config.custom_gamma_kg_s), 0.0)
        return _gamma_report(
            gamma_kg_s=gamma,
            regime="custom",
            mean_free_path_m=np.nan,
            kn=np.nan,
            particle_mass_kg=particle_mass_kg,
        )

    _require_positive("particle_radius_m", particle_radius_m)
    mean_free_path = gas_mean_free_path(
        env_config.temperature_k,
        env_config.pressure_pa,
    )
    kn = knudsen_number(mean_free_path, particle_radius_m)
    regime = classify_drag_regime(kn)

    if mode == "air_auto" and regime == "stokes":
        gamma = stokes_drag_gamma(particle_radius_m, env_config.gas_viscosity_pa_s)
    elif mode in {"air_auto", "pressure_gas"} and regime in {"stokes", "stokes_cunningham"}:
        gamma = stokes_cunningham_drag_gamma(
            particle_radius_m,
            env_config.gas_viscosity_pa_s,
            kn,
        )
    else:
        gas_density = (
            env_config.pressure_pa
            * env_config.gas_molecular_mass_kg
            / (BOLTZMANN_CONSTANT * env_config.temperature_k)
        )
        thermal_speed = sqrt(
            8.0 * BOLTZMANN_CONSTANT * env_config.temperature_k
            / (pi * env_config.gas_molecular_mass_kg)
        )
        gamma = epstein_drag_gamma(
            particle_radius_m,
            gas_density,
            thermal_speed,
            env_config.accommodation_coefficient,
        )

    return _gamma_report(gamma, regime, mean_free_path, kn, particle_mass_kg)


def _gamma_report(
    gamma_kg_s: float,
    regime: str,
    mean_free_path_m: float,
    kn: float,
    particle_mass_kg: float,
) -> dict:
    """Build a small diagnostics dictionary for UI display and tests."""

    damping_time_s = (
        float(particle_mass_kg / gamma_kg_s)
        if gamma_kg_s > 0.0
        else np.inf
    )
    return {
        "gamma_kg_s": float(gamma_kg_s),
        "regime": regime,
        "mean_free_path_m": float(mean_free_path_m),
        "knudsen_number": float(kn),
        "damping_time_s": damping_time_s,
    }


def _require_positive(name: str, value: float):
    """Raise a readable error for non-positive physical inputs."""

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
