"""Interpolation of the electric field at arbitrary particle positions."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from config import DEFAULT_CONFIG, SimulationConfig
from field_data import FieldGrid
from fem_solver import FemResult


def build_field_interpolator(
    fem_result: FemResult,
    outside_value: float = np.nan,
) -> RegularGridInterpolator:
    """Create an interpolator for the electric field vector.

    The returned SciPy object accepts positions with shape ``(n, 3)`` and
    returns electric-field vectors with shape ``(n, 3)``.
    """

    return build_field_interpolator_from_arrays(
        fem_result.x_grid,
        fem_result.y_grid,
        fem_result.z_grid,
        fem_result.electric_field_grid,
        outside_value=outside_value,
    )


def build_field_interpolator_from_field_grid(
    field_grid: FieldGrid,
    outside_value: float = np.nan,
) -> RegularGridInterpolator:
    """Create an electric-field interpolator from an imported ``FieldGrid``."""

    return build_field_interpolator_from_arrays(
        field_grid.x_grid,
        field_grid.y_grid,
        field_grid.z_grid,
        field_grid.electric_field_grid,
        outside_value=outside_value,
    )


def build_field_interpolator_from_arrays(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    electric_field_grid: np.ndarray,
    outside_value: float = np.nan,
) -> RegularGridInterpolator:
    """Create a vector-field interpolator from structured grid arrays."""

    return RegularGridInterpolator(
        (x_grid, y_grid, z_grid),
        electric_field_grid,
        bounds_error=False,
        fill_value=outside_value,
    )


def E_at_position(
    position: np.ndarray | tuple[float, float, float],
    field_interpolator: RegularGridInterpolator,
) -> np.ndarray:
    """Return the electric field vector at one 3D position.

    If the point is outside the computational domain, the default interpolator
    returns ``[nan, nan, nan]``.  The dynamics module uses this to stop the
    simulation cleanly instead of crashing.
    """

    point = np.asarray(position, dtype=float)
    if point.shape != (3,):
        raise ValueError("position must be a 3-component vector [x, y, z].")

    value = field_interpolator(point.reshape(1, 3))[0]
    return np.asarray(value, dtype=float)


def make_E_at_position(fem_result: FemResult):
    """Return a small convenience function E(position)."""

    interpolator = build_field_interpolator(fem_result)

    def field_function(position):
        return E_at_position(position, interpolator)

    return field_function


def make_E_at_position_from_field_grid(field_grid: FieldGrid):
    """Return a small convenience function E(position) for imported fields."""

    interpolator = build_field_interpolator_from_field_grid(field_grid)

    def field_function(position):
        return E_at_position(position, interpolator)

    return field_function


def inside_domain(
    position: np.ndarray | tuple[float, float, float],
    config: SimulationConfig = DEFAULT_CONFIG,
) -> bool:
    """Check whether a position lies inside the rectangular domain."""

    point = np.asarray(position, dtype=float)
    half_size = 0.5 * np.asarray(config.domain_size, dtype=float)
    return bool(np.all(np.abs(point) <= half_size))
