"""Validation and comparison helpers for imported field grids.

These functions are meant as diagnostics for external solver exports.  They do
not change the particle dynamics or the built-in FEM solver; they only check and
compare structured data already loaded into the prototype.
"""

from __future__ import annotations

import numpy as np

from mathieu_analysis import estimate_potential_curvature_near_center


def validate_field_grid(field_grid) -> dict:
    """Return a readable validation report for a field-like grid object."""

    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "grid_shape": None,
        "electric_field_shape": None,
        "potential_shape": None,
        "has_potential": getattr(field_grid, "potential_grid", None) is not None,
        "finite": {},
    }

    x_grid = np.asarray(getattr(field_grid, "x_grid", np.asarray([])))
    y_grid = np.asarray(getattr(field_grid, "y_grid", np.asarray([])))
    z_grid = np.asarray(getattr(field_grid, "z_grid", np.asarray([])))
    electric_field_grid = np.asarray(
        getattr(field_grid, "electric_field_grid", np.asarray([]))
    )
    potential_grid = getattr(field_grid, "potential_grid", None)

    _check_axis("x_grid", x_grid, report)
    _check_axis("y_grid", y_grid, report)
    _check_axis("z_grid", z_grid, report)

    if x_grid.ndim == y_grid.ndim == z_grid.ndim == 1:
        report["grid_shape"] = (len(x_grid), len(y_grid), len(z_grid))
        _check_coordinate_scale(x_grid, y_grid, z_grid, report)

    report["electric_field_shape"] = tuple(electric_field_grid.shape)
    expected_field_shape = (
        len(x_grid),
        len(y_grid),
        len(z_grid),
        3,
    )
    if electric_field_grid.shape != expected_field_shape:
        report["errors"].append(
            "electric_field_grid shape must be "
            f"{expected_field_shape}, got {electric_field_grid.shape}."
        )

    _check_finite("electric_field_grid", electric_field_grid, report)

    if potential_grid is not None:
        potential_grid = np.asarray(potential_grid)
        report["potential_shape"] = tuple(potential_grid.shape)
        expected_potential_shape = expected_field_shape[:3]
        if potential_grid.shape != expected_potential_shape:
            report["errors"].append(
                "potential_grid shape must be "
                f"{expected_potential_shape}, got {potential_grid.shape}."
            )
        _check_finite("potential_grid", potential_grid, report)

    report["valid"] = len(report["errors"]) == 0
    return report


def compare_potential_grids(reference, candidate) -> dict:
    """Compare two potential grids that already live on the same grid."""

    if reference.potential_grid is None or candidate.potential_grid is None:
        raise ValueError("Both grids must contain potential_grid for this comparison.")

    reference_values = np.asarray(reference.potential_grid, dtype=float)
    candidate_values = np.asarray(candidate.potential_grid, dtype=float)
    if reference_values.shape != candidate_values.shape:
        raise ValueError(
            "Potential grid shapes do not match: "
            f"{reference_values.shape} vs {candidate_values.shape}."
        )

    return _scalar_error_metrics(reference_values, candidate_values)


def compare_field_grids(reference, candidate) -> dict:
    """Compare two vector electric-field grids on the same structured grid."""

    reference_field = np.asarray(reference.electric_field_grid, dtype=float)
    candidate_field = np.asarray(candidate.electric_field_grid, dtype=float)
    if reference_field.shape != candidate_field.shape:
        raise ValueError(
            "Electric field grid shapes do not match: "
            f"{reference_field.shape} vs {candidate_field.shape}."
        )

    difference = candidate_field - reference_field
    pointwise_magnitude_error = np.linalg.norm(difference, axis=-1)
    reference_norm = np.linalg.norm(reference_field.ravel())
    diff_norm = np.linalg.norm(difference.ravel())

    component_errors = {}
    for component_index, component_name in enumerate(("Ex", "Ey", "Ez")):
        component_errors[component_name] = _scalar_error_metrics(
            reference_field[..., component_index],
            candidate_field[..., component_index],
        )

    return {
        "mean_abs_error": float(np.mean(pointwise_magnitude_error)),
        "max_abs_error": float(np.max(pointwise_magnitude_error)),
        "relative_l2_error": _relative_error(diff_norm, reference_norm),
        "component_errors": component_errors,
    }


def estimate_symmetry_checks(field_grid_or_fem_result) -> dict:
    """Estimate simple local quadrupole-like symmetry checks near the center."""

    report = {
        "kx": float("nan"),
        "ky": float("nan"),
        "kz": float("nan"),
        "kx_plus_ky": float("nan"),
        "warnings": [],
    }

    if getattr(field_grid_or_fem_result, "potential_grid", None) is None:
        report["warnings"].append(
            "No potential_grid is available, so curvature symmetry checks were skipped."
        )
        return report

    try:
        curvature = estimate_potential_curvature_near_center(field_grid_or_fem_result)
    except Exception as exc:
        report["warnings"].append(f"Could not estimate local curvature: {exc}")
        return report

    kx = float(curvature["kx"])
    ky = float(curvature["ky"])
    kz = float(curvature["kz"])
    kx_plus_ky = kx + ky
    scale_xy = max(abs(kx), abs(ky), 1.0e-30)

    report.update(
        {
            "kx": kx,
            "ky": ky,
            "kz": kz,
            "kx_plus_ky": kx_plus_ky,
            "fit_radius": float(curvature["fit_radius"]),
            "number_of_points_used": int(curvature["number_of_points_used"]),
            "rms_residual": float(curvature["rms_residual"]),
        }
    )

    if abs(kx_plus_ky) > 0.2 * scale_xy:
        report["warnings"].append(
            "abs(kx + ky) is large compared with abs(kx) and abs(ky); "
            "the local field may deviate from an ideal x-y quadrupole."
        )

    if abs(kz) > 0.2 * scale_xy:
        report["warnings"].append(
            "abs(kz) is unexpectedly large compared with the transverse curvature."
        )

    return report


def grids_match_exactly(reference, candidate) -> bool:
    """Return True when x/y/z coordinate arrays match exactly."""

    return (
        np.array_equal(reference.x_grid, candidate.x_grid)
        and np.array_equal(reference.y_grid, candidate.y_grid)
        and np.array_equal(reference.z_grid, candidate.z_grid)
    )


def _check_axis(name: str, values: np.ndarray, report: dict):
    """Validate one coordinate axis and append readable report entries."""

    report["finite"][name] = bool(np.all(np.isfinite(values)))

    if values.ndim != 1:
        report["errors"].append(f"{name} must be a 1D array.")
        return
    if len(values) < 2:
        report["errors"].append(f"{name} must contain at least two points.")
        return
    if not report["finite"][name]:
        report["errors"].append(f"{name} contains non-finite values.")
    if not np.all(np.diff(values) > 0.0):
        report["errors"].append(f"{name} must be strictly increasing.")


def _check_finite(name: str, values: np.ndarray, report: dict):
    """Record finite-value status and append an error if needed."""

    is_finite = bool(np.all(np.isfinite(values)))
    report["finite"][name] = is_finite
    if not is_finite:
        report["errors"].append(f"{name} contains non-finite values.")


def _check_coordinate_scale(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    report: dict,
):
    """Add soft warnings for coordinate scales that often indicate unit mistakes."""

    if min(len(x_grid), len(y_grid), len(z_grid)) < 2:
        return
    spans = np.asarray(
        [
            x_grid[-1] - x_grid[0],
            y_grid[-1] - y_grid[0],
            z_grid[-1] - z_grid[0],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(spans)) or np.any(spans <= 0.0):
        return

    max_span = float(np.max(spans))
    min_span = float(np.min(spans))
    if max_span > 1.0:
        report["warnings"].append(
            "Grid span is larger than 1 m. Check that coordinates were exported "
            "in SI meters, not millimeters or another unit."
        )
    if min_span < 1.0e-12:
        report["warnings"].append(
            "Grid span is extremely small. Check that coordinates are in SI meters."
        )


def _scalar_error_metrics(reference_values: np.ndarray, candidate_values: np.ndarray) -> dict:
    """Return common absolute and relative error metrics for scalar arrays."""

    difference = candidate_values - reference_values
    absolute_difference = np.abs(difference)
    return {
        "mean_abs_error": float(np.mean(absolute_difference)),
        "max_abs_error": float(np.max(absolute_difference)),
        "relative_l2_error": _relative_error(
            np.linalg.norm(difference.ravel()),
            np.linalg.norm(reference_values.ravel()),
        ),
    }


def _relative_error(diff_norm: float, reference_norm: float) -> float:
    """Compute a stable relative error, including zero-reference cases."""

    if reference_norm == 0.0:
        return float(diff_norm)
    return float(diff_norm / reference_norm)
