"""Load and save structured electric-field or potential grids.

External tools such as Wolfram, COMSOL, or another FEM solver can export data
onto a regular x-y-z grid.  This module keeps the expected format deliberately
simple so it is easy to inspect with NumPy:

Field grid ``.npz``:
    x_grid, y_grid, z_grid, electric_field_grid

Potential grid ``.npz``:
    x_grid, y_grid, z_grid, potential_grid

All coordinates should be in SI meters.  Electric fields should be in V/m.
Potentials should be in volts.  In the particle dynamics, the loaded field is
treated as the same kind of base field as the built-in normalized FEM field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FieldGrid:
    """Structured field data used for interpolation and plotting."""

    x_grid: np.ndarray
    y_grid: np.ndarray
    z_grid: np.ndarray
    electric_field_grid: np.ndarray
    potential_grid: np.ndarray | None = None
    source_description: str | None = None


def save_fem_field_to_npz(fem_result, path: str | Path):
    """Save the current built-in FEM result as a reference field-grid file."""

    field_grid = FieldGrid(
        x_grid=np.asarray(fem_result.x_grid, dtype=float),
        y_grid=np.asarray(fem_result.y_grid, dtype=float),
        z_grid=np.asarray(fem_result.z_grid, dtype=float),
        electric_field_grid=np.asarray(fem_result.electric_field_grid, dtype=float),
        potential_grid=np.asarray(fem_result.potential_grid, dtype=float),
        source_description="Built-in normalized 1 V FEM field",
    )
    save_field_grid_to_npz(field_grid, path)


def save_field_grid_to_npz(field_grid: FieldGrid, path: str | Path):
    """Save any FieldGrid-compatible object using the documented .npz format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "x_grid": np.asarray(field_grid.x_grid, dtype=float),
        "y_grid": np.asarray(field_grid.y_grid, dtype=float),
        "z_grid": np.asarray(field_grid.z_grid, dtype=float),
        "electric_field_grid": np.asarray(field_grid.electric_field_grid, dtype=float),
        "source_description": field_grid.source_description or "FieldGrid export",
    }
    if field_grid.potential_grid is not None:
        arrays["potential_grid"] = np.asarray(field_grid.potential_grid, dtype=float)
    np.savez_compressed(path, **arrays)


def load_field_grid_npz(path: str | Path) -> FieldGrid:
    """Load a structured electric-field grid from a ``.npz`` file."""

    source = _prepare_npz_source(path)
    _rewind_if_possible(path)
    with np.load(path, allow_pickle=False) as data:
        _require_keys(data, source, ["x_grid", "y_grid", "z_grid", "electric_field_grid"])
        potential_grid = (
            np.asarray(data["potential_grid"], dtype=float)
            if "potential_grid" in data.files
            else None
        )
        field_grid = FieldGrid(
            x_grid=np.asarray(data["x_grid"], dtype=float),
            y_grid=np.asarray(data["y_grid"], dtype=float),
            z_grid=np.asarray(data["z_grid"], dtype=float),
            electric_field_grid=np.asarray(data["electric_field_grid"], dtype=float),
            potential_grid=potential_grid,
            source_description=_read_optional_description(data, source),
        )

    validate_field_grid(field_grid)
    return field_grid


def load_potential_grid_npz(path: str | Path) -> FieldGrid:
    """Load a potential grid and compute its electric field by finite differences."""

    source = _prepare_npz_source(path)
    _rewind_if_possible(path)
    with np.load(path, allow_pickle=False) as data:
        _require_keys(data, source, ["x_grid", "y_grid", "z_grid", "potential_grid"])
        x_grid = np.asarray(data["x_grid"], dtype=float)
        y_grid = np.asarray(data["y_grid"], dtype=float)
        z_grid = np.asarray(data["z_grid"], dtype=float)
        potential_grid = np.asarray(data["potential_grid"], dtype=float)
        electric_field_grid = compute_field_from_potential_grid(
            potential_grid,
            x_grid,
            y_grid,
            z_grid,
        )
        field_grid = FieldGrid(
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=z_grid,
            electric_field_grid=electric_field_grid,
            potential_grid=potential_grid,
            source_description=_read_optional_description(data, source),
        )

    validate_field_grid(field_grid)
    return field_grid


def compute_field_from_potential_grid(
    potential_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
) -> np.ndarray:
    """Compute ``E = -grad(phi)`` on a regular grid."""

    potential_grid = np.asarray(potential_grid, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)

    expected_shape = (len(x_grid), len(y_grid), len(z_grid))
    if potential_grid.shape != expected_shape:
        raise ValueError(
            "potential_grid shape must be "
            f"{expected_shape}, got {potential_grid.shape}."
        )

    edge_order = 2 if min(potential_grid.shape) >= 3 else 1
    dphi_dx, dphi_dy, dphi_dz = np.gradient(
        potential_grid,
        x_grid,
        y_grid,
        z_grid,
        edge_order=edge_order,
    )
    return -np.stack((dphi_dx, dphi_dy, dphi_dz), axis=-1)


def validate_field_grid(field_grid: FieldGrid):
    """Check grid dimensions and field shape early, with readable errors."""

    _validate_axis("x_grid", field_grid.x_grid)
    _validate_axis("y_grid", field_grid.y_grid)
    _validate_axis("z_grid", field_grid.z_grid)

    expected_field_shape = (
        len(field_grid.x_grid),
        len(field_grid.y_grid),
        len(field_grid.z_grid),
        3,
    )
    if field_grid.electric_field_grid.shape != expected_field_shape:
        raise ValueError(
            "electric_field_grid shape must be "
            f"{expected_field_shape}, got {field_grid.electric_field_grid.shape}."
        )

    if field_grid.potential_grid is not None:
        expected_potential_shape = expected_field_shape[:3]
        if field_grid.potential_grid.shape != expected_potential_shape:
            raise ValueError(
                "potential_grid shape must be "
                f"{expected_potential_shape}, got {field_grid.potential_grid.shape}."
            )


def _validate_axis(name: str, values: np.ndarray):
    """Validate one coordinate axis."""

    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two points.")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")


def _prepare_npz_source(path_or_file) -> str:
    """Return a readable source label for a path or uploaded file object."""

    if isinstance(path_or_file, (str, Path)):
        return str(path_or_file)
    return getattr(path_or_file, "name", "uploaded .npz file")


def _rewind_if_possible(path_or_file):
    """Reset uploaded/file-like objects before passing them to ``np.load``."""

    if hasattr(path_or_file, "seek"):
        path_or_file.seek(0)


def _require_keys(data, source: str, keys: list[str]):
    """Raise a readable error when an expected NPZ array is missing."""

    missing = [key for key in keys if key not in data.files]
    if missing:
        raise KeyError(f"{source} is missing required arrays: {', '.join(missing)}")


def _read_optional_description(data, source: str) -> str:
    """Read a short source description if present, otherwise use the filename."""

    if "source_description" not in data.files:
        return f"Loaded from {source}"
    return str(np.asarray(data["source_description"]).item())
