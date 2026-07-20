"""Image-based planar surface-electrode fields by Poisson-kernel summation.

This module implements a numerical quadrature of the analytic Dirichlet
half-space solution.  It is not a 3D FEM solver.

The electrode image is the geometry:

* white/light connected regions are planar electrodes at z = 0,
* black/dark pixels are 0 V background in this first MVP,
* each connected white component can be assigned its own voltage.

For z > 0, the upper-half-space potential is

    phi(x, y, z) = integral V(x', y') K(x-x', y-y', z) dx' dy'

where

    K(dx, dy, z) = (1/(2*pi)) * z / (dx^2 + dy^2 + z^2)^(3/2).

The integral is approximated by summing over image pixel centers.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from PIL import Image
from scipy import ndimage

from field_data import FieldGrid


@dataclass(frozen=True)
class SurfaceMaskConfig:
    """Settings for image-mask Poisson-kernel field computation."""

    x_size_m: float = 1.0e-3
    y_size_m: float = 1.0e-3
    z_max_m: float = 1.0e-3
    min_z_m: float = 2.0e-5
    nx: int = 31
    ny: int = 31
    nz: int = 21
    mask_threshold: float = 0.5
    grid_mode_xy: str = "uniform"
    grid_mode_z: str = "near_surface_clustered"
    cluster_strength_xy: float = 2.5
    cluster_strength_z: float = 2.5
    edge_refinement_enabled: bool = True
    edge_refinement_radius_m: float = 4.0e-5
    edge_refinement_points_per_edge: int = 5
    max_edge_grid_points: int = 200
    min_grid_spacing_m: float = 2.0e-6
    downsample_mask_for_computation: bool = True
    max_computational_mask_size: int = 192
    max_active_pixels_for_direct_sum: int = 10000
    chunk_size_points: int = 1024
    active_pixel_chunk_size: int = 1024
    source_description: str | None = None


def load_binary_electrode_mask(image_path_or_file, threshold: float = 0.5) -> np.ndarray:
    """Load a PNG/JPG image and convert light pixels to a boolean electrode mask."""

    _rewind_if_possible(image_path_or_file)
    image = Image.open(image_path_or_file).convert("L")
    grayscale = np.asarray(image, dtype=float) / 255.0
    return grayscale >= threshold


def downsample_binary_mask_for_computation(
    mask: np.ndarray,
    max_size: int,
    method: str = "area_threshold",
) -> tuple[np.ndarray, dict]:
    """Return a smaller boolean mask for numerical quadrature.

    The physical mask size is not changed here.  Only the number of pixels used
    in the Poisson-kernel direct sum is reduced.  Area-like resampling followed
    by a threshold keeps the output boolean while preserving smooth electrode
    shapes better than nearest-neighbor resizing.
    """

    if method != "area_threshold":
        raise ValueError("Only method='area_threshold' is currently supported.")

    original_mask = np.asarray(mask, dtype=bool)
    if original_mask.ndim != 2:
        raise ValueError("mask must be a 2D boolean array.")

    max_size = int(max_size)
    if max_size < 1:
        raise ValueError("max_size must be at least 1.")

    original_rows, original_columns = original_mask.shape
    original_longest_side = max(original_rows, original_columns)
    if original_longest_side <= max_size:
        computational_mask = original_mask.copy()
        scale = 1.0
    else:
        scale = max_size / float(original_longest_side)
        new_columns = max(1, int(round(original_columns * scale)))
        new_rows = max(1, int(round(original_rows * scale)))
        image = Image.fromarray(original_mask.astype(np.float32))
        resized = image.resize(
            (new_columns, new_rows),
            resample=_box_resampling_filter(),
        )
        computational_mask = np.asarray(resized, dtype=float) >= 0.5

    metadata = _build_downsampling_metadata(original_mask, computational_mask, scale)
    return computational_mask.astype(bool), metadata


def prepare_surface_voltage_map_for_computation(
    voltage_map: np.ndarray,
    config: SurfaceMaskConfig,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Prepare the voltage map and mask actually used by the direct sum."""

    voltage_map = np.asarray(voltage_map, dtype=float)
    if voltage_map.ndim != 2:
        raise ValueError("voltage_map must be a 2D array.")

    if mask is None:
        mask = voltage_map != 0.0
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != voltage_map.shape:
        raise ValueError("mask and voltage_map must have the same shape.")

    if config.downsample_mask_for_computation:
        computational_mask, metadata = downsample_binary_mask_for_computation(
            mask,
            config.max_computational_mask_size,
        )
    else:
        computational_mask = mask.copy()
        metadata = _build_downsampling_metadata(mask, computational_mask, scale=1.0)

    computational_voltage_map = _resize_voltage_map_to_shape(
        voltage_map,
        computational_mask.shape,
    )
    computational_voltage_map[~computational_mask] = 0.0
    computational_voltage_map[np.isclose(computational_voltage_map, 0.0, atol=1.0e-15)] = 0.0

    metadata["active_voltage_pixels_after"] = int(
        np.count_nonzero(computational_voltage_map != 0.0)
    )
    metadata["active_voltage_fraction_after"] = float(
        metadata["active_voltage_pixels_after"] / computational_voltage_map.size
    )
    return computational_voltage_map, computational_mask, metadata


def detect_electrode_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected white electrode regions.

    Background is label 0.  Electrode components are labels 1..n.  Eight-neighbor
    connectivity is used so diagonally connected white pixels belong to the
    same electrode.
    """

    structure = np.ones((3, 3), dtype=int)
    labels, number_of_components = ndimage.label(np.asarray(mask, dtype=bool), structure)
    return labels.astype(int), int(number_of_components)


def build_voltage_map_from_components(
    component_labels: np.ndarray,
    electrode_potentials: dict[int, float],
) -> np.ndarray:
    """Assign voltages to connected electrode components."""

    voltage_map = np.zeros_like(component_labels, dtype=float)
    for component_id in range(1, int(np.max(component_labels)) + 1):
        voltage_map[component_labels == component_id] = float(
            electrode_potentials.get(component_id, 0.0)
        )
    return voltage_map


def assign_four_electrode_voltages(
    component_labels: np.ndarray,
    mask: np.ndarray,
    config: SurfaceMaskConfig,
    positive_voltage: float = 1.0,
    negative_voltage: float = -1.0,
) -> dict[int, float]:
    """Assign a simple planar quadrupole-like voltage pattern by component centroid.

    Components farther left/right than up/down get ``positive_voltage``.
    Components farther up/down than left/right get ``negative_voltage``.
    This is only a convenient demo pattern for four-pad masks.
    """

    x_centers, y_centers, _pixel_area = make_mask_pixel_coordinates(mask, config)
    potentials = {}
    for component_id in range(1, int(np.max(component_labels)) + 1):
        rows, columns = np.nonzero(component_labels == component_id)
        if len(rows) == 0:
            continue
        centroid_x = float(np.mean(x_centers[columns]))
        centroid_y = float(np.mean(y_centers[rows]))
        if abs(centroid_x) >= abs(centroid_y):
            potentials[component_id] = float(positive_voltage)
        else:
            potentials[component_id] = float(negative_voltage)
    return potentials


def detect_electrode_edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return electrode edge pixels and their image row/column coordinates."""

    mask = np.asarray(mask, dtype=bool)
    eroded = ndimage.binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    edge_mask = mask & ~eroded
    edge_coordinates = np.argwhere(edge_mask)
    return edge_mask, edge_coordinates


def make_rectilinear_grid_1d(
    size_m: float,
    n: int,
    mode: str = "uniform",
    cluster_strength: float = 2.5,
) -> np.ndarray:
    """Create a strictly increasing 1D observation grid."""

    if n < 2:
        raise ValueError("n must be at least 2.")
    if size_m <= 0.0:
        raise ValueError("size_m must be positive.")

    if mode == "uniform":
        grid = np.linspace(-0.5 * size_m, 0.5 * size_m, n)
    elif mode == "center_clustered_tanh":
        s = np.linspace(-1.0, 1.0, n)
        u = np.sign(s) * np.abs(s) ** 3
        strength = max(float(cluster_strength), 1.0e-12)
        grid = 0.5 * size_m * np.tanh(strength * u) / np.tanh(strength)
    else:
        raise ValueError(f"Unknown x/y grid mode: {mode}")

    return _strictly_increasing_unique(grid)


def make_z_grid(
    z_max_m: float,
    min_z_m: float,
    nz: int,
    mode: str = "uniform",
    cluster_strength: float = 2.5,
) -> np.ndarray:
    """Create a z grid above the electrode plane without including z = 0."""

    if min_z_m <= 0.0:
        raise ValueError("min_z_m must be > 0 to avoid the z = 0 singularity.")
    if z_max_m <= min_z_m:
        raise ValueError("z_max_m must be larger than min_z_m.")
    if nz < 2:
        raise ValueError("nz must be at least 2.")

    if mode == "uniform":
        grid = np.linspace(min_z_m, z_max_m, nz)
    elif mode == "near_surface_clustered":
        s = np.linspace(0.0, 1.0, nz)
        strength = max(float(cluster_strength), 1.0e-12)
        normalized = np.expm1(strength * s) / np.expm1(strength)
        grid = min_z_m + (z_max_m - min_z_m) * normalized
    else:
        raise ValueError(f"Unknown z grid mode: {mode}")

    return _strictly_increasing_unique(grid)


def make_mask_pixel_coordinates(
    mask: np.ndarray,
    config: SurfaceMaskConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map image pixel centers to physical coordinates on z = 0.

    Convention:
    ``mask[row, column]`` follows image coordinates with row 0 at the top.
    The physical x coordinate increases from left to right.  The physical y
    coordinate increases upward, so image row 0 maps to positive y.
    """

    rows, columns = np.asarray(mask).shape
    pixel_width = config.x_size_m / columns
    pixel_height = config.y_size_m / rows

    x_centers = -0.5 * config.x_size_m + (np.arange(columns) + 0.5) * pixel_width
    y_centers = 0.5 * config.y_size_m - (np.arange(rows) + 0.5) * pixel_height
    pixel_area = pixel_width * pixel_height
    return x_centers, y_centers, float(pixel_area)


def make_edge_aware_xy_grid(
    mask: np.ndarray,
    config: SurfaceMaskConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a non-uniform rectilinear grid with extra points near edges."""

    base_x = make_rectilinear_grid_1d(config.x_size_m, config.nx, mode="uniform")
    base_y = make_rectilinear_grid_1d(config.y_size_m, config.ny, mode="uniform")
    center_x = make_rectilinear_grid_1d(
        config.x_size_m,
        config.nx,
        mode="center_clustered_tanh",
        cluster_strength=config.cluster_strength_xy,
    )
    center_y = make_rectilinear_grid_1d(
        config.y_size_m,
        config.ny,
        mode="center_clustered_tanh",
        cluster_strength=config.cluster_strength_xy,
    )

    if not config.edge_refinement_enabled:
        return (
            _merge_grid_points([base_x, center_x], config.min_grid_spacing_m),
            _merge_grid_points([base_y, center_y], config.min_grid_spacing_m),
        )

    _edge_mask, edge_coordinates = detect_electrode_edges(mask)
    if len(edge_coordinates) == 0:
        return (
            _merge_grid_points([base_x, center_x], config.min_grid_spacing_m),
            _merge_grid_points([base_y, center_y], config.min_grid_spacing_m),
        )

    if len(edge_coordinates) > config.max_edge_grid_points:
        warnings.warn(
            "Too many edge pixels for edge-aware grid; deterministic subsampling "
            f"to {config.max_edge_grid_points} edge coordinates.",
            RuntimeWarning,
        )
        indices = np.linspace(
            0,
            len(edge_coordinates) - 1,
            config.max_edge_grid_points,
            dtype=int,
        )
        edge_coordinates = edge_coordinates[indices]

    x_centers, y_centers, _pixel_area = make_mask_pixel_coordinates(mask, config)
    edge_rows = edge_coordinates[:, 0]
    edge_columns = edge_coordinates[:, 1]
    edge_x = x_centers[edge_columns]
    edge_y = y_centers[edge_rows]

    offsets = np.linspace(
        -config.edge_refinement_radius_m,
        config.edge_refinement_radius_m,
        max(config.edge_refinement_points_per_edge, 1),
    )
    refined_x = (edge_x[:, None] + offsets[None, :]).ravel()
    refined_y = (edge_y[:, None] + offsets[None, :]).ravel()

    x_grid = _merge_grid_points([base_x, center_x, refined_x], config.min_grid_spacing_m)
    y_grid = _merge_grid_points([base_y, center_y, refined_y], config.min_grid_spacing_m)
    x_grid = np.clip(x_grid, -0.5 * config.x_size_m, 0.5 * config.x_size_m)
    y_grid = np.clip(y_grid, -0.5 * config.y_size_m, 0.5 * config.y_size_m)
    return (
        _merge_grid_points([x_grid], config.min_grid_spacing_m),
        _merge_grid_points([y_grid], config.min_grid_spacing_m),
    )


def make_surface_observation_grid(
    config: SurfaceMaskConfig,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the rectilinear observation grid used to tabulate phi and E."""

    if config.grid_mode_xy == "edge_aware":
        if mask is None:
            raise ValueError("mask is required when grid_mode_xy='edge_aware'.")
        x_grid, y_grid = make_edge_aware_xy_grid(mask, config)
    else:
        x_grid = make_rectilinear_grid_1d(
            config.x_size_m,
            config.nx,
            mode=config.grid_mode_xy,
            cluster_strength=config.cluster_strength_xy,
        )
        y_grid = make_rectilinear_grid_1d(
            config.y_size_m,
            config.ny,
            mode=config.grid_mode_xy,
            cluster_strength=config.cluster_strength_xy,
        )

    z_grid = make_z_grid(
        config.z_max_m,
        config.min_z_m,
        config.nz,
        mode=config.grid_mode_z,
        cluster_strength=config.cluster_strength_z,
    )
    return x_grid, y_grid, z_grid


def compute_potential_from_surface_voltage_map(
    voltage_map: np.ndarray,
    config: SurfaceMaskConfig,
    mask: np.ndarray | None = None,
    return_metadata: bool = False,
    active_pixel_chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute potential by direct Poisson-kernel pixel superposition."""

    voltage_map, mask, metadata = prepare_surface_voltage_map_for_computation(
        voltage_map,
        config,
        mask=mask,
    )

    x_grid, y_grid, z_grid = make_surface_observation_grid(config, mask=mask)
    x_pixels, y_pixels, pixel_area = make_mask_pixel_coordinates(voltage_map, config)
    metadata["pixel_width_m"] = float(config.x_size_m / voltage_map.shape[1])
    metadata["pixel_height_m"] = float(config.y_size_m / voltage_map.shape[0])
    metadata["pixel_area_m2"] = float(pixel_area)

    active_rows, active_columns = np.nonzero(voltage_map != 0.0)
    active_count = len(active_rows)
    if active_count == 0:
        result = (
            np.zeros((len(x_grid), len(y_grid), len(z_grid)), dtype=float),
            x_grid,
            y_grid,
            z_grid,
        )
        return (*result, metadata) if return_metadata else result

    if active_count > config.max_active_pixels_for_direct_sum:
        raise ValueError(
            "Слишком много активных пикселей электродов для прямого "
            "суммирования. Уменьшите computational mask size, упростите "
            "маску или осторожно увеличьте max_active_pixels_for_direct_sum. "
            f"active_pixels={active_count}, "
            f"limit={config.max_active_pixels_for_direct_sum}, "
            f"computational_mask_shape={voltage_map.shape}."
        )

    active_x = x_pixels[active_columns]
    active_y = y_pixels[active_rows]
    weighted_active_voltage = voltage_map[active_rows, active_columns] * pixel_area

    obs_x, obs_y, obs_z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    observation_points = np.column_stack(
        (obs_x.ravel(), obs_y.ravel(), obs_z.ravel())
    )
    potential_flat = np.zeros(len(observation_points), dtype=float)

    point_chunk_size = max(1, int(config.chunk_size_points))
    pixel_chunk_size = max(
        1,
        int(
            active_pixel_chunk_size
            if active_pixel_chunk_size is not None
            else config.active_pixel_chunk_size
        ),
    )
    metadata["active_pixel_chunk_size"] = int(pixel_chunk_size)
    metadata["observation_point_chunk_size"] = int(point_chunk_size)
    for start in range(0, len(observation_points), point_chunk_size):
        stop = min(start + point_chunk_size, len(observation_points))
        observation_chunk = observation_points[start:stop]
        chunk_potential = np.zeros(len(observation_chunk), dtype=float)
        z = observation_chunk[:, 2, None]
        for pixel_start in range(0, active_count, pixel_chunk_size):
            pixel_stop = min(pixel_start + pixel_chunk_size, active_count)
            dx = observation_chunk[:, 0, None] - active_x[None, pixel_start:pixel_stop]
            dy = observation_chunk[:, 1, None] - active_y[None, pixel_start:pixel_stop]
            denominator = (dx**2 + dy**2 + z**2) ** 1.5
            kernel = (1.0 / (2.0 * np.pi)) * z / denominator
            chunk_potential += kernel @ weighted_active_voltage[pixel_start:pixel_stop]
        potential_flat[start:stop] = chunk_potential

    potential_grid = potential_flat.reshape((len(x_grid), len(y_grid), len(z_grid)))
    if not np.all(np.isfinite(potential_grid)):
        raise FloatingPointError("Surface potential computation produced non-finite values.")
    result = (potential_grid, x_grid, y_grid, z_grid)
    return (*result, metadata) if return_metadata else result


def compute_field_from_potential_grid_nonuniform(
    potential_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
) -> np.ndarray:
    """Compute electric field ``E = -grad(phi)`` on a non-uniform grid."""

    edge_order = 2 if min(potential_grid.shape) >= 3 else 1
    dphi_dx, dphi_dy, dphi_dz = np.gradient(
        potential_grid,
        x_grid,
        y_grid,
        z_grid,
        edge_order=edge_order,
    )
    return -np.stack((dphi_dx, dphi_dy, dphi_dz), axis=-1)


def compute_surface_field_grid(
    voltage_map: np.ndarray,
    config: SurfaceMaskConfig,
    mask: np.ndarray | None = None,
) -> FieldGrid:
    """Compute a FieldGrid from a planar surface-electrode voltage map."""

    (
        potential_grid,
        x_grid,
        y_grid,
        z_grid,
        metadata,
    ) = compute_potential_from_surface_voltage_map(
        voltage_map,
        config,
        mask=mask,
        return_metadata=True,
    )
    electric_field_grid = compute_field_from_potential_grid_nonuniform(
        potential_grid,
        x_grid,
        y_grid,
        z_grid,
    )
    if not np.all(np.isfinite(electric_field_grid)):
        raise FloatingPointError("Surface field computation produced non-finite values.")

    field_grid = FieldGrid(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        electric_field_grid=electric_field_grid,
        potential_grid=potential_grid,
        source_description=(
            config.source_description
            or "Surface electrode mask, Dirichlet half-space Poisson kernel"
        ),
    )
    field_grid.surface_computation_metadata = metadata
    return field_grid


def _build_downsampling_metadata(
    original_mask: np.ndarray,
    computational_mask: np.ndarray,
    scale: float,
) -> dict:
    """Build a readable report for mask downsampling."""

    original_active_pixels = int(np.count_nonzero(original_mask))
    computational_active_pixels = int(np.count_nonzero(computational_mask))
    original_size = int(original_mask.size)
    computational_size = int(computational_mask.size)

    return {
        "original_shape": tuple(int(v) for v in original_mask.shape),
        "computational_shape": tuple(int(v) for v in computational_mask.shape),
        "scale": float(scale),
        "downsample_factor": float(1.0 / scale) if scale > 0.0 else np.inf,
        "active_pixels_before": original_active_pixels,
        "active_pixels_after": computational_active_pixels,
        "active_fraction_before": float(original_active_pixels / original_size),
        "active_fraction_after": float(
            computational_active_pixels / computational_size
        ),
    }


def _resize_voltage_map_to_shape(
    voltage_map: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Resize a voltage map with area-like averaging."""

    if tuple(voltage_map.shape) == tuple(target_shape):
        return np.asarray(voltage_map, dtype=float).copy()

    target_rows, target_columns = target_shape
    image = Image.fromarray(np.asarray(voltage_map, dtype=np.float32))
    resized = image.resize(
        (int(target_columns), int(target_rows)),
        resample=_box_resampling_filter(),
    )
    return np.asarray(resized, dtype=float)


def _box_resampling_filter():
    """Return the Pillow BOX resampling enum across Pillow versions."""

    if hasattr(Image, "Resampling"):
        return Image.Resampling.BOX
    return Image.BOX


def _strictly_increasing_unique(values: np.ndarray) -> np.ndarray:
    """Sort, deduplicate, and verify a 1D grid."""

    grid = np.unique(np.asarray(values, dtype=float))
    if len(grid) < 2:
        raise ValueError("Grid must contain at least two unique points.")
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError("Grid points must be strictly increasing.")
    return grid


def _merge_grid_points(point_sets: list[np.ndarray], min_spacing_m: float) -> np.ndarray:
    """Merge point sets and enforce a minimum spacing."""

    points = np.concatenate([np.asarray(values, dtype=float).ravel() for values in point_sets])
    points = np.sort(points[np.isfinite(points)])
    if len(points) == 0:
        raise ValueError("Cannot build a grid from zero points.")

    merged = [float(points[0])]
    min_spacing_m = max(float(min_spacing_m), 0.0)
    for point in points[1:]:
        if point - merged[-1] >= min_spacing_m:
            merged.append(float(point))

    grid = np.asarray(merged, dtype=float)
    if len(grid) < 2:
        raise ValueError("Merged grid has fewer than two points.")
    return grid


def _rewind_if_possible(path_or_file):
    """Reset uploaded/file-like objects before image loading."""

    if hasattr(path_or_file, "seek"):
        path_or_file.seek(0)
