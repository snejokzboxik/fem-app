"""Compute a surface-electrode field grid from a black-white mask image."""

from __future__ import annotations

import argparse
from pathlib import Path

from field_data import save_field_grid_to_npz
from surface_superposition import (
    SurfaceMaskConfig,
    assign_four_electrode_voltages,
    build_voltage_map_from_components,
    compute_surface_field_grid,
    detect_electrode_components,
    load_binary_electrode_mask,
)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Surface-electrode mask Poisson-kernel field demo."
    )
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", default=Path("results/surface_field.npz"), type=Path)
    parser.add_argument("--x-size", default=1.0e-3, type=float)
    parser.add_argument("--y-size", default=1.0e-3, type=float)
    parser.add_argument("--z-max", default=1.0e-3, type=float)
    parser.add_argument("--min-z", default=2.0e-5, type=float)
    parser.add_argument("--nx", default=15, type=int)
    parser.add_argument("--ny", default=15, type=int)
    parser.add_argument("--nz", default=8, type=int)
    parser.add_argument(
        "--grid-mode-xy",
        default="edge_aware",
        choices=["uniform", "center_clustered_tanh", "edge_aware"],
    )
    parser.add_argument(
        "--grid-mode-z",
        default="near_surface_clustered",
        choices=["uniform", "near_surface_clustered"],
    )
    parser.add_argument(
        "--voltage-pattern",
        default="four-electrode",
        choices=["four-electrode"],
    )
    parser.add_argument("--max-active-pixels", default=2500, type=int)
    parser.add_argument("--max-edge-grid-points", default=12, type=int)
    parser.add_argument("--edge-refinement-points", default=3, type=int)
    return parser.parse_args()


def main():
    """Run the mask-to-field-grid demo."""

    args = parse_args()
    config = SurfaceMaskConfig(
        x_size_m=args.x_size,
        y_size_m=args.y_size,
        z_max_m=args.z_max,
        min_z_m=args.min_z,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        grid_mode_xy=args.grid_mode_xy,
        grid_mode_z=args.grid_mode_z,
        max_active_pixels_for_direct_sum=args.max_active_pixels,
        max_edge_grid_points=args.max_edge_grid_points,
        edge_refinement_points_per_edge=args.edge_refinement_points,
        source_description=f"Surface mask demo from {args.mask}",
    )

    mask = load_binary_electrode_mask(args.mask, threshold=config.mask_threshold)
    labels, number_of_components = detect_electrode_components(mask)
    potentials = assign_four_electrode_voltages(labels, mask, config)
    voltage_map = build_voltage_map_from_components(labels, potentials)

    print("Surface mask demo")
    print("-" * 40)
    print(f"mask: {args.mask}")
    print(f"detected electrodes: {number_of_components}")
    print(f"active pixels: {(voltage_map != 0.0).sum()}")
    print(f"voltage pattern: {args.voltage_pattern}")
    print(f"component potentials: {potentials}")
    print(f"grid_mode_xy: {config.grid_mode_xy}")
    print(f"grid_mode_z: {config.grid_mode_z}")
    print(f"edge-aware refinement used: {config.grid_mode_xy == 'edge_aware'}")

    field_grid = compute_surface_field_grid(voltage_map, config, mask=mask)
    save_field_grid_to_npz(field_grid, args.output)

    print(f"output grid shape: {field_grid.electric_field_grid.shape}")
    print(f"saved field grid to: {args.output}")


if __name__ == "__main__":
    main()
