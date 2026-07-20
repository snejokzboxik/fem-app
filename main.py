"""Run the charged-particle trap prototype from start to finish."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_CONFIG, with_config_overrides
from field_data import (
    load_field_grid_npz,
    load_potential_grid_npz,
    save_fem_field_to_npz,
)
from fem_solver import solve_laplace
from field_interpolation import make_E_at_position, make_E_at_position_from_field_grid
from particle_dynamics import simulate_particle
from visualization import plot_all


def parse_args():
    """Parse optional field-import command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the charged-particle trap prototype."
    )
    field_group = parser.add_mutually_exclusive_group()
    field_group.add_argument(
        "--field-grid",
        type=Path,
        help="Load a .npz file containing x_grid, y_grid, z_grid, electric_field_grid.",
    )
    field_group.add_argument(
        "--potential-grid",
        type=Path,
        help="Load a .npz file containing x_grid, y_grid, z_grid, potential_grid.",
    )
    parser.add_argument(
        "--export-fem-field",
        type=Path,
        help="Save the built-in normalized 1 V FEM field to this .npz file.",
    )
    return parser.parse_args()


def config_with_grid_domain(config, field_grid):
    """Use the imported grid extents as the rectangular simulation domain."""

    domain_size = (
        float(field_grid.x_grid[-1] - field_grid.x_grid[0]),
        float(field_grid.y_grid[-1] - field_grid.y_grid[0]),
        float(field_grid.z_grid[-1] - field_grid.z_grid[0]),
    )
    return with_config_overrides(config, domain_size=domain_size)


def load_field_source(args, config):
    """Return field data, interpolator function, and possibly updated config."""

    if args.field_grid is not None:
        print(f"Loading electric-field grid from: {args.field_grid}")
        field_grid = load_field_grid_npz(args.field_grid)
        config = config_with_grid_domain(config, field_grid)
        return field_grid, make_E_at_position_from_field_grid(field_grid), config

    if args.potential_grid is not None:
        print(f"Loading potential grid from: {args.potential_grid}")
        field_grid = load_potential_grid_npz(args.potential_grid)
        config = config_with_grid_domain(config, field_grid)
        return field_grid, make_E_at_position_from_field_grid(field_grid), config

    print("Solving normalized 1 V 3D electrostatic Laplace problem...")
    fem_result = solve_laplace(config)
    if args.export_fem_field is not None:
        save_fem_field_to_npz(fem_result, args.export_fem_field)
        print(f"Saved built-in FEM field to: {args.export_fem_field}")

    print("Building electric-field interpolator...")
    return fem_result, make_E_at_position(fem_result), config


def main():
    """Solve or load the field, integrate particle motion, and show plots."""

    args = parse_args()
    config = DEFAULT_CONFIG

    field_result, E_function, config = load_field_source(args, config)

    print("Integrating particle trajectory...")
    particle_result = simulate_particle(E_function, config)

    print(f"Computed {len(particle_result.t)} time samples.")
    if particle_result.left_domain:
        print(f"Particle left the domain at t = {particle_result.exit_time:.3e} s.")
    else:
        print("Particle stayed inside the domain for the full simulation time.")

    print(f"Saving plots to: {config.output_dir}/")
    plot_all(
        field_result,
        particle_result,
        config=config,
        show=True,
        output_dir=config.output_dir,
    )


if __name__ == "__main__":
    main()
