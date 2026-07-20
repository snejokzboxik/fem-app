"""Generate a tiny example electric-field grid in the documented NPZ format.

Run from the project root:

    python examples/generate_example_field_npz.py

The synthetic potential is not a validated trap geometry.  It is only a compact
reference file showing how external solvers should arrange their arrays.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from field_data import compute_field_from_potential_grid


def main():
    """Save ``examples/example_field_grid.npz``."""

    x_grid = np.linspace(-5.0e-4, 5.0e-4, 9)
    y_grid = np.linspace(-5.0e-4, 5.0e-4, 9)
    z_grid = np.linspace(-5.0e-4, 5.0e-4, 7)

    x_mesh, y_mesh, _z_mesh = np.meshgrid(
        x_grid,
        y_grid,
        z_grid,
        indexing="ij",
    )

    r0 = 5.0e-4
    potential_grid = (x_mesh**2 - y_mesh**2) / (2.0 * r0**2)
    electric_field_grid = compute_field_from_potential_grid(
        potential_grid,
        x_grid,
        y_grid,
        z_grid,
    )

    output_path = Path(__file__).with_name("example_field_grid.npz")
    np.savez_compressed(
        output_path,
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        electric_field_grid=electric_field_grid,
        potential_grid=potential_grid,
        source_description="Synthetic example quadrupole-like field grid",
    )
    print(f"Saved example field grid to: {output_path}")


if __name__ == "__main__":
    main()
