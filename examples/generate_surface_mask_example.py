"""Generate small example surface-electrode mask images and one field grid.

Run from the project root:

    python examples/generate_surface_mask_example.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from field_data import save_field_grid_to_npz
from surface_superposition import (
    SurfaceMaskConfig,
    assign_four_electrode_voltages,
    build_voltage_map_from_components,
    compute_surface_field_grid,
    detect_electrode_components,
    load_binary_electrode_mask,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "surface_masks"


def draw_simple_four_electrode_mask(path: Path, image_size: int = 64):
    """Draw four separated white pads on a black background."""

    image = Image.new("L", (image_size, image_size), color=0)
    draw = ImageDraw.Draw(image)
    pad_w = image_size // 5
    pad_h = image_size // 4
    gap = image_size // 10
    center = image_size // 2

    # Left and right pads.
    draw.rectangle(
        [gap, center - pad_h // 2, gap + pad_w, center + pad_h // 2],
        fill=255,
    )
    draw.rectangle(
        [image_size - gap - pad_w, center - pad_h // 2, image_size - gap, center + pad_h // 2],
        fill=255,
    )

    # Top and bottom pads.
    draw.rectangle(
        [center - pad_h // 2, gap, center + pad_h // 2, gap + pad_w],
        fill=255,
    )
    draw.rectangle(
        [center - pad_h // 2, image_size - gap - pad_w, center + pad_h // 2, image_size - gap],
        fill=255,
    )
    image.save(path)


def draw_parabolic_electrode_mask(path: Path, image_size: int = 96):
    """Draw two curved parabolic electrodes as a mask example."""

    yy, xx = np.mgrid[0:image_size, 0:image_size]
    x = (xx - image_size / 2) / (image_size / 2)
    y = (image_size / 2 - yy) / (image_size / 2)

    upper_curve = y > 0.15 + 0.55 * x**2
    lower_curve = y < -0.15 - 0.55 * x**2
    side_cut = np.abs(x) < 0.85
    mask = (upper_curve | lower_curve) & side_cut

    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image.save(path)


def generate_example_field(mask_path: Path, output_path: Path):
    """Compute a small field grid for the four-electrode mask."""

    config = SurfaceMaskConfig(
        x_size_m=1.0e-3,
        y_size_m=1.0e-3,
        z_max_m=4.0e-4,
        min_z_m=3.0e-5,
        nx=11,
        ny=11,
        nz=7,
        grid_mode_xy="edge_aware",
        grid_mode_z="near_surface_clustered",
        max_active_pixels_for_direct_sum=5000,
        edge_refinement_points_per_edge=3,
        max_edge_grid_points=12,
        source_description="Example surface-electrode Poisson-kernel field",
    )
    mask = load_binary_electrode_mask(mask_path, threshold=0.5)
    labels, _number_of_components = detect_electrode_components(mask)
    potentials = assign_four_electrode_voltages(labels, mask, config)
    voltage_map = build_voltage_map_from_components(labels, potentials)
    field_grid = compute_surface_field_grid(voltage_map, config, mask=mask)
    save_field_grid_to_npz(field_grid, output_path)


def main():
    """Generate PNG masks and one small field-grid NPZ."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    four_mask_path = OUTPUT_DIR / "simple_four_electrode_mask.png"
    parabolic_mask_path = OUTPUT_DIR / "parabolic_electrode_mask.png"
    field_path = OUTPUT_DIR / "simple_four_electrode_field.npz"

    draw_simple_four_electrode_mask(four_mask_path)
    draw_parabolic_electrode_mask(parabolic_mask_path)
    generate_example_field(four_mask_path, field_path)

    print(f"Saved: {four_mask_path}")
    print(f"Saved: {parabolic_mask_path}")
    print(f"Saved: {field_path}")


if __name__ == "__main__":
    main()
