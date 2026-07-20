# Wolfram / External Solver Export Notes

This project can load structured `.npz` files containing either an electric
field grid or a potential grid.  The goal is to make data from Wolfram,
COMSOL, or another solver comparable with the built-in Python FEM prototype.

## Required Units

Use SI units everywhere:

- `x_grid`, `y_grid`, `z_grid`: meters
- `potential_grid`: volts
- `electric_field_grid`: V/m

If the external solver works in millimeters, micrometers, or normalized units,
convert before saving the `.npz` file.

## Field Grid Format

Use this format when the external solver exports the electric field directly:

```text
x_grid: shape (nx,)
y_grid: shape (ny,)
z_grid: shape (nz,)
electric_field_grid: shape (nx, ny, nz, 3)
potential_grid: optional, shape (nx, ny, nz)
source_description: optional string
```

The vector components must be ordered as:

```text
electric_field_grid[..., 0] = Ex
electric_field_grid[..., 1] = Ey
electric_field_grid[..., 2] = Ez
```

Make sure the sign convention is:

```text
E = -grad(phi)
```

Some tools may report `grad(phi)` or use different sign conventions in custom
expressions, so check this explicitly.

## Potential Grid Format

Use this format when the external solver exports only potential:

```text
x_grid: shape (nx,)
y_grid: shape (ny,)
z_grid: shape (nz,)
potential_grid: shape (nx, ny, nz)
source_description: optional string
```

When this file is loaded, Python computes:

```text
E = -grad(phi)
```

using `numpy.gradient`.

## Array Ordering

The expected array ordering is:

```text
potential_grid[ix, iy, iz]
electric_field_grid[ix, iy, iz, component]
```

where:

```text
x = x_grid[ix]
y = y_grid[iy]
z = z_grid[iz]
```

The coordinate arrays must be strictly increasing.

## Conceptual Wolfram Workflow

In Wolfram, the conceptual steps are:

1. Create numeric arrays `xGrid`, `yGrid`, `zGrid` in meters.
2. Evaluate the potential or electric-field components on every grid point.
3. Arrange the result in `(ix, iy, iz)` order.
4. Save the arrays in a format Python can convert to `.npz`.

One simple route is to export arrays from Wolfram as CSV/HDF5/JSON and then use
a small Python conversion script to create the final `.npz`.  The final Python
save call should look conceptually like:

```python
np.savez_compressed(
    "external_field_grid.npz",
    x_grid=x_grid,
    y_grid=y_grid,
    z_grid=z_grid,
    electric_field_grid=electric_field_grid,
    potential_grid=potential_grid,  # optional
    source_description="Wolfram export, SI units",
)
```

or, for potential only:

```python
np.savez_compressed(
    "external_potential_grid.npz",
    x_grid=x_grid,
    y_grid=y_grid,
    z_grid=z_grid,
    potential_grid=potential_grid,
    source_description="Wolfram potential export, SI units",
)
```

## Validation Workflow

Export the built-in Python FEM reference field:

```bash
python main.py --export-fem-field data/builtin_fem_field_grid.npz
```

Then compare against an external export on the same grid:

```bash
python compare_fields.py \
  --reference data/builtin_fem_field_grid.npz \
  --candidate data/external_field_grid.npz
```

The first comparison tool requires the coordinate grids to match exactly.  It
does not yet interpolate between different grids.
