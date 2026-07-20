# Charged Particle Trap Prototype

This is an educational Python prototype for simulating one charged micro- or
nano-particle in a simple 3D quadrupole-like electrostatic trap.

The project is intentionally small and readable.  It is not a final lab model.
The real electrode geometry, voltages, particle parameters, and damping model
should be added later.

Important limitation: this first version does not use cup-trap geometry.  It
uses a simple box with four electrode-like boundary patches to imitate a
quadrupole-like potential.

## Physical Model

The electrostatic potential `phi` is found from Laplace's equation:

```text
div(grad(phi)) = 0
```

The electric field is then computed from the potential:

```text
E = -grad(phi)
```

The particle state is

```text
[x, y, z, vx, vy, vz]
```

and the equation of motion is

```text
dr/dt = v
dv/dt = (q/m) E(r, t) - (gamma/m) v
```

The FEM solver computes a normalized static base field for electrode patches at
`+1 V` and `-1 V`.  The dynamics code then scales this field in time:

```text
E(t, r) = voltage_scale(t) * E_base(r)
```

If `use_time_dependent_voltage = False`, the scale factor is the static
`voltage_amplitude`, preserving the original static behavior.  If
`use_time_dependent_voltage = True`, the scale factor is

```text
voltage_scale(t) = dc_voltage + rf_voltage*cos(rf_angular_frequency*t)
```

This is a first simplified RF model based on the linearity of Laplace's
equation.

## Project Structure

```text
charged_particle_trap/
  README.md
  requirements.txt
  main.py
  compare_fields.py
  surface_mask_demo.py
  sweep.py
  case_study.py
  metrics.py
  voltage_protocols.py
  drag_models.py
  experiment_config.py
  experiment_presets.py
  report_export.py
  field_data.py
  field_validation.py
  surface_superposition.py
  surface_geometry_sources.py
  mathieu_analysis.py
  config.py
  fem_solver.py
  field_interpolation.py
  particle_dynamics.py
  visualization.py
  examples/
    generate_example_field_npz.py
    generate_surface_mask_example.py
  docs/
    wolfram_export_notes.md
  tests/
    test_basic.py
```

File roles:

- `config.py` stores editable parameters in small sections:
  geometry, mesh, particle, voltage, solver, and output.  Flat access such as
  `config.particle_mass` is kept for backward compatibility.
- `fem_solver.py` builds the 3D mesh, applies simple boundary conditions, solves
  the Laplace equation, and computes `E = -grad(phi)`.
- `field_interpolation.py` turns the grid electric field into a function
  `E_at_position(position)`.
- `particle_dynamics.py` integrates the 6D particle equation using
  `scipy.integrate.solve_ivp`.
- `sweep.py` runs RF voltage/frequency sweeps, saves CSV metrics, and produces
  survived/escaped and confined/not-confined maps.
- `case_study.py` inspects selected RF sweep points with longer trajectories
  and saves detailed trajectory/time plots.
- `metrics.py` computes trajectory metrics and classifies escaped, survived,
  and confined trajectories.
- `voltage_protocols.py` contains reusable RF timing and RF-case configuration
  helpers.
- `drag_models.py` contains approximate vacuum, Stokes, Cunningham-slip, and
  Epstein-like gas damping helpers used by the Streamlit UI.
- `experiment_config.py` captures a full Streamlit experiment as a
  JSON-serializable snapshot: workflow, geometry source, electrode roles,
  particle settings, RF/DC settings, dynamics settings, and scan settings.
- `experiment_presets.py` contains built-in demo experiments for quick
  supervisor demonstrations and reproducible UI starting points.
- `report_export.py` builds Markdown/HTML reports and optional ZIP experiment
  packages from the current UI state and latest results.
- `field_data.py` loads and saves structured electric-field or potential grids
  in a simple `.npz` format for external solvers.
- `field_validation.py` validates imported grids, compares field/potential
  grids, and estimates simple local symmetry checks.
- `compare_fields.py` compares two `.npz` field exports, for example built-in
  Python FEM versus Wolfram or COMSOL data.
- `surface_superposition.py` computes planar surface-electrode fields from
  black-white mask images using the Dirichlet half-space Poisson kernel.
- `surface_geometry_sources.py` defines manual electrode roles, safe
  function-defined electrode regions, RF/DC voltage maps, and the two-channel
  surface-field helper.
- `surface_mask_demo.py` turns a surface-electrode mask image into a `.npz`
  `FieldGrid` file from the command line.
- `mathieu_analysis.py` computes ideal and effective Mathieu parameters and
  plots the dimensionless Mathieu stability diagram.
- `visualization.py` plots potential slices, electric-field slices, the 3D
  trajectory, coordinate-time graphs, and speed versus time.
- `main.py` runs the full pipeline.
- `tests/test_basic.py` contains basic smoke tests.

## Installation

From inside this folder, create and activate a virtual environment if desired,
then install the dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- `numpy`
- `scipy`
- `matplotlib`
- `scikit-fem`
- `pytest`
- `streamlit`
- `streamlit-drawable-canvas`
- `Pillow`

## Running the Simulation

From inside `charged_particle_trap/`, run:

```bash
python main.py
```

The script will:

1. create a simple 3D box mesh,
2. solve the electrostatic Laplace equation,
3. compute the electric field,
4. interpolate the field at the particle position,
5. integrate the particle trajectory,
6. show plots and save PNG images in `results/`.

You can also use an imported external field or potential:

```bash
python main.py --field-grid examples/example_field_grid.npz
python main.py --potential-grid data/my_potential_grid.npz
```

To export the current built-in FEM field as a reference `.npz` file:

```bash
python main.py --export-fem-field data/builtin_fem_field_grid.npz
```

## Быстрый запуск на Windows

Для тестировщика или пользователя на Windows добавлены лёгкие launcher-ы:

- `run_app_windows.bat` - самый простой вариант, можно запускать двойным кликом;
- `run_app_windows.ps1` - то же самое через PowerShell;
- `scripts/build_windows_launcher.ps1` - необязательная сборка `dist/ChargedTrapLauncher.exe`.

Это запускатели, а не полноценный автономный exe со всем научным приложением внутри. Они запускают локальный проект через `.venv`, устанавливают зависимости из `requirements.txt` и открывают Streamlit в браузере.

Подробная инструкция: [docs/LAUNCH_WINDOWS.md](docs/LAUNCH_WINDOWS.md).

## Headless regression scenarios

Для быстрой регрессионной проверки без запуска Streamlit и без браузера можно
использовать лёгкий сценарный runner:

```bash
python scripts/run_regression_scenarios.py
```

Он запускает несколько коротких deterministic-сценариев для surface-геометрий,
импортированного `.npz` поля, RF/DC-разделения и quick-start preset. Результаты
сохраняются в:

```text
results/regression/regression_summary.json
results/regression/regression_summary.csv
```

Чтобы только посмотреть список сценариев:

```bash
python scripts/run_regression_scenarios.py --list
```

## Running the Streamlit UI

The Streamlit UI is an optional local prototype interface.  It is now organized
as a Russian-language research workflow for surface-electrode trap experiments.
It lets you change particle, gas/friction, voltage, field-source, simulation,
diagnostic, sweep, and export settings without editing Python files.

Install dependencies, then run:

```bash
streamlit run app.py
```

The UI opens as a compact light engineering dashboard.  The sidebar is now
minimal: it only selects the workflow, shows project status, and enables expert
settings.  The actual inputs live in main-page cards.  The large landing
scenario cards contain real buttons and stay synchronized with the sidebar
workflow selector.

Main workflow modes:

- `Проверить локализацию`: build/load the field, compute
  gas damping, run particle dynamics, then show localization-like / escaped /
  unclear diagnostics.
- `Подобрать RF-параметры`: build/load the field once, then scan RF
  voltage and RF frequency values without recomputing the field for every point.
- `Посмотреть поле / экспорт`: build/load a field, inspect slices and diagnostics,
  and export the current `FieldGrid` without running particle dynamics.
- `Экспертный режим`: expose additional numerical/debug options and the Mathieu
  analysis panel.

Field-source options:

- `Рисунок электродов`: upload a PNG/JPG surface-electrode mask and compute the
  field with the Poisson-kernel superposition model.
- `Электроды функциями`: define planar electrode regions with mathematical
  inequalities in `x` and `y`, similar in spirit to a Desmos sketch, then
  assign each region a manual electrical role.
- `Нарисовать электроды`: draw planar electrode regions directly in Streamlit
  with an interactive canvas or PNG fallback, then assign manual electrical
  roles and reuse the same Poisson-kernel surface solver.
- `Загрузить поле .npz`: import a structured electric-field grid.
- `Загрузить потенциал .npz`: import potential and compute `E = -grad(phi)`.
- `Демо / legacy FEM`: keep the old placeholder box FEM for pipeline debugging.
  It is not the recommended mode for real surface-electrode geometry.

Main input cards:

- `1. Геометрия и поле`: source selector and source-specific controls.  For a
  surface mask this includes image upload, binary mask threshold, physical size,
  z range, grid size, computational-mask downsampling, manual electrode roles,
  and previews of mask/components/RF/DC maps/grid.
- `2. Питание ловушки`: DC scale, RF amplitude, RF frequency, static field
  scale, and the double-scaling warning.
- `3. Частица`: particle preset, charge, mass, radius, density, and computed
  mass from `m = 4/3*pi*r^3*rho` when enabled.
- `4. Среда`: vacuum, air-like automatic damping, pressure-controlled gas, or
  custom `gamma`, with Knudsen number and damping time.
- `5. Начальные условия и время расчёта`: initial position/velocity, simulation
  time, step size, and localization threshold.  This replaces the old confusing
  `Параметры динамики` wording.
- `6. Запуск`: main run button and a compact workload summary.

Result tabs:

- `Обзор`: status badge, key metrics, min-|E| candidate, and damping summary.
- `Поле`: potential and electric-field slices plus field diagnostics.
- `Псевдопотенциал`: RF pseudopotential slice and warnings when RF is disabled.
- `Траектория`: 3D trajectory, x/y/z(t), radius, and speed plots.
- `Диагностика`: textual interpretation and physical warnings.
- `Эксперименты`: built-in presets, `experiment_config.json` save/load,
  Markdown/HTML report export, and optional ZIP packaging.
- `Экспорт`: FieldGrid, trajectory, metrics, and search-result export actions.

The gas-drag model is approximate.  Vacuum mode sets `gamma = 0`.  For gas
cases the UI estimates the mean free path, Knudsen number, drag regime, linear
damping `gamma`, and damping time `m/gamma`.  It chooses between Stokes,
Stokes-Cunningham, and an Epstein-like model based on Knudsen number.  This is
only a practical learning model for distinguishing vacuum, air, and particle
size effects, not a final gas-dynamics solver.

The UI saves outputs to timestamped folders such as
`results/ui_YYYYMMDD_HHMMSS/`.  Fixed-parameter runs export `field_grid.npz`,
`trajectory.csv`, `metrics.csv`, and PNG plots.  Parameter searches export
`parameter_search_results.csv` and heatmaps.

### Experiment presets, save/load, and reports

The UI has an `Эксперименты` tab for reproducible demos:

- built-in presets such as quick-start RF electrodes, RF + DC compensation,
  parabolic electrodes, a ring electrode, a canvas demo, and an imported-field
  demo;
- a one-click quick demo button on the landing dashboard;
- `experiment_config.json` download/upload for restoring the workflow mode,
  geometry source, function-defined electrode expressions, canvas design
  metadata when available, manual electrode roles, particle parameters, RF/DC
  parameters, environment settings, dynamics settings, and scan settings;
- `report.md` and `report.html` downloads with experiment settings, electrode
  roles, result metrics, field diagnostics, damping information, exported files,
  and physical warnings;
- optional `experiment_package.zip` export containing `experiment_config.json`,
  `report.md`, `report.html`, CSV/PNG outputs from the latest run, optional
  `field_grid.npz`, and `canvas_design.json` when the current geometry is a
  canvas design.

The presets are intentionally small and use placeholder scientific parameters.
They are meant for demonstrating the workflow and checking numerical behavior,
not for claiming validated trap stability.

Important physical warning: a static electrostatic potential alone does not
prove stable 3D trapping.  For RF traps, inspect the actual trajectory, RF-null
candidate, pseudopotential estimate, and parameter-search maps.  The geometry,
particle parameters, voltage settings, and gas damping remain placeholders until
they are replaced by real lab data.

## Using External Fields

The project can now use a structured electric-field or potential grid exported
from an external solver such as Wolfram, COMSOL, or another FEM tool.  This is
an optional path: if no file is supplied, the built-in placeholder FEM geometry
is still used.

All grid units must be SI:

- `x_grid`, `y_grid`, `z_grid`: meters,
- `electric_field_grid`: V/m,
- `potential_grid`: volts.

Expected electric-field `.npz` format:

```text
x_grid: 1D array with shape (nx,)
y_grid: 1D array with shape (ny,)
z_grid: 1D array with shape (nz,)
electric_field_grid: array with shape (nx, ny, nz, 3)
potential_grid: optional array with shape (nx, ny, nz)
source_description: optional short string
```

Expected potential `.npz` format:

```text
x_grid: 1D array with shape (nx,)
y_grid: 1D array with shape (ny,)
z_grid: 1D array with shape (nz,)
potential_grid: array with shape (nx, ny, nz)
source_description: optional short string
```

When a potential grid is loaded, the code computes:

```text
E = -grad(phi)
```

using `numpy.gradient`, the same transparent finite-difference approach used
by the current prototype.

To generate a small reference file:

```bash
python examples/generate_example_field_npz.py
```

This writes:

```text
examples/example_field_grid.npz
```

To run with an imported electric field:

```bash
python main.py --field-grid examples/example_field_grid.npz
```

To run with an imported potential:

```bash
python main.py --potential-grid data/my_potential_grid.npz
```

In Streamlit, use the sidebar **Field source** selector:

- `Built-in placeholder FEM`,
- `Uploaded electric field grid .npz`,
- `Uploaded potential grid .npz`,
- `Surface electrode mask, Poisson kernel`,
- `Function-defined surface electrodes`,
- `Canvas-drawn surface electrodes`.

The loaded field is treated as the base field used by the particle dynamics.
That base field is still multiplied by the selected static or RF voltage scale:

```text
E(t, r) = voltage_scale(t) * E_base(r)
```

So if your external solver exports a field already in physical V/m for the
actual voltage, set the UI/static voltage scale appropriately, commonly `1`.
If your external solver exports a field normalized to 1 V, keep using the
voltage scaling as before.

## Validating External Solver Exports

After exporting an external field from Wolfram, COMSOL, or another solver, use
the validation tools before trusting a particle trajectory.  A typical workflow
is:

```bash
python main.py --export-fem-field data/builtin_fem_field_grid.npz
python compare_fields.py \
  --reference data/builtin_fem_field_grid.npz \
  --candidate data/external_field_grid.npz
```

The comparison script:

1. loads both `.npz` files,
2. validates coordinate arrays, field shapes, optional potential shape, and
   finite values,
3. checks that the grids match exactly,
4. compares potentials if both files contain `potential_grid`,
5. compares electric fields component by component,
6. saves `comparison_report.json`.

For this first version, the coordinate grids must match exactly.  The script
does not yet interpolate one field onto another grid.

The report includes:

- `mean_abs_error`
- `max_abs_error`
- `relative_l2_error`
- component-wise `Ex`, `Ey`, `Ez` errors
- local curvature/symmetry checks such as `kx`, `ky`, `kz`, and `kx + ky`

The symmetry checks are only approximate diagnostics for the current
box-quadrupole-like placeholder geometry.  They are not a proof that an
external field is physically correct.

For Wolfram-specific export notes, see:

```text
docs/wolfram_export_notes.md
```

## Surface Electrode Masks Via Poisson Kernel

Planar surface-electrode traps are different from the placeholder 3D box FEM
geometry.  When all electrodes lie in the plane `z = 0`, the electrostatic
potential in the upper half-space can be computed from the analytic Dirichlet
half-space solution instead of building a 3D FEM mesh.

The model solves:

```text
grad^2 phi = 0,  z > 0
phi(x, y, 0) = V_surface(x, y)
```

with the Poisson kernel:

```text
K(dx, dy, z) = (1 / (2*pi)) * z / (dx^2 + dy^2 + z^2)^(3/2)
```

and:

```text
phi(x, y, z) = integral V_surface(x', y') K(x-x', y-y', z) dx' dy'
```

The implementation approximates this integral by pixel-center quadrature:

```text
phi(x, y, z) ~= sum_pixels V_pixel * K(x-x_pixel, y-y_pixel, z) * pixel_area
```

Then it computes:

```text
E = -grad(phi)
```

using `numpy.gradient(potential_grid, x_grid, y_grid, z_grid)`.  Passing the
actual coordinate arrays matters because the observation grid may be
non-uniform.

The image mask is the geometry input:

- white/light pixels are electrodes,
- black/dark pixels are background,
- connected white components are separate electrodes,
- connected components define geometry only,
- each component must be manually assigned one electrical role: `RF`, `DC`,
  `GND`, or `CUSTOM`,
- `RF` components contribute to a normalized RF base map,
- `DC` and `CUSTOM` components contribute to the static voltage map in volts,
- `GND` components are explicit 0 V electrodes,
- rectangles, circles, rings, triangles, parabolic shapes, curved electrodes,
  and hand-drawn regions are supported because the code integrates over pixels,
  not over shape-specific formulas.

To generate example masks:

```bash
python examples/generate_surface_mask_example.py
```

This creates:

```text
examples/surface_masks/simple_four_electrode_mask.png
examples/surface_masks/parabolic_electrode_mask.png
examples/surface_masks/simple_four_electrode_field.npz
```

To compute a surface-mask field from the command line:

```bash
python surface_mask_demo.py \
  --mask examples/surface_masks/simple_four_electrode_mask.png \
  --output results/surface_field.npz
```

Useful options include:

```text
--x-size
--y-size
--z-max
--min-z
--nx
--ny
--nz
--grid-mode-xy uniform|center_clustered_tanh|edge_aware
--grid-mode-z uniform|near_surface_clustered
--voltage-pattern four-electrode
--max-active-pixels
```

In Streamlit, choose **Field source**:

```text
Surface electrode mask, Poisson kernel
```

Then upload a PNG/JPG mask, set the threshold, assign manual RF/DC/GND/CUSTOM
roles, choose grid settings, and run the selected dashboard workflow.

High-resolution images are automatically downsampled for the Poisson-kernel
direct summation by default.  The original uploaded image is still used for
preview and connected-component detection.  The smaller computational mask is
used only for numerical quadrature, which keeps interactive runs practical for
images such as 1024 x 1024 masks.  The physical size does not change:
`x_size_m` and `y_size_m` stay fixed, while the pixel area in the integral is
recomputed from the computational mask resolution:

```text
dx_pixel = x_size_m / nx_computational_pixels
dy_pixel = y_size_m / ny_computational_pixels
pixel_area = dx_pixel * dy_pixel
```

In Streamlit, the surface-mask sidebar includes controls for enabling/disabling
this computational downsampling and setting the maximum computational mask
size.  The UI reports the original resolution, computational resolution, active
pixels before/after downsampling, and the active fraction used in the direct
sum.

The observation grid is still rectilinear and compatible with
`RegularGridInterpolator`, but it can be non-uniform:

- `uniform`: simple linear x/y sampling,
- `center_clustered_tanh`: more points near the trap center,
- `edge_aware`: combines a coarse base grid, center-clustered points, and extra
  points around electrode edge pixel coordinates.

Edge-aware sampling is useful because electric fields change rapidly near
boundaries between electrodes and background.  It is not a full adaptive mesh;
it only creates a non-uniform rectilinear grid that remains compatible with the
existing `FieldGrid` pipeline.

Performance safeguards:

- large mask images can be downsampled automatically for computation while
  preserving the same physical `x_size_m` and `y_size_m`,
- direct summation raises a readable error if too many electrode pixels are
  active,
- computation is chunked over observation points to avoid large temporary
  arrays,
- edge coordinates are deterministically subsampled when too many edge pixels
  are present,
- nearly duplicate grid points are filtered using `min_grid_spacing_m`.

Limitations:

- dielectric substrate effects are ignored,
- finite electrode thickness is ignored,
- black background is treated as 0 V,
- exact gap physics is simplified by the pixel mask,
- pixel resolution matters,
- observation grid resolution matters,
- edge-aware grids cannot recover geometry absent from the input image,
- the field is not evaluated at `z = 0`; use `min_z_m > 0`,
- direct summation can be slow for very large masks,
- units must be SI: meters, volts, and V/m.

## Drawing Surface Electrodes In Streamlit

The Streamlit source `Нарисовать электроды` adds an interactive surface
electrode editor.  It uses `streamlit-drawable-canvas` when the package is
installed.  If the component is unavailable, the same processing path can still
be used by uploading a drawn PNG/JPG mask.

Workflow:

1. Choose `Нарисовать электроды` as the field source.
2. Set the physical `x_size_m` and `y_size_m`.
3. Draw dark electrode regions on the light canvas with free draw, rectangle,
   circle, or polygon mode when supported by the canvas component.
4. Click `Подтвердить геометрию`, or upload a drawn PNG/JPG fallback.
5. Inspect connected components.
6. Assign each detected component one role: `RF`, `DC`, `GND`, or `CUSTOM`.
7. Inspect RF and DC/static maps.
8. Run the same field, trajectory, pseudopotential, and diagnostics workflows.

The canvas image is converted into a binary electrode mask by detecting visible
dark pixels.  Tiny noise components are removed with the `min area [px]`
setting.  Connected components become electrode regions; geometry is still
separate from electrical role assignment, so the app does not guess polarity.

The canvas mode feeds into the same surface pipeline as uploaded image masks and
function-defined electrodes:

```text
canvas image -> binary mask -> connected labels -> RF/DC maps
             -> Poisson-kernel FieldGrid -> particle dynamics/diagnostics
```

For RF/DC-separated surface fields, trajectory dynamics uses:

```text
E_total(t, r) = E_dc(r) + Vrf*cos(Omega*t + phase)*E_RF_base(r)
```

The pseudopotential uses only the RF field:

```text
Psi = Q^2 |E_RF|^2 / (4 m Omega^2)
```

DC electrodes affect the full trajectory as a static field, but are not placed
inside `|E_RF|` for the pseudopotential.

Canvas designs can be downloaded as JSON from the RF/DC maps tab and uploaded
again later.  The design JSON stores:

- geometry source type `canvas`,
- physical sizes,
- canvas resolution,
- encoded binary mask,
- manual electrode assignments,
- optional RF metadata and notes.

This is a raster geometry editor, not an exact CAD kernel.  Pixel resolution,
mask cleanup, and computational downsampling affect edge accuracy.  Dielectric
substrates, finite electrode thickness, exact CAD geometry, FFT acceleration,
and adaptive unstructured meshes are still not modeled.

## Function-Defined Surface Electrodes

The Streamlit field source `Электроды функциями` defines planar electrode
regions by mathematical inequalities over `x` and `y`.  The coordinate domain
is:

```text
x in [-x_size_m/2, x_size_m/2]
y in [-y_size_m/2, y_size_m/2]
```

Examples:

```text
abs(x) < 200e-6 and abs(y - 250e-6) < 80e-6
(x**2 + y**2) < (250e-6)**2
((x**2 + y**2) > (150e-6)**2) and ((x**2 + y**2) < (250e-6)**2)
y > 0.02*(x/1e-4)**2 + 50e-6 and y < 0.02*(x/1e-4)**2 + 250e-6
```

Supported syntax is intentionally small and safe:

- variables: `x`, `y`,
- arithmetic: `+`, `-`, `*`, `/`, `**`,
- comparisons: `<`, `<=`, `>`, `>=`, `==`, `!=`,
- boolean logic: `and`, `or`, `not`,
- functions: `abs`, `sqrt`, `sin`, `cos`,
- constants: `pi`, `e`.

The code uses a strict AST-based evaluator, not unrestricted `eval`.
Unsafe input such as `__import__("os")`, `open(...)`, `exec(...)`,
`eval(...)`, or attribute access like `x.__class__` is rejected.

If regions overlap, the deterministic rule is that later definitions override
earlier definitions.  The UI shows a warning when overlap is detected.

Manual electrical roles are shared by uploaded masks and function-defined
regions:

- `RF`: electrode belongs to the RF set,
- `DC`: electrode has a constant static voltage,
- `GND`: explicit 0 V electrode,
- `CUSTOM`: user-provided static voltage.

For surface geometries the app keeps RF and DC maps separate:

```text
E_total(t, r) = E_dc(r) + Vrf*cos(Omega*t + phase) * E_RF_base(r)
```

The pseudopotential estimate uses only the RF field:

```text
Psi = Q^2 |E_RF|^2 / (4 m Omega^2)
```

DC electrodes still affect the full trajectory through `E_dc(r)`, but they are
not included inside `|E_RF|` for the pseudopotential.

Mathieu diagnostics remain available in expert mode.  For arbitrary
surface-electrode geometry, effective Mathieu parameters are only local
curvature estimates near the chosen point; they are not exact global stability
parameters.

This phase does not implement browser drawing canvas, STL/Gmsh import,
dielectric substrate physics, exact closed-form electrode formulas, FFT
acceleration, or adaptive unstructured meshes.

## Running a Coarse RF Sweep

From inside `charged_particle_trap/`, run:

```bash
python sweep.py
```

The sweep script:

1. solves the normalized 1 V FEM base field once,
2. loops over a small grid of `rf_voltage` and `rf_angular_frequency`,
3. runs the particle simulation for each pair,
4. computes trajectory metrics such as `max_radius`, `final_radius`, final
   position, and final speed,
5. classifies the result as:
   - `escaped`: the particle left the computational domain,
   - `survived`: the particle stayed inside until final time but exceeded the
     confinement threshold,
   - `confined`: the particle stayed inside and `max_radius` stayed below the
     confinement threshold,
6. prints a table and a summary,
7. saves:
   - `results/rf_sweep_results.csv`,
   - `results/rf_stability_map.png`,
   - `results/rf_confinement_map.png`,
   - `results/rf_mathieu_stability_diagram.png`.

Edit the easy-to-read settings at the top of `sweep.py` to change the voltage
array, frequency array, sweep simulation time, and whether to run a finer sweep
near the coarse transition region.  The confinement threshold is configured in
`config.py` as `confinement_radius_threshold`.

The default sweep excludes `0 V` so the main map focuses on driven RF cases.
Add `0.0` manually to `RF_VOLTAGES` when you want a no-drive control case.
The current default sweep uses placeholder voltages `[2, 5, 10, 15, 20, 30,
40] V` and a mixed kHz range from `5 kHz` to `100 kHz`.

For each RF case, the ODE step limit is reduced to resolve the RF period:

```text
max_time_step = min(base_config.max_time_step, 1 / (50*rf_frequency_hz))
```

The CSV also reports `rf_period` and `simulated_rf_periods`.  This sweep is
still only for exploring numerical behavior with placeholder particle
parameters and placeholder electrode geometry.  It also includes optional
effective Mathieu columns:

- `mathieu_a_x`
- `mathieu_q_x`
- `mathieu_a_y`
- `mathieu_q_y`

These are estimated from the local FEM potential curvature near the trap center.
If the curvature estimate fails, the columns are left as `NaN` and the sweep
continues.

## Mathieu Stability Diagram

Classic quadrupole or Paul-trap theory often rewrites each transverse equation
of motion in the dimensionless Mathieu form:

```text
d2u/dtau2 + (a_mathieu - 2*q_mathieu*cos(2*tau))*u = 0
```

Here `a_mathieu` measures the DC focusing strength and `q_mathieu` measures the
RF focusing strength after scaling by particle mass, particle charge, trap size
or curvature, and RF angular frequency.  The name `q_mathieu` is deliberately
used in the code and documentation because it is not the same thing as
`particle_charge`, which is the real electric charge in coulombs.

The RF sweep maps and the Mathieu stability diagram answer different questions:

- RF sweep survival map: did the full numerical trajectory stay inside the
  computational box for each voltage-frequency pair?
- RF sweep confinement map: did the full numerical trajectory also stay below
  the chosen radius threshold?
- Mathieu stability diagram: where do the parameters lie in the ideal
  dimensionless a-q_mathieu stability plane?

The standard Mathieu plot is shown in the usual first-region window:

```text
0 <= q_mathieu <= 1.2
-0.5 <= a_mathieu <= 0.5
```

The plot does not auto-expand by default.  If the current point is outside this
window, the point is far from the usual dimensionless first-stability-region
range for the selected placeholder DC/RF/frequency values.  In Streamlit, the
checkbox **Auto zoom to current Mathieu point** can expand the axes for
inspection, but the fixed standard view is better for comparing against the
classic first region.

For an ideal quadrupole potential, `mathieu_analysis.py` provides
`compute_ideal_mathieu_parameters(...)`.  For the current FEM placeholder
geometry, the code first fits the normalized base potential near the center:

```text
phi_base ~= c0 + cx*x + cy*y + cz*z
            + 0.5*kx*x^2 + 0.5*ky*y^2 + 0.5*kz*z^2
```

Then it computes effective local values `mathieu_a_x`, `mathieu_q_x`,
`mathieu_a_y`, and `mathieu_q_y` from `kx` and `ky`.  These FEM-derived values
are useful learning diagnostics, but they are not exact global stability
parameters for the real 3D geometry.

In Streamlit, open the **Mathieu stability** tab to see:

- the fitted local FEM curvature near the center,
- the current effective Mathieu parameters from the sidebar settings,
- the Mathieu stability diagram with the current x/y points overlaid.

Remember: the voltage-frequency RF sweep maps are empirical trajectory
experiments in the current numerical model.  The Mathieu a-q diagram is a
dimensionless ideal/effective stability analysis.  They are related, but they
are not the same plot and should not be interpreted as interchangeable.

## Manual QA / тестирование

Для ручной проверки Streamlit-интерфейса добавлен QA-kit на русском языке:

- [README для тестировщика](docs/qa/README_FOR_TESTER.md)
- [Быстрый старт](docs/qa/QUICK_START_FOR_TESTER.md)
- [Полный тест-план](docs/qa/TEST_PLAN_FULL.md)
- [Шаблон баг-репорта](docs/qa/BUG_REPORT_TEMPLATE.md)
- [Шаблон итогового отчёта](docs/qa/TEST_REPORT_TEMPLATE.md)
- [Известные ограничения](docs/qa/KNOWN_LIMITATIONS.md)
- [QA-ассеты](examples/qa_assets/)

Готовые тестовые маски можно пересоздать командой:

```bash
python examples/qa_assets/generate_qa_assets.py
```

Лёгкая проверка QA-kit без запуска Streamlit:

```bash
python scripts/smoke_check.py
```

## Running RF Case Studies

From inside `charged_particle_trap/`, run:

```bash
python case_study.py
```

The case-study script solves the normalized 1 V FEM base field once, then runs
longer simulations for selected RF points such as:

- `10 V`, `5 kHz`
- `10 V`, `15 kHz`
- `10 V`, `30 kHz`
- `20 V`, `20 kHz`
- `20 V`, `50 kHz`
- `40 V`, `100 kHz`

For each case it saves:

- 3D trajectory,
- `x(t)`, `y(t)`, `z(t)`,
- `r(t) = sqrt(x^2 + y^2 + z^2)`,
- speed versus time.

Plots are saved in `results/case_studies/`.  The printed summary includes
status, final time, radius metrics, final speed, and exit time for escaped
cases.

## Running Tests

From inside `charged_particle_trap/`, run:

```bash
pytest
```

The tests check that:

- configuration values load,
- the FEM solver runs on a coarse mesh,
- the field interpolation returns a 3-component vector,
- the ODE solver returns trajectory arrays with the expected shapes.

## Where To Change Parameters

Start with `config.py`.

The most important placeholder values are:

- `domain_size`
- `mesh_cells`
- `voltage_amplitude`
- `particle_mass`
- `particle_charge`
- `damping_coefficient`
- `initial_position`
- `initial_velocity`
- `simulation_time`
- `time_step`
- ODE tolerances
- `confinement_radius_threshold`

All placeholder physical values are marked with `TODO` comments.

## What Is Placeholder Physics?

This prototype uses a simple rectangular computational domain.  The boundary
conditions imitate a quadrupole-like trap:

- two opposite x-wall patches are held at `+V0`,
- two opposite y-wall patches are held at `-V0`,
- the remaining outer boundary is grounded.

This is only a test model.  It is useful for learning the FEM and particle
dynamics pipeline, but it is not a real electrode geometry.

When real geometry is available, replace:

- the mesh generation in `fem_solver.py`,
- the `boundary_potential(...)` function,
- the placeholder values in `config.py`,
- possibly the electric-field calculation if the mesh becomes unstructured.

For a real RF trap model, also update `voltage_scale(...)` in
`particle_dynamics.py` and the voltage parameters in `config.py`.
