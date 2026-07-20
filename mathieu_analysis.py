"""Mathieu-parameter analysis for quadrupole-like RF trapping.

This module is intentionally separate from the particle ODE simulation.  The
RF sweeps in ``sweep.py`` are direct numerical experiments in voltage and
frequency.  The functions here convert the same physical inputs into the
dimensionless Mathieu parameters used for ideal Paul-trap theory.

Terminology note:
``particle_charge`` is the physical electric charge in coulombs.
``mathieu_q`` is the dimensionless Mathieu stability parameter.
They are never the same quantity.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


MATHIEU_PARAMETER_KEYS = [
    "mathieu_a_x",
    "mathieu_q_x",
    "mathieu_a_y",
    "mathieu_q_y",
]


def frequency_hz_to_omega(f_hz: float) -> float:
    """Convert frequency in Hz to angular frequency in rad/s."""

    return float(2.0 * np.pi * f_hz)


def omega_to_frequency_hz(omega: float) -> float:
    """Convert angular frequency in rad/s to frequency in Hz."""

    return float(omega / (2.0 * np.pi))


def compute_ideal_mathieu_parameters(
    particle_charge: float,
    particle_mass: float,
    dc_voltage: float,
    rf_voltage: float,
    r0: float,
    omega: float,
) -> dict:
    """Return ideal x/y Mathieu parameters for a simple quadrupole model.

    Convention used here
    --------------------
    The ideal normalized potential is

        phi_base = (x^2 - y^2) / (2*r0^2)

    and the physical voltage scale is

        V(t) = dc_voltage + rf_voltage*cos(omega*t).

    Near the center, the x equation becomes

        x_ddot + (particle_charge/particle_mass)
                 * (1/r0^2) * V(t) * x = 0.

    With tau = omega*t/2, this is written in the standard Mathieu form

        d2x/dtau2 + (mathieu_a_x - 2*mathieu_q_x*cos(2*tau))*x = 0.

    Therefore

        mathieu_a_x =  4*particle_charge*dc_voltage
                       / (particle_mass*r0^2*omega^2)
        mathieu_q_x = -2*particle_charge*rf_voltage
                       / (particle_mass*r0^2*omega^2)

    The y curvature has the opposite sign, so ``mathieu_a_y`` and
    ``mathieu_q_y`` are the negatives of the x values.  Other books may use a
    different RF phase or potential sign; the important thing is to keep the
    convention consistent with the equations above.
    """

    if particle_mass <= 0.0:
        raise ValueError("particle_mass must be positive.")
    if r0 <= 0.0:
        raise ValueError("r0 must be positive.")
    if omega == 0.0:
        raise ValueError("omega must be nonzero for Mathieu parameters.")

    denominator = particle_mass * r0**2 * omega**2
    mathieu_a_x = 4.0 * particle_charge * dc_voltage / denominator
    mathieu_q_x = -2.0 * particle_charge * rf_voltage / denominator

    return {
        "mathieu_a_x": float(mathieu_a_x),
        "mathieu_q_x": float(mathieu_q_x),
        "mathieu_a_y": float(-mathieu_a_x),
        "mathieu_q_y": float(-mathieu_q_x),
    }


def estimate_potential_curvature_near_center(fem_result, fit_radius: float | None = None) -> dict:
    """Fit a local quadratic model to the normalized FEM potential.

    The fitted model is

        phi_base ~= c0 + cx*x + cy*y + cz*z
                    + 0.5*kx*x^2 + 0.5*ky*y^2 + 0.5*kz*z^2

    The FEM solution is normalized to a 1 V drive, so the returned curvatures
    have units of 1/m^2 per volt of drive.  These values are local estimates
    near the center of the placeholder geometry, not exact global trap
    parameters.
    """

    x_mesh, y_mesh, z_mesh = np.meshgrid(
        fem_result.x_grid,
        fem_result.y_grid,
        fem_result.z_grid,
        indexing="ij",
    )
    coordinates = np.column_stack(
        (
            x_mesh.ravel(),
            y_mesh.ravel(),
            z_mesh.ravel(),
        )
    )
    potential = fem_result.potential_grid.ravel()
    radii = np.linalg.norm(coordinates, axis=1)

    if fit_radius is None:
        fit_radius = 0.30 * min(
            fem_result.x_grid[-1] - fem_result.x_grid[0],
            fem_result.y_grid[-1] - fem_result.y_grid[0],
            fem_result.z_grid[-1] - fem_result.z_grid[0],
        )

    design = None
    selected = None
    selected_potential = None
    rank = 0
    current_fit_radius = float(fit_radius)
    max_radius = float(np.max(radii))

    # Very coarse meshes can have a symmetric cluster of nearest points where
    # x^2, y^2, and z^2 are not independently identifiable.  Expand the local
    # fit region until the quadratic design matrix has full rank.
    for _attempt in range(8):
        mask = radii <= current_fit_radius
        if int(np.count_nonzero(mask)) < 10:
            closest = np.argsort(radii)[: min(10, len(radii))]
            mask = np.zeros_like(radii, dtype=bool)
            mask[closest] = True
            current_fit_radius = float(np.max(radii[closest]))

        selected = coordinates[mask]
        selected_potential = potential[mask]

        if len(selected) >= 7:
            x = selected[:, 0]
            y = selected[:, 1]
            z = selected[:, 2]
            design = np.column_stack(
                (
                    np.ones_like(x),
                    x,
                    y,
                    z,
                    0.5 * x**2,
                    0.5 * y**2,
                    0.5 * z**2,
                )
            )
            rank = int(np.linalg.matrix_rank(design))
            if rank == design.shape[1]:
                break

        if current_fit_radius >= max_radius:
            break
        current_fit_radius = min(max_radius, 1.35 * current_fit_radius)

    if selected is None or design is None or len(selected) < 7:
        raise ValueError("Not enough FEM grid points to fit a quadratic curvature.")

    coefficients, residuals, rank, _singular_values = np.linalg.lstsq(
        design,
        selected_potential,
        rcond=None,
    )
    if rank < design.shape[1]:
        raise ValueError("Quadratic curvature fit is rank deficient.")

    fitted = design @ coefficients
    rms_residual = float(np.sqrt(np.mean((fitted - selected_potential) ** 2)))

    return {
        "kx": float(coefficients[4]),
        "ky": float(coefficients[5]),
        "kz": float(coefficients[6]),
        "fit_radius": float(current_fit_radius),
        "number_of_points_used": int(len(selected)),
        "rms_residual": rms_residual,
    }


def compute_effective_mathieu_parameters_from_curvature(
    curvature: dict,
    particle_charge: float,
    particle_mass: float,
    dc_voltage: float,
    rf_voltage: float,
    omega: float,
) -> dict:
    """Estimate local FEM-based Mathieu parameters from fitted curvature.

    The normalized FEM potential is locally approximated as

        phi_base ~= 0.5*kx*x^2 + 0.5*ky*y^2 + ...

    With ``V(t) = dc_voltage + rf_voltage*cos(omega*t)``, the x equation is

        x_ddot + (particle_charge/particle_mass)*kx*V(t)*x = 0.

    Matching the same Mathieu convention used by
    ``compute_ideal_mathieu_parameters`` gives

        mathieu_a_axis =  4*particle_charge*k_axis*dc_voltage
                          / (particle_mass*omega^2)
        mathieu_q_axis = -2*particle_charge*k_axis*rf_voltage
                          / (particle_mass*omega^2)

    These are effective local parameters.  They are useful for comparing the
    placeholder FEM trap to ideal theory, but they are not exact global
    stability boundaries for the full 3D geometry.
    """

    if particle_mass <= 0.0:
        raise ValueError("particle_mass must be positive.")
    if omega == 0.0:
        raise ValueError("omega must be nonzero for Mathieu parameters.")

    denominator = particle_mass * omega**2

    def axis_parameters(axis_curvature: float) -> tuple[float, float]:
        mathieu_a_axis = (
            4.0 * particle_charge * axis_curvature * dc_voltage / denominator
        )
        mathieu_q_axis = (
            -2.0 * particle_charge * axis_curvature * rf_voltage / denominator
        )
        return float(mathieu_a_axis), float(mathieu_q_axis)

    mathieu_a_x, mathieu_q_x = axis_parameters(float(curvature["kx"]))
    mathieu_a_y, mathieu_q_y = axis_parameters(float(curvature["ky"]))

    return {
        "mathieu_a_x": mathieu_a_x,
        "mathieu_q_x": mathieu_q_x,
        "mathieu_a_y": mathieu_a_y,
        "mathieu_q_y": mathieu_q_y,
    }


def nan_mathieu_parameters() -> dict:
    """Return a placeholder dictionary for failed Mathieu calculations."""

    return {key: float("nan") for key in MATHIEU_PARAMETER_KEYS}


def plot_mathieu_stability_diagram(
    points=None,
    ax=None,
    mathieu_q_range: tuple[float, float] = (0.0, 1.2),
    mathieu_a_range: tuple[float, float] = (-0.5, 0.5),
    grid_size: tuple[int, int] = (121, 91),
    steps_per_period: int = 96,
    auto_zoom: bool = False,
):
    """Plot the first numerical Mathieu stability region.

    The diagram is computed for

        d2u/dtau2 + (a_mathieu - 2*q_mathieu*cos(2*tau))*u = 0.

    Stability is classified from the monodromy matrix over one coefficient
    period, tau in [0, pi].  The grid is deliberately modest so the plot remains
    useful in the Streamlit UI and tests.

    By default the axes stay fixed on the usual first-region view:
    ``0 <= mathieu_q <= 1.2`` and ``-0.5 <= mathieu_a <= 0.5``.  Large points
    are annotated as outside the visible view instead of forcing the first
    stability region to disappear.  Set ``auto_zoom=True`` when you explicitly
    want the axes to expand around the supplied points.

    ``points`` may be a single dict or a list of dicts.  A point can either use
    generic keys ``mathieu_a`` and ``mathieu_q`` or axis-specific keys
    ``mathieu_a_x``, ``mathieu_q_x``, ``mathieu_a_y``, and ``mathieu_q_y``.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    point_coordinates = _extract_mathieu_point_coordinates(points)
    visible_q_range = mathieu_q_range
    visible_a_range = mathieu_a_range
    if auto_zoom:
        visible_q_range, visible_a_range = _expanded_mathieu_ranges(
            mathieu_q_range,
            mathieu_a_range,
            point_coordinates,
        )

    mathieu_q_values = np.linspace(visible_q_range[0], visible_q_range[1], grid_size[0])
    mathieu_a_values = np.linspace(visible_a_range[0], visible_a_range[1], grid_size[1])
    stable = _compute_mathieu_stability_grid(
        mathieu_a_values,
        mathieu_q_values,
        steps_per_period=steps_per_period,
    )

    image = ax.contourf(
        mathieu_q_values,
        mathieu_a_values,
        stable.astype(float),
        levels=[-0.5, 0.5, 1.5],
        colors=["#f3d3d3", "#cfe8cf"],
        alpha=0.85,
    )
    ax.contour(
        mathieu_q_values,
        mathieu_a_values,
        stable.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=1.0,
    )
    colorbar = fig.colorbar(image, ax=ax, ticks=[0.0, 1.0])
    colorbar.ax.set_yticklabels(["unstable", "stable"])

    _overlay_mathieu_points(ax, point_coordinates)

    outside_count = _count_points_outside_ranges(
        point_coordinates,
        visible_q_range,
        visible_a_range,
    )
    if outside_count:
        ax.text(
            0.02,
            0.98,
            f"{outside_count} point(s) outside standard view",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.85,
            },
        )

    ax.set_xlabel("q_mathieu")
    ax.set_ylabel("a_mathieu")
    ax.set_title("Mathieu stability diagram, first region")
    ax.set_xlim(visible_q_range)
    ax.set_ylim(visible_a_range)
    ax.grid(True, alpha=0.25)
    return fig, ax


def _compute_mathieu_stability_grid(
    mathieu_a_values: np.ndarray,
    mathieu_q_values: np.ndarray,
    steps_per_period: int,
) -> np.ndarray:
    """Return a boolean stability grid using a vectorized RK4 monodromy solve."""

    mathieu_a_grid, mathieu_q_grid = np.meshgrid(
        mathieu_a_values,
        mathieu_q_values,
        indexing="ij",
    )

    tau_start = 0.0
    tau_end = np.pi
    step = (tau_end - tau_start) / steps_per_period

    # Fundamental solution 1 starts at u=1, u_tau=0.
    u1 = np.ones_like(mathieu_a_grid)
    v1 = np.zeros_like(mathieu_a_grid)

    # Fundamental solution 2 starts at u=0, u_tau=1.
    u2 = np.zeros_like(mathieu_a_grid)
    v2 = np.ones_like(mathieu_a_grid)

    tau = tau_start
    for _ in range(steps_per_period):
        u1, v1 = _mathieu_rk4_step(u1, v1, tau, step, mathieu_a_grid, mathieu_q_grid)
        u2, v2 = _mathieu_rk4_step(u2, v2, tau, step, mathieu_a_grid, mathieu_q_grid)
        tau += step

    monodromy_trace = u1 + v2
    return np.abs(monodromy_trace) <= 2.0


def _mathieu_rk4_step(
    u: np.ndarray,
    v: np.ndarray,
    tau: float,
    step: float,
    mathieu_a_grid: np.ndarray,
    mathieu_q_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Take one RK4 step for the Mathieu equation on a whole parameter grid."""

    def acceleration(local_tau, local_u):
        coefficient = mathieu_a_grid - 2.0 * mathieu_q_grid * np.cos(2.0 * local_tau)
        return -coefficient * local_u

    k1_u = v
    k1_v = acceleration(tau, u)

    k2_u = v + 0.5 * step * k1_v
    k2_v = acceleration(tau + 0.5 * step, u + 0.5 * step * k1_u)

    k3_u = v + 0.5 * step * k2_v
    k3_v = acceleration(tau + 0.5 * step, u + 0.5 * step * k2_u)

    k4_u = v + step * k3_v
    k4_v = acceleration(tau + step, u + step * k3_u)

    next_u = u + (step / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
    next_v = v + (step / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return next_u, next_v


def _extract_mathieu_point_coordinates(points) -> list[dict]:
    """Normalize generic or x/y point dictionaries into plottable points."""

    if points is None:
        return []

    if isinstance(points, dict):
        points = [points]

    coordinates = []
    for index, point in enumerate(points):
        status = point.get("status", "current")
        base_label = point.get("label", status if status else f"point {index + 1}")

        if "mathieu_a" in point and "mathieu_q" in point:
            coordinates.append(
                {
                    "mathieu_a": point["mathieu_a"],
                    "mathieu_q": point["mathieu_q"],
                    "status": status,
                    "label": base_label,
                    "marker": "o",
                }
            )
            continue

        if "mathieu_a_x" in point and "mathieu_q_x" in point:
            coordinates.append(
                {
                    "mathieu_a": point["mathieu_a_x"],
                    "mathieu_q": point["mathieu_q_x"],
                    "status": status,
                    "label": f"{base_label} x",
                    "marker": "o",
                }
            )

        if "mathieu_a_y" in point and "mathieu_q_y" in point:
            coordinates.append(
                {
                    "mathieu_a": point["mathieu_a_y"],
                    "mathieu_q": point["mathieu_q_y"],
                    "status": status,
                    "label": f"{base_label} y",
                    "marker": "s",
                }
            )

    return coordinates


def _expanded_mathieu_ranges(
    mathieu_q_range: tuple[float, float],
    mathieu_a_range: tuple[float, float],
    point_coordinates: list[dict],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Expand ranges to include finite points when auto_zoom is enabled."""

    finite_q = [
        float(point["mathieu_q"])
        for point in point_coordinates
        if np.isfinite(point["mathieu_q"])
    ]
    finite_a = [
        float(point["mathieu_a"])
        for point in point_coordinates
        if np.isfinite(point["mathieu_a"])
    ]

    q_min, q_max = mathieu_q_range
    a_min, a_max = mathieu_a_range
    if finite_q:
        q_min = min(q_min, min(finite_q))
        q_max = max(q_max, max(finite_q))
    if finite_a:
        a_min = min(a_min, min(finite_a))
        a_max = max(a_max, max(finite_a))

    q_margin = 0.05 * max(q_max - q_min, 1.0)
    a_margin = 0.05 * max(a_max - a_min, 1.0)
    return (q_min - q_margin, q_max + q_margin), (a_min - a_margin, a_max + a_margin)


def _count_points_outside_ranges(
    point_coordinates: list[dict],
    mathieu_q_range: tuple[float, float],
    mathieu_a_range: tuple[float, float],
) -> int:
    """Count finite points outside the plotted a-q window."""

    count = 0
    for point in point_coordinates:
        mathieu_q_value = point["mathieu_q"]
        mathieu_a_value = point["mathieu_a"]
        if not np.isfinite(mathieu_q_value) or not np.isfinite(mathieu_a_value):
            continue
        outside_q = not (mathieu_q_range[0] <= mathieu_q_value <= mathieu_q_range[1])
        outside_a = not (mathieu_a_range[0] <= mathieu_a_value <= mathieu_a_range[1])
        if outside_q or outside_a:
            count += 1
    return count


def _overlay_mathieu_points(ax, point_coordinates: list[dict]):
    """Overlay optional Mathieu points with simple status colors."""

    if not point_coordinates:
        return

    status_colors = {
        "escaped": "crimson",
        "survived": "darkorange",
        "confined": "seagreen",
        "current": "black",
    }
    used_labels = set()

    for point in point_coordinates:
        status = point.get("status", "current")
        color = status_colors.get(status, "black")
        _scatter_one_mathieu_point(
            ax,
            point["mathieu_q"],
            point["mathieu_a"],
            color,
            point.get("marker", "o"),
            point.get("label", status),
            used_labels,
        )

    if used_labels:
        ax.legend(loc="best", fontsize=8)


def _scatter_one_mathieu_point(
    ax,
    mathieu_q_value: float,
    mathieu_a_value: float,
    color: str,
    marker: str,
    label: str,
    used_labels: set,
):
    """Scatter one finite point and avoid repeated legend labels."""

    if not np.isfinite(mathieu_q_value) or not np.isfinite(mathieu_a_value):
        return

    legend_label = label if label not in used_labels else None
    ax.scatter(
        [mathieu_q_value],
        [mathieu_a_value],
        s=55,
        marker=marker,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        label=legend_label,
        zorder=5,
    )
    used_labels.add(label)
