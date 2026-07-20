"""Plotting helpers for the potential, electric field, and trajectory."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import DEFAULT_CONFIG, SimulationConfig
from fem_solver import FemResult
from particle_dynamics import ParticleResult


def _nearest_index(grid: np.ndarray, value: float) -> int:
    """Return the index of the grid point closest to ``value``."""

    return int(np.argmin(np.abs(grid - value)))


def plot_potential_slice(
    fem_result,
    z_value: float = 0.0,
    ax=None,
):
    """Plot a 2D x-y slice of the electric potential."""

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if getattr(fem_result, "potential_grid", None) is None:
        ax.text(
            0.5,
            0.5,
            "No potential grid available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return ax

    k = _nearest_index(fem_result.z_grid, z_value)
    potential_slice = fem_result.potential_grid[:, :, k].T

    image = ax.imshow(
        potential_slice,
        origin="lower",
        extent=[
            fem_result.x_grid[0],
            fem_result.x_grid[-1],
            fem_result.y_grid[0],
            fem_result.y_grid[-1],
        ],
        aspect="equal",
        cmap="coolwarm",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Base potential slice at z = {fem_result.z_grid[k]:.2e} m")
    plt.colorbar(image, ax=ax, label="phi_base [V for 1 V drive]")
    return ax


def plot_electric_field_slice(
    fem_result,
    z_value: float = 0.0,
    stride: int = 2,
    ax=None,
):
    """Plot electric-field arrows on a 2D x-y slice."""

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    k = _nearest_index(fem_result.z_grid, z_value)
    x = fem_result.x_grid[::stride]
    y = fem_result.y_grid[::stride]
    xx, yy = np.meshgrid(x, y, indexing="ij")

    field = fem_result.electric_field_grid[::stride, ::stride, k, :]
    ex = field[:, :, 0]
    ey = field[:, :, 1]
    speed = np.sqrt(ex**2 + ey**2).T

    image = ax.imshow(
        speed,
        origin="lower",
        extent=[
            fem_result.x_grid[0],
            fem_result.x_grid[-1],
            fem_result.y_grid[0],
            fem_result.y_grid[-1],
        ],
        aspect="equal",
        cmap="viridis",
    )
    ax.quiver(xx, yy, ex, ey, color="white", pivot="mid", scale=None)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Base electric field slice at z = {fem_result.z_grid[k]:.2e} m")
    plt.colorbar(image, ax=ax, label="|E_base,xy| [V/m per V]")
    return ax


def plot_trajectory_3d(particle_result: ParticleResult, ax=None):
    """Plot the 3D particle trajectory."""

    if ax is None:
        figure = plt.figure(figsize=(6, 5))
        ax = figure.add_subplot(111, projection="3d")

    r = particle_result.positions
    ax.plot(r[:, 0], r[:, 1], r[:, 2], lw=1.5)
    ax.scatter(r[0, 0], r[0, 1], r[0, 2], color="green", label="start")
    ax.scatter(r[-1, 0], r[-1, 1], r[-1, 2], color="red", label="end")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Particle trajectory")
    ax.legend()
    return ax


def plot_coordinates_vs_time(particle_result: ParticleResult, ax=None):
    """Plot x(t), y(t), and z(t)."""

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    t = particle_result.t
    r = particle_result.positions
    ax.plot(t, r[:, 0], label="x")
    ax.plot(t, r[:, 1], label="y")
    ax.plot(t, r[:, 2], label="z")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("coordinate [m]")
    ax.set_title("Coordinates vs time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_speed_vs_time(particle_result: ParticleResult, ax=None):
    """Plot particle speed as a function of time."""

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.plot(particle_result.t, particle_result.speed)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [m/s]")
    ax.set_title("Speed vs time")
    ax.grid(True, alpha=0.3)
    return ax


def plot_radius_vs_time(particle_result: ParticleResult, ax=None):
    """Plot radial distance r(t) = sqrt(x^2 + y^2 + z^2)."""

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    radius = np.linalg.norm(particle_result.positions, axis=1)
    ax.plot(particle_result.t, radius)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("r [m]")
    ax.set_title("Radial distance vs time")
    ax.grid(True, alpha=0.3)
    return ax


def plot_all(
    fem_result,
    particle_result: ParticleResult,
    config: SimulationConfig = DEFAULT_CONFIG,
    show: bool = True,
    output_dir: str | None = None,
):
    """Create all standard figures and optionally save them as PNG files."""

    figures = []

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    plot_potential_slice(fem_result, ax=ax1)
    figures.append(("potential_slice.png", fig1))

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    plot_electric_field_slice(fem_result, ax=ax2)
    figures.append(("electric_field_slice.png", fig2))

    fig3 = plt.figure(figsize=(6, 5))
    ax3 = fig3.add_subplot(111, projection="3d")
    plot_trajectory_3d(particle_result, ax=ax3)
    figures.append(("particle_trajectory_3d.png", fig3))

    fig4, ax4 = plt.subplots(figsize=(7, 4))
    plot_coordinates_vs_time(particle_result, ax=ax4)
    figures.append(("coordinates_vs_time.png", fig4))

    fig5, ax5 = plt.subplots(figsize=(7, 4))
    plot_speed_vs_time(particle_result, ax=ax5)
    figures.append(("speed_vs_time.png", fig5))

    if output_dir is not None:
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for filename, figure in figures:
            figure.savefig(save_dir / filename, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    return figures
