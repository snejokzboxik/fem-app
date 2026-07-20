"""Editable parameters for the charged-particle trap prototype.

The values below are placeholders for learning and code testing.  They are not
real laboratory parameters yet.  When real trap dimensions, voltages, particle
properties, and gas damping are known, start by editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import pi


@dataclass(frozen=True)
class GeometryConfig:
    """Geometry settings for the current simple trap model."""

    # TODO: replace with real lab dimensions.
    # The domain is a rectangular box centered at the origin.
    domain_size: tuple[float, float, float] = (1.0e-3, 1.0e-3, 1.0e-3)

    # Fraction of each side length used by the simple electrode-like patches.
    # Four side patches imitate a quadrupole-like boundary condition:
    # x-faces are +V0, y-faces are -V0, and the rest of the outer box is 0 V.
    electrode_patch_fraction: float = 0.65

    # Future geometry loaders can switch on this value.
    geometry_type: str = "box_quadrupole_patch"


@dataclass(frozen=True)
class MeshConfig:
    """Mesh settings for the FEM solve."""

    # Number of cells along x, y, z for the tensor-product tetrahedral mesh.
    # Increase these values for smoother fields; keep them small while learning.
    mesh_cells: tuple[int, int, int] = (10, 10, 10)


@dataclass(frozen=True)
class ParticleConfig:
    """Particle material, charge, damping, and initial state."""

    # TODO: replace with real particle mass, charge, and damping.
    particle_mass: float | None = 1.0e-18  # kg
    particle_charge: float = 1.0e-16  # C

    # Optional material parameters.  If particle_mass is None and both values
    # below are provided, resolved_mass() computes a spherical-particle mass.
    particle_radius: float | None = None  # m
    particle_density: float | None = None  # kg/m^3

    damping_coefficient: float = 1.0e-15  # kg/s

    # Initial state: position in meters, velocity in meters/second.
    initial_position: tuple[float, float, float] = (5.0e-5, 0.0, 0.0)
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def sphere_mass(self) -> float:
        """Compute mass = density * 4/3 * pi * radius^3 for a sphere."""

        if self.particle_radius is None or self.particle_density is None:
            raise ValueError(
                "particle_radius and particle_density are required to compute "
                "a spherical particle mass."
            )
        return self.particle_density * (4.0 / 3.0) * pi * self.particle_radius**3

    def resolved_mass(self) -> float:
        """Return explicit mass, or compute it from radius and density."""

        if self.particle_mass is not None:
            return self.particle_mass
        return self.sphere_mass()


@dataclass(frozen=True)
class VoltageConfig:
    """Static and RF voltage settings."""

    # TODO: replace with real electrode voltages.
    # Used only when use_time_dependent_voltage is False.  The FEM solver
    # computes a normalized base field for 1 V and the dynamics code multiplies
    # that base field by this static amplitude.
    voltage_amplitude: float = 20.0  # volts

    # These RF fields are disabled in the first prototype.  They are included
    # so the dynamics code already has a natural place for V(t).
    use_time_dependent_voltage: bool = False
    dc_voltage: float = 0.0
    rf_voltage: float = 20.0
    rf_angular_frequency: float = 2.0 * pi * 1.0e3  # rad/s


@dataclass(frozen=True)
class SolverConfig:
    """Time integration and sweep-analysis settings."""

    # TODO: choose times and tolerances appropriate for the real dynamics.
    simulation_time: tuple[float, float] = (0.0, 2.0e-3)
    time_step: float = 2.0e-6
    ode_rtol: float = 1.0e-7
    ode_atol: float = 1.0e-10
    max_time_step: float = 2.0e-6

    # TODO: choose this from the real trap size and the region considered
    # physically useful.  A particle can survive inside the box while still
    # exploring a large radius, so "confined" is stricter than "survived".
    confinement_radius_threshold: float = 2.0e-4  # m


@dataclass(frozen=True)
class OutputConfig:
    """Output location settings."""

    output_dir: str = "results"


@dataclass(frozen=True)
class SimulationConfig:
    """Compatibility wrapper around smaller logical config sections.

    Existing code can still read flat fields such as ``config.domain_size`` or
    ``config.particle_mass``.  New code can use the clearer nested structure:
    ``config.geometry.domain_size``, ``config.particle.particle_charge``, etc.
    """

    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    particle: ParticleConfig = field(default_factory=ParticleConfig)
    voltage: VoltageConfig = field(default_factory=VoltageConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # --- Backward-compatible flat read-only properties ---------------------
    @property
    def domain_size(self) -> tuple[float, float, float]:
        return self.geometry.domain_size

    @property
    def electrode_patch_fraction(self) -> float:
        return self.geometry.electrode_patch_fraction

    @property
    def geometry_type(self) -> str:
        return self.geometry.geometry_type

    @property
    def mesh_cells(self) -> tuple[int, int, int]:
        return self.mesh.mesh_cells

    @property
    def voltage_amplitude(self) -> float:
        return self.voltage.voltage_amplitude

    @property
    def use_time_dependent_voltage(self) -> bool:
        return self.voltage.use_time_dependent_voltage

    @property
    def dc_voltage(self) -> float:
        return self.voltage.dc_voltage

    @property
    def rf_voltage(self) -> float:
        return self.voltage.rf_voltage

    @property
    def rf_angular_frequency(self) -> float:
        return self.voltage.rf_angular_frequency

    @property
    def particle_mass(self) -> float:
        return self.particle.resolved_mass()

    @property
    def particle_charge(self) -> float:
        return self.particle.particle_charge

    @property
    def particle_radius(self) -> float | None:
        return self.particle.particle_radius

    @property
    def particle_density(self) -> float | None:
        return self.particle.particle_density

    @property
    def damping_coefficient(self) -> float:
        return self.particle.damping_coefficient

    @property
    def initial_position(self) -> tuple[float, float, float]:
        return self.particle.initial_position

    @property
    def initial_velocity(self) -> tuple[float, float, float]:
        return self.particle.initial_velocity

    @property
    def simulation_time(self) -> tuple[float, float]:
        return self.solver.simulation_time

    @property
    def time_step(self) -> float:
        return self.solver.time_step

    @property
    def ode_rtol(self) -> float:
        return self.solver.ode_rtol

    @property
    def ode_atol(self) -> float:
        return self.solver.ode_atol

    @property
    def max_time_step(self) -> float:
        return self.solver.max_time_step

    @property
    def confinement_radius_threshold(self) -> float:
        return self.solver.confinement_radius_threshold

    @property
    def output_dir(self) -> str:
        return self.output.output_dir


_FLAT_FIELD_TO_SECTION = {
    "domain_size": ("geometry", "domain_size"),
    "electrode_patch_fraction": ("geometry", "electrode_patch_fraction"),
    "geometry_type": ("geometry", "geometry_type"),
    "mesh_cells": ("mesh", "mesh_cells"),
    "particle_mass": ("particle", "particle_mass"),
    "particle_charge": ("particle", "particle_charge"),
    "particle_radius": ("particle", "particle_radius"),
    "particle_density": ("particle", "particle_density"),
    "damping_coefficient": ("particle", "damping_coefficient"),
    "initial_position": ("particle", "initial_position"),
    "initial_velocity": ("particle", "initial_velocity"),
    "voltage_amplitude": ("voltage", "voltage_amplitude"),
    "use_time_dependent_voltage": ("voltage", "use_time_dependent_voltage"),
    "dc_voltage": ("voltage", "dc_voltage"),
    "rf_voltage": ("voltage", "rf_voltage"),
    "rf_angular_frequency": ("voltage", "rf_angular_frequency"),
    "simulation_time": ("solver", "simulation_time"),
    "time_step": ("solver", "time_step"),
    "ode_rtol": ("solver", "ode_rtol"),
    "ode_atol": ("solver", "ode_atol"),
    "max_time_step": ("solver", "max_time_step"),
    "confinement_radius_threshold": ("solver", "confinement_radius_threshold"),
    "output_dir": ("output", "output_dir"),
}


def with_config_overrides(config: SimulationConfig, **overrides) -> SimulationConfig:
    """Return a new config with flat or nested fields replaced.

    This keeps scripts readable during the transition to nested configs:

        config = with_config_overrides(DEFAULT_CONFIG, mesh_cells=(5, 5, 5))

    Direct nested replacement is also supported:

        config = with_config_overrides(DEFAULT_CONFIG, particle=ParticleConfig(...))
    """

    sections = {
        "geometry": config.geometry,
        "mesh": config.mesh,
        "particle": config.particle,
        "voltage": config.voltage,
        "solver": config.solver,
        "output": config.output,
    }

    for key, value in overrides.items():
        if key in sections:
            sections[key] = value
            continue

        if key not in _FLAT_FIELD_TO_SECTION:
            raise TypeError(f"Unknown SimulationConfig field: {key}")

        section_name, field_name = _FLAT_FIELD_TO_SECTION[key]
        sections[section_name] = replace(
            sections[section_name],
            **{field_name: value},
        )

    return replace(config, **sections)


DEFAULT_CONFIG = SimulationConfig()
