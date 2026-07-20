"""Particle equation of motion in the interpolated electric field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from config import DEFAULT_CONFIG, SimulationConfig


@dataclass
class ParticleResult:
    """Trajectory data returned by the ODE solver."""

    t: np.ndarray
    states: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    speed: np.ndarray
    left_domain: bool
    exit_time: float | None


def voltage_scale(t: float, config: SimulationConfig) -> float:
    """Return the voltage multiplier for the normalized FEM base field.

    The FEM solve is performed once for a 1 V boundary condition.  Since
    Laplace's equation is linear, the physical field is approximated as

        E(t, r) = voltage_scale(t) * E_base(r)

    where ``E_base`` is the electric field produced by the normalized 1 V FEM
    solve.  This is a first simplified RF model, valid only while the electrode
    geometry is fixed and the electrostatic approximation is appropriate.
    """

    if not config.use_time_dependent_voltage:
        return config.voltage_amplitude

    return config.dc_voltage + config.rf_voltage * np.cos(
        config.rf_angular_frequency * t
    )


def particle_rhs(
    t: float,
    state: np.ndarray,
    E_function,
    config: SimulationConfig,
) -> np.ndarray:
    """Right-hand side of the 6D particle equation.

    The state vector is

        [x, y, z, vx, vy, vz]

    and the model is

        dr/dt = v
        dv/dt = (q/m) E(r, t) - (gamma/m) v
    """

    position = state[:3]
    velocity = state[3:]

    if getattr(E_function, "expects_time", False):
        electric_field = E_function(t, position)
    else:
        electric_field_base = E_function(position)

        # The interpolator returns NaN outside the domain.  The terminal event
        # below should stop integration at the boundary, but adaptive solvers
        # may briefly evaluate the RHS just outside; using zero field there
        # keeps the RHS finite.
        if not np.all(np.isfinite(electric_field_base)):
            electric_field_base = np.zeros(3)
        electric_field = voltage_scale(t, config) * electric_field_base

    if not np.all(np.isfinite(electric_field)):
        electric_field = np.zeros(3)

    acceleration = (
        (config.particle_charge / config.particle_mass) * electric_field
        - (config.damping_coefficient / config.particle_mass) * velocity
    )

    return np.concatenate((velocity, acceleration))


def make_domain_exit_event(config: SimulationConfig):
    """Create an event function that stops integration at the box boundary."""

    half_size = 0.5 * np.asarray(config.domain_size, dtype=float)

    def event(_t, state):
        distance_to_each_wall = half_size - np.abs(state[:3])
        return np.min(distance_to_each_wall)

    event.terminal = True
    event.direction = -1.0
    return event


def simulate_particle(E_function, config: SimulationConfig = DEFAULT_CONFIG) -> ParticleResult:
    """Integrate the charged-particle trajectory using ``solve_ivp``."""

    t0, t1 = config.simulation_time
    initial_state = np.array(
        (*config.initial_position, *config.initial_velocity),
        dtype=float,
    )

    # Include the final time when it falls exactly on the time grid.
    t_eval = np.arange(t0, t1 + 0.5 * config.time_step, config.time_step)

    solution = solve_ivp(
        fun=lambda t, y: particle_rhs(t, y, E_function, config),
        t_span=(t0, t1),
        y0=initial_state,
        t_eval=t_eval,
        events=make_domain_exit_event(config),
        rtol=config.ode_rtol,
        atol=config.ode_atol,
        max_step=config.max_time_step,
    )

    states = solution.y.T
    positions = states[:, :3]
    velocities = states[:, 3:]
    speed = np.linalg.norm(velocities, axis=1)

    left_domain = len(solution.t_events[0]) > 0
    exit_time = float(solution.t_events[0][0]) if left_domain else None

    return ParticleResult(
        t=solution.t,
        states=states,
        positions=positions,
        velocities=velocities,
        speed=speed,
        left_domain=left_domain,
        exit_time=exit_time,
    )
