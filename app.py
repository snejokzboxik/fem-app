"""Streamlit UI for the charged-particle trap prototype.

Run with:

    streamlit run app.py

The UI is an optional layer.  It calls the same FEM, interpolation, dynamics,
metrics, sweep, and plotting functions used by the scripts.
"""

from __future__ import annotations

import csv
from datetime import datetime
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None

from config import DEFAULT_CONFIG, with_config_overrides
from field_data import (
    load_field_grid_npz,
    load_potential_grid_npz,
    save_fem_field_to_npz,
    save_field_grid_to_npz,
)
from fem_solver import solve_laplace
from field_interpolation import make_E_at_position, make_E_at_position_from_field_grid
from field_validation import (
    estimate_symmetry_checks,
    validate_field_grid as validate_field_grid_report,
)
from drag_models import (
    EnvironmentConfig,
    compute_damping_gamma,
    particle_mass_from_radius_density,
)
from experiment_config import (
    apply_experiment_config_to_session_state,
    collect_current_experiment_config_from_ui,
    experiment_config_from_dict,
    validate_experiment_config,
)
from experiment_presets import built_in_experiment_presets, get_experiment_preset
from mathieu_analysis import (
    compute_effective_mathieu_parameters_from_curvature,
    estimate_potential_curvature_near_center,
    plot_mathieu_stability_diagram,
)
from metrics import (
    classify_localization_status,
    classify_particle_result,
    compute_trajectory_metrics,
    localization_status_label,
)
from particle_dynamics import simulate_particle
from report_export import (
    build_html_report,
    build_markdown_report,
    export_experiment_zip,
)
from surface_superposition import (
    SurfaceMaskConfig,
    assign_four_electrode_voltages,
    build_voltage_map_from_components,
    compute_surface_field_grid,
    detect_electrode_components,
    load_binary_electrode_mask,
    make_surface_observation_grid,
    prepare_surface_voltage_map_for_computation,
)
from surface_geometry_sources import (
    ROLE_CUSTOM,
    ROLE_DC,
    ROLE_GND,
    ROLE_RF,
    ElectrodeAssignment,
    ElectrodeRegionDefinition,
    TwoChannelElectricField,
    build_regions_from_canvas_mask,
    build_voltage_maps_from_assignments,
    canvas_design_from_dict,
    canvas_design_to_dict,
    canvas_design_to_voltage_maps,
    canvas_image_to_binary_mask,
    clean_binary_mask,
    combine_field_grids_for_preview,
    compute_pseudopotential_from_rf_field_grid,
    function_geometry_presets,
    label_canvas_electrodes,
    rasterize_function_regions,
)
from sweep import (
    plot_binary_map,
    run_parameter_sweep,
    save_results_csv,
)
from visualization import (
    plot_coordinates_vs_time,
    plot_electric_field_slice,
    plot_potential_slice,
    plot_radius_vs_time,
    plot_speed_vs_time,
    plot_trajectory_3d,
)
from voltage_protocols import make_rf_case_config


WARNING_TEXT = (
    "Это упрощённый исследовательский прототип с приближённой геометрией, "
    "параметрами частиц и газовым трением. Результаты показывают численное "
    "поведение модели, а не валидированную устойчивость реальной ловушки."
)


PHYSICS_WARNING_TEXT = (
    "Статический электростатический потенциал сам по себе не доказывает "
    "устойчивую 3D-ловушку. Для RF-ловушек смотрите динамику частицы, RF-null "
    "и псевдопотенциал."
)


DRAG_WARNING_TEXT = (
    "Модель трения приближённая: она нужна для различения вакуума, воздуха и "
    "размера частицы, а не как финальная газодинамическая модель."
)


WORKFLOW_CHECK = "Проверить локализацию"
WORKFLOW_SEARCH = "Подобрать RF-параметры"
WORKFLOW_FIELD_ONLY = "Посмотреть поле / экспорт"
WORKFLOW_EXPERT = "Экспертный режим"


DASHBOARD_WORKFLOW_MODES = [
    WORKFLOW_CHECK,
    WORKFLOW_SEARCH,
    WORKFLOW_FIELD_ONLY,
    WORKFLOW_EXPERT,
]

APP_WORKFLOW_MODE_KEY = "app_workflow_mode"
WORKFLOW_WIDGET_KEY = "workflow_mode_widget"
PENDING_WORKFLOW_MODE_KEY = "pending_workflow_mode"
LEGACY_WORKFLOW_WIDGET_KEY = "workflow_mode"
DEFAULT_SURFACE_MAX_ACTIVE_PIXELS = 10000
SURFACE_ACTIVE_PIXELS_SLOW_WARNING = 5000


WORKFLOW_DESCRIPTIONS = {
    WORKFLOW_CHECK: (
        "Задайте геометрию, частицу, среду и напряжения. Приложение рассчитает "
        "поле, траекторию и даст осторожную диагностику."
    ),
    WORKFLOW_SEARCH: (
        "Задайте диапазоны RF-напряжения и частоты. Приложение покажет, где "
        "частица не вылетает."
    ),
    WORKFLOW_FIELD_ONLY: (
        "Рассчитать или загрузить поле, посмотреть срезы и экспортировать "
        "FieldGrid."
    ),
    WORKFLOW_EXPERT: (
        "Расширенный режим с debug/legacy FEM, Mathieu-анализом и экспертными "
        "параметрами."
    ),
}


SOURCE_SURFACE = "Рисунок электродов"
SOURCE_FUNCTIONS = "Электроды функциями"
SOURCE_CANVAS = "Нарисовать электроды"
SOURCE_FIELD_NPZ = "Загрузить поле .npz"
SOURCE_POTENTIAL_NPZ = "Загрузить потенциал .npz"
SOURCE_LEGACY_FEM = "Демо / legacy FEM"


FIELD_SOURCE_TO_INTERNAL = {
    SOURCE_SURFACE: "Surface electrode mask, Poisson kernel",
    SOURCE_FUNCTIONS: "Function-defined surface electrodes",
    SOURCE_CANVAS: "Canvas-drawn surface electrodes",
    SOURCE_FIELD_NPZ: "Uploaded electric field grid .npz",
    SOURCE_POTENTIAL_NPZ: "Uploaded potential grid .npz",
    SOURCE_LEGACY_FEM: "Built-in placeholder FEM",
}


ROLE_LABEL_TO_CODE = {
    "RF": ROLE_RF,
    "DC": ROLE_DC,
    "GND / 0 V": ROLE_GND,
    "CUSTOM": ROLE_CUSTOM,
}

ROLE_CODE_TO_LABEL = {value: key for key, value in ROLE_LABEL_TO_CODE.items()}


DASHBOARD_MODE_TO_ACTION = {
    WORKFLOW_CHECK: "check",
    WORKFLOW_SEARCH: "search",
    WORKFLOW_FIELD_ONLY: "field_only",
    WORKFLOW_EXPERT: "expert",
}


PARTICLE_PRESETS = {
    "Микрочастица": {
        "particle_charge": 1.0e-16,
        "particle_radius": 1.0e-6,
        "particle_density": 2200.0,
        "initial_position": (5.0e-5, 0.0, 1.0e-4),
        "initial_velocity": (0.0, 0.0, 0.0),
        "derive_mass": True,
    },
    "Наночастица": {
        "particle_charge": 1.0e-18,
        "particle_radius": 50.0e-9,
        "particle_density": 2200.0,
        "initial_position": (1.0e-5, 0.0, 5.0e-5),
        "initial_velocity": (0.0, 0.0, 0.0),
        "derive_mass": True,
    },
    "Тестовая частица": {
        "particle_charge": 1.0e-16,
        "particle_radius": 1.0e-6,
        "particle_density": 2200.0,
        "particle_mass": 1.0e-18,
        "initial_position": (5.0e-5, 0.0, 1.0e-4),
        "initial_velocity": (0.0, 0.0, 0.0),
        "derive_mass": False,
    },
    "Пользовательские параметры": {
        "derive_mass": False,
    },
}


DEMO_PRESETS = {
    "Custom": {
        "description": "Manual controls using the current Python defaults.",
    },
    "Static unstable test": {
        "description": "Placeholder static-voltage case that tends to leave the box.",
        "mode": "static",
        "particle_mass": 1.0e-18,
        "particle_charge": 1.0e-16,
        "damping_coefficient": 1.0e-15,
        "initial_position": (5.0e-5, 0.0, 0.0),
        "initial_velocity": (0.0, 0.0, 0.0),
        "voltage_amplitude": 20.0,
        "rf_voltage": 20.0,
        "rf_frequency_hz": 1.0e3,
        "dc_voltage": 0.0,
        "simulation_duration": 2.0e-3,
        "time_step": 2.0e-6,
        "confinement_radius_threshold": 2.0e-4,
        "mesh_cells": (5, 5, 5),
        "sweep_voltages": "5,10,20",
        "sweep_frequencies": "5000,15000,30000",
    },
    "Low-frequency RF escape test": {
        "description": "Placeholder RF case chosen to demonstrate fast escape.",
        "mode": "RF",
        "particle_mass": 1.0e-18,
        "particle_charge": 1.0e-16,
        "damping_coefficient": 1.0e-15,
        "initial_position": (5.0e-5, 0.0, 0.0),
        "initial_velocity": (0.0, 0.0, 0.0),
        "voltage_amplitude": 20.0,
        "rf_voltage": 10.0,
        "rf_frequency_hz": 5.0e3,
        "dc_voltage": 0.0,
        "simulation_duration": 5.0e-3,
        "time_step": 1.0e-5,
        "confinement_radius_threshold": 2.0e-4,
        "mesh_cells": (5, 5, 5),
        "sweep_voltages": "5,10,20",
        "sweep_frequencies": "5000,10000,15000",
    },
    "High-frequency RF confined test": {
        "description": "Placeholder RF case that often stays near the center.",
        "mode": "RF",
        "particle_mass": 1.0e-18,
        "particle_charge": 1.0e-16,
        "damping_coefficient": 1.0e-15,
        "initial_position": (5.0e-5, 0.0, 0.0),
        "initial_velocity": (0.0, 0.0, 0.0),
        "voltage_amplitude": 20.0,
        "rf_voltage": 20.0,
        "rf_frequency_hz": 5.0e4,
        "dc_voltage": 0.0,
        "simulation_duration": 5.0e-3,
        "time_step": 1.0e-5,
        "confinement_radius_threshold": 2.0e-4,
        "mesh_cells": (5, 5, 5),
        "sweep_voltages": "10,20,40",
        "sweep_frequencies": "30000,50000,100000",
    },
    "Mathieu RF demo": {
        "description": "Placeholder RF-only case for viewing the Mathieu a-q point.",
        "mode": "RF",
        "particle_mass": 1.0e-18,
        "particle_charge": 1.0e-16,
        "damping_coefficient": 1.0e-15,
        "initial_position": (5.0e-5, 0.0, 0.0),
        "initial_velocity": (0.0, 0.0, 0.0),
        "voltage_amplitude": 20.0,
        "rf_voltage": 20.0,
        "rf_frequency_hz": 3.0e4,
        "dc_voltage": 0.0,
        "simulation_duration": 2.0e-3,
        "time_step": 1.0e-5,
        "confinement_radius_threshold": 2.0e-4,
        "mesh_cells": (5, 5, 5),
        "sweep_voltages": "10,20,30",
        "sweep_frequencies": "15000,30000,50000",
    },
    "Quick sweep demo": {
        "description": "Small placeholder grid for a quick survived/escaped map.",
        "mode": "RF",
        "particle_mass": 1.0e-18,
        "particle_charge": 1.0e-16,
        "damping_coefficient": 1.0e-15,
        "initial_position": (5.0e-5, 0.0, 0.0),
        "initial_velocity": (0.0, 0.0, 0.0),
        "voltage_amplitude": 20.0,
        "rf_voltage": 20.0,
        "rf_frequency_hz": 3.0e4,
        "dc_voltage": 0.0,
        "simulation_duration": 2.0e-3,
        "time_step": 1.0e-5,
        "confinement_radius_threshold": 2.0e-4,
        "mesh_cells": (4, 4, 4),
        "sweep_voltages": "5,10,20",
        "sweep_frequencies": "5000,30000,100000",
    },
}


def preset_value(preset: dict, key: str, default):
    """Return a preset value or the default when the preset is Custom."""

    return preset.get(key, default)


def preset_key(preset_name: str, field_name: str) -> str:
    """Create a Streamlit widget key that resets when the preset changes."""

    safe_name = preset_name.lower().replace(" ", "_").replace("-", "_")
    return f"{safe_name}_{field_name}"


def dashboard_mode_to_action(mode: str) -> str:
    """Map a Russian dashboard mode label to a compact action code."""

    return DASHBOARD_MODE_TO_ACTION.get(mode, "check")


def _normalize_workflow_mode(mode: str | None) -> str:
    """Return a valid dashboard workflow mode."""

    return mode if mode in DASHBOARD_WORKFLOW_MODES else WORKFLOW_CHECK


def initialize_workflow_state(st) -> str:
    """Initialize internal workflow state before creating the sidebar widget."""

    pending_mode = st.session_state.pop(PENDING_WORKFLOW_MODE_KEY, None)
    if pending_mode is not None:
        current_mode = pending_mode
    elif WORKFLOW_WIDGET_KEY in st.session_state:
        current_mode = st.session_state[WORKFLOW_WIDGET_KEY]
    else:
        current_mode = st.session_state.get(
            APP_WORKFLOW_MODE_KEY,
            st.session_state.get(LEGACY_WORKFLOW_WIDGET_KEY, WORKFLOW_CHECK),
        )
    current_mode = _normalize_workflow_mode(current_mode)
    st.session_state[APP_WORKFLOW_MODE_KEY] = current_mode
    st.session_state[WORKFLOW_WIDGET_KEY] = current_mode
    return current_mode


def set_pending_workflow_mode(st, mode: str) -> None:
    """Request a workflow change for the next Streamlit rerun."""

    st.session_state[PENDING_WORKFLOW_MODE_KEY] = _normalize_workflow_mode(mode)


def status_badge_html(status: str) -> str:
    """Return a small HTML status badge for overview cards."""

    labels = {
        "localized_like": "Похоже на локализацию",
        "escaped": "Частица вылетела",
        "unclear": "Неясно",
    }
    colors = {
        "localized_like": ("#dcfce7", "#166534", "#86efac"),
        "escaped": ("#fee2e2", "#991b1b", "#fecaca"),
        "unclear": ("#fef9c3", "#854d0e", "#fde68a"),
    }
    background, text, border = colors.get(status, ("#f1f5f9", "#334155", "#cbd5e1"))
    label = labels.get(status, status)
    return (
        f'<span class="status-badge" style="background:{background};'
        f'color:{text};border:1px solid {border};">{label}</span>'
    )


def render_dashboard_style(st):
    """Apply a compact light dashboard style."""

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1420px;
        }
        section[data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #e2e8f0;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .dashboard-kicker {
            color: #475569;
            font-size: 0.95rem;
            margin-top: -0.35rem;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.28rem 0.7rem;
            font-weight: 700;
            font-size: 0.88rem;
        }
        .landing-card {
            min-height: 126px;
            padding: 0.15rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(container, title: str, subtitle: str | None = None):
    """Render a compact dashboard section header."""

    container.markdown(f"### {title}")
    if subtitle:
        container.caption(subtitle)


def render_card(container, title: str, subtitle: str | None = None):
    """Create a bordered Streamlit card with a title."""

    card = container.container(border=True)
    render_section_header(card, title, subtitle)
    return card


def render_status_badge(st, status: str):
    """Display a colored localization status badge."""

    st.markdown(status_badge_html(status), unsafe_allow_html=True)


def parse_comma_separated_floats(text: str) -> np.ndarray:
    """Parse comma-separated numbers from a Streamlit text input."""

    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(float(chunk))
    if not values:
        raise ValueError("Enter at least one numeric value.")
    return np.asarray(values, dtype=float)


def create_ui_output_dir(base_output_dir: str = "results") -> Path:
    """Create a timestamped UI output folder."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / f"ui_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_config_from_sidebar(st):
    """Create a SimulationConfig and UI options from Russian sidebar controls."""

    defaults = DEFAULT_CONFIG
    sidebar = st.sidebar

    sidebar.header("Режим работы")
    workflow_mode = sidebar.radio(
        "Что сделать?",
        [WORKFLOW_CHECK, WORKFLOW_SEARCH, WORKFLOW_FIELD_ONLY],
    )

    sidebar.header("Источник поля / геометрия")
    source_label = sidebar.selectbox(
        "Источник поля",
        [
            SOURCE_SURFACE,
            SOURCE_FUNCTIONS,
            SOURCE_CANVAS,
            SOURCE_FIELD_NPZ,
            SOURCE_POTENTIAL_NPZ,
            SOURCE_LEGACY_FEM,
        ],
    )
    field_source = FIELD_SOURCE_TO_INTERNAL[source_label]
    uploaded_field_file = None
    surface_options = None

    if source_label == SOURCE_LEGACY_FEM:
        sidebar.warning(
            "Встроенная FEM-геометрия используется только для отладки пайплайна. "
            "Для реальных surface-конфигураций используйте рисунок электродов "
            "или импорт .npz."
        )

    if source_label in {SOURCE_FIELD_NPZ, SOURCE_POTENTIAL_NPZ}:
        uploaded_field_file = sidebar.file_uploader(
            "Загрузить файл .npz",
            type=["npz"],
            help="Координаты должны быть в метрах, поле в В/м, потенциал в В.",
        )

    if source_label == SOURCE_SURFACE:
        with sidebar.expander("Параметры расчёта поля", expanded=True):
            uploaded_surface_mask = sidebar.file_uploader(
                "Загрузить рисунок электродов PNG/JPG",
                type=["png", "jpg", "jpeg"],
            )
            sidebar.caption("Светлые связные области считаются электродами; фон = 0 В.")
            mask_threshold = sidebar.slider(
                "Порог бинаризации маски",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
            )
            x_size_m = sidebar.number_input(
                "Физический размер по x [м]",
                value=1.0e-3,
                format="%.6e",
            )
            y_size_m = sidebar.number_input(
                "Физический размер по y [м]",
                value=1.0e-3,
                format="%.6e",
            )
            z_max_m = sidebar.number_input("z max [м]", value=1.0e-3, format="%.6e")
            min_z_m = sidebar.number_input("min z [м]", value=2.0e-5, format="%.6e")
            surface_nx = sidebar.number_input("nx поля", value=25, min_value=3, step=1)
            surface_ny = sidebar.number_input("ny поля", value=25, min_value=3, step=1)
            surface_nz = sidebar.number_input("nz поля", value=15, min_value=3, step=1)
            grid_mode_xy = sidebar.selectbox(
                "Сетка x-y",
                ["uniform", "center_clustered_tanh", "edge_aware"],
                index=2,
            )
            grid_mode_z = sidebar.selectbox(
                "Сетка z",
                ["uniform", "near_surface_clustered"],
                index=1,
            )
            downsample_mask_for_computation = sidebar.checkbox(
                "Уменьшать маску для расчёта",
                value=True,
            )
            max_computational_mask_size = sidebar.slider(
                "Максимальный размер маски для расчёта",
                min_value=32,
                max_value=512,
                value=192,
                step=16,
            )

        with sidebar.expander("Экспертные настройки surface-сетки"):
            cluster_strength_xy = sidebar.number_input(
                "Сила сгущения x-y",
                value=2.5,
                min_value=0.1,
            )
            cluster_strength_z = sidebar.number_input(
                "Сила сгущения z",
                value=2.5,
                min_value=0.1,
            )
            edge_refinement_radius_m = sidebar.number_input(
                "Радиус уточнения около края [м]",
                value=4.0e-5,
                format="%.6e",
            )
            edge_refinement_points_per_edge = sidebar.number_input(
                "Точек на edge-пиксель",
                value=5,
                min_value=1,
                step=1,
            )
            max_edge_grid_points = sidebar.number_input(
                "Максимум edge-пикселей для сетки",
                value=120,
                min_value=1,
                step=1,
            )
            min_grid_spacing_m = sidebar.number_input(
                "Минимальный шаг сетки [м]",
                value=2.0e-6,
                format="%.6e",
            )
            max_active_pixels = sidebar.number_input(
                "max_active_pixels_for_direct_sum",
                value=DEFAULT_SURFACE_MAX_ACTIVE_PIXELS,
                min_value=1,
                step=100,
            )

        surface_config = SurfaceMaskConfig(
            x_size_m=x_size_m,
            y_size_m=y_size_m,
            z_max_m=z_max_m,
            min_z_m=min_z_m,
            nx=int(surface_nx),
            ny=int(surface_ny),
            nz=int(surface_nz),
            mask_threshold=mask_threshold,
            grid_mode_xy=grid_mode_xy,
            grid_mode_z=grid_mode_z,
            cluster_strength_xy=cluster_strength_xy,
            cluster_strength_z=cluster_strength_z,
            edge_refinement_radius_m=edge_refinement_radius_m,
            edge_refinement_points_per_edge=int(edge_refinement_points_per_edge),
            max_edge_grid_points=int(max_edge_grid_points),
            min_grid_spacing_m=min_grid_spacing_m,
            downsample_mask_for_computation=downsample_mask_for_computation,
            max_computational_mask_size=int(max_computational_mask_size),
            max_active_pixels_for_direct_sum=int(max_active_pixels),
        )
        surface_options = build_surface_options_from_sidebar(
            st,
            sidebar,
            uploaded_surface_mask,
            surface_config,
            mask_threshold,
        )

    particle_preset_name, particle_values = build_particle_controls(sidebar, defaults)
    particle_mass = particle_values["particle_mass"]
    environment_report = build_environment_controls(
        sidebar,
        particle_radius_m=particle_values["particle_radius"],
        particle_mass_kg=particle_mass,
    )

    with sidebar.expander("Напряжения ловушки", expanded=True):
        voltage_mode_label = sidebar.selectbox(
            "Режим напряжения",
            ["Статическое", "RF"],
            index=1,
        )
        voltage_amplitude = sidebar.number_input(
            "Масштаб поля / static voltage [В]",
            value=float(defaults.voltage_amplitude),
        )
        dc_voltage = sidebar.number_input("Постоянное напряжение / DC scale [В]", value=0.0)
        rf_voltage = sidebar.number_input("RF-амплитуда [В]", value=20.0)
        rf_frequency_hz = sidebar.number_input(
            "RF-частота [Гц]",
            value=3.0e4,
            min_value=0.0,
        )
        sidebar.info(
            "Если загруженное или рассчитанное поле уже соответствует физическим "
            "напряжениям, не умножайте его второй раз без необходимости."
        )

    with sidebar.expander("Параметры динамики", expanded=True):
        simulation_duration = sidebar.number_input(
            "Время моделирования [с]",
            value=2.0e-3,
            format="%.6e",
        )
        time_step = sidebar.number_input(
            "Базовый шаг записи [с]",
            value=float(defaults.time_step),
            format="%.6e",
        )
        confinement_radius_threshold = sidebar.number_input(
            "Порог локализации по max r [м]",
            value=float(defaults.confinement_radius_threshold),
            format="%.6e",
        )
        plot_quality = sidebar.selectbox(
            "Качество графиков",
            ["Fast preview", "Better quality"],
            format_func=lambda value: (
                "Быстрый просмотр" if value == "Fast preview" else "Лучшее качество"
            ),
        )

    with sidebar.expander("Экспертные настройки", expanded=False):
        default_mesh_cells = defaults.mesh_cells
        nx = sidebar.number_input(
            "legacy FEM mesh nx",
            value=int(default_mesh_cells[0]),
            min_value=2,
            step=1,
        )
        ny = sidebar.number_input(
            "legacy FEM mesh ny",
            value=int(default_mesh_cells[1]),
            min_value=2,
            step=1,
        )
        nz = sidebar.number_input(
            "legacy FEM mesh nz",
            value=int(default_mesh_cells[2]),
            min_value=2,
            step=1,
        )

    config = with_config_overrides(
        defaults,
        particle_mass=particle_mass,
        particle_charge=particle_values["particle_charge"],
        particle_radius=particle_values["particle_radius"],
        particle_density=particle_values["particle_density"],
        damping_coefficient=environment_report["gamma_kg_s"],
        initial_position=particle_values["initial_position"],
        initial_velocity=particle_values["initial_velocity"],
        voltage_amplitude=voltage_amplitude,
        use_time_dependent_voltage=(voltage_mode_label == "RF"),
        dc_voltage=dc_voltage,
        rf_voltage=rf_voltage,
        rf_angular_frequency=2.0 * np.pi * rf_frequency_hz,
        simulation_time=(0.0, simulation_duration),
        time_step=time_step,
        max_time_step=time_step,
        confinement_radius_threshold=confinement_radius_threshold,
        mesh_cells=(int(nx), int(ny), int(nz)),
    )

    if voltage_mode_label == "RF" and rf_frequency_hz > 0.0:
        config = make_rf_case_config(
            config,
            rf_voltage=rf_voltage,
            rf_angular_frequency=2.0 * np.pi * rf_frequency_hz,
        )

    return (
        config,
        workflow_mode,
        particle_preset_name,
        {"sweep_voltages": "5,10,20", "sweep_frequencies": "5000,30000,100000"},
        plot_quality,
        field_source,
        uploaded_field_file,
        surface_options,
        environment_report,
        source_label,
    )


def build_surface_options_from_sidebar(
    st,
    sidebar,
    uploaded_surface_mask,
    surface_config: SurfaceMaskConfig,
    mask_threshold: float,
) -> dict:
    """Parse the uploaded surface mask and component voltages."""

    mask = None
    component_labels = None
    number_of_components = 0
    electrode_potentials = {}
    voltage_map = None
    computational_mask = None
    computational_voltage_map = None
    downsample_metadata = None
    uploaded_surface_mask_bytes = None

    if uploaded_surface_mask is not None:
        try:
            uploaded_surface_mask_bytes = uploaded_surface_mask.getvalue()
            mask = load_binary_electrode_mask(
                io.BytesIO(uploaded_surface_mask_bytes),
                threshold=mask_threshold,
            )
            component_labels, number_of_components = detect_electrode_components(mask)
            st.session_state["surface_mask_bytes"] = uploaded_surface_mask_bytes
            st.session_state["surface_binary_mask"] = mask
            st.session_state["surface_component_labels"] = component_labels
            st.session_state["surface_number_of_components"] = number_of_components
            st.session_state["surface_mask_threshold"] = mask_threshold
        except Exception as exc:
            sidebar.error(f"Не удалось прочитать маску: {exc}")
    elif "surface_binary_mask" in st.session_state:
        mask = st.session_state["surface_binary_mask"]
        component_labels = st.session_state["surface_component_labels"]
        number_of_components = st.session_state["surface_number_of_components"]
        uploaded_surface_mask_bytes = st.session_state.get("surface_mask_bytes")
        sidebar.caption("Используется маска, сохранённая в текущей Streamlit-сессии.")

    if mask is not None and component_labels is not None:
        voltage_defaults = assign_four_electrode_voltages(
            component_labels,
            mask,
            surface_config,
        )
        sidebar.write(f"Найдено электродов: {number_of_components}")
        for component_id in range(1, number_of_components + 1):
            electrode_potentials[component_id] = sidebar.number_input(
                f"Электрод {component_id}: напряжение [В]",
                value=float(voltage_defaults.get(component_id, 0.0)),
                key=f"surface_electrode_voltage_{component_id}",
            )
        voltage_map = build_voltage_map_from_components(
            component_labels,
            electrode_potentials,
        )
        (
            computational_voltage_map,
            computational_mask,
            downsample_metadata,
        ) = prepare_surface_voltage_map_for_computation(
            voltage_map,
            surface_config,
            mask=mask,
        )
        st.session_state["surface_voltage_map"] = voltage_map
        st.session_state["surface_computational_voltage_map"] = computational_voltage_map
        st.session_state["surface_computational_mask"] = computational_mask
        st.session_state["surface_downsample_metadata"] = downsample_metadata

    return {
        "uploaded_surface_mask": uploaded_surface_mask,
        "uploaded_surface_mask_bytes": uploaded_surface_mask_bytes,
        "config": surface_config,
        "mask": mask,
        "component_labels": component_labels,
        "number_of_components": number_of_components,
        "electrode_potentials": electrode_potentials,
        "voltage_map": voltage_map,
        "computational_mask": computational_mask,
        "computational_voltage_map": computational_voltage_map,
        "downsample_metadata": downsample_metadata,
    }


def _assignment_rows(assignments: list[ElectrodeAssignment]) -> list[dict]:
    """Return a compact table of manual electrode assignments."""

    return [
        {
            "region_id": assignment.region_id,
            "name": assignment.name,
            "role": assignment.role,
            "static_voltage_V": assignment.voltage,
            "rf_phase_rad": assignment.rf_phase,
        }
        for assignment in assignments
    ]


def _active_voltage_pixels_from_metadata(metadata: dict | None) -> int:
    """Return active voltage pixel count from surface metadata."""

    if not metadata:
        return 0
    return int(
        metadata.get(
            "active_voltage_pixels_after",
            metadata.get("active_pixels_after", 0),
        )
    )


def warn_about_surface_active_pixels(st, metadata: dict | None, limit: int) -> None:
    """Warn when a surface direct-sum computation is likely to be slow."""

    active_pixels = _active_voltage_pixels_from_metadata(metadata)
    if active_pixels > limit:
        st.warning(
            "Слишком много активных пикселей. Уменьшите computational mask size "
            "или упростите маску."
        )
    elif active_pixels > SURFACE_ACTIVE_PIXELS_SLOW_WARNING:
        st.warning("Много активных пикселей: расчёт может быть медленным.")


def is_active_pixel_limit_error(exc: Exception) -> bool:
    """Return True for the readable direct-sum active-pixel limit error."""

    message = str(exc)
    return "active_pixels=" in message and "max_active_pixels_for_direct_sum" in message


def show_surface_computation_error(st, exc: Exception) -> None:
    """Display surface computation failures without a raw traceback by default."""

    if is_active_pixel_limit_error(exc):
        st.error(
            "Слишком много активных пикселей электродов для прямого суммирования."
        )
        st.write(
            "Что попробовать: уменьшить computational mask size, включить "
            "downsampling, упростить маску или осторожно увеличить "
            "max_active_pixels_for_direct_sum."
        )
        if st.session_state.get("expert_mode_enabled", False):
            st.exception(exc)
        else:
            st.caption(str(exc))
        return
    raise exc


def _build_surface_options_from_labeled_regions(
    *,
    source_kind: str,
    surface_config: SurfaceMaskConfig,
    mask: np.ndarray | None,
    component_labels: np.ndarray | None,
    assignments: list[ElectrodeAssignment],
    number_of_components: int,
    uploaded_surface_mask=None,
    uploaded_surface_mask_bytes=None,
    overlap_warning: str | None = None,
) -> dict:
    """Build shared surface options from labels plus manual RF/DC roles."""

    voltage_map = None
    rf_voltage_map_base = None
    dc_voltage_map = None
    computational_mask = None
    computational_voltage_map = None
    downsample_metadata = None
    if mask is not None and component_labels is not None:
        rf_voltage_map_base, dc_voltage_map = build_voltage_maps_from_assignments(
            component_labels,
            assignments,
        )
        voltage_map = dc_voltage_map + rf_voltage_map_base
        active_voltage_mask = (rf_voltage_map_base != 0.0) | (dc_voltage_map != 0.0)
        if not np.any(active_voltage_mask):
            active_voltage_mask = mask
        (
            computational_voltage_map,
            computational_mask,
            downsample_metadata,
        ) = prepare_surface_voltage_map_for_computation(
            voltage_map,
            surface_config,
            mask=active_voltage_mask,
        )

    return {
        "source_kind": source_kind,
        "uploaded_surface_mask": uploaded_surface_mask,
        "uploaded_surface_mask_bytes": uploaded_surface_mask_bytes,
        "config": surface_config,
        "mask": mask,
        "component_labels": component_labels,
        "number_of_components": number_of_components,
        "assignments": assignments,
        "electrode_potentials": {
            assignment.region_id: assignment.voltage for assignment in assignments
        },
        "voltage_map": voltage_map,
        "rf_voltage_map_base": rf_voltage_map_base,
        "dc_voltage_map": dc_voltage_map,
        "uses_rf_dc_separation": True,
        "computational_mask": computational_mask,
        "computational_voltage_map": computational_voltage_map,
        "downsample_metadata": downsample_metadata,
        "overlap_warning": overlap_warning,
    }


def build_surface_options_from_sidebar_manual_roles(
    st,
    sidebar,
    uploaded_surface_mask,
    surface_config: SurfaceMaskConfig,
    mask_threshold: float,
) -> dict:
    """Parse an uploaded electrode mask and ask the user for every role."""

    mask = None
    component_labels = None
    number_of_components = 0
    uploaded_surface_mask_bytes = None
    assignments = []

    if uploaded_surface_mask is not None:
        try:
            uploaded_surface_mask_bytes = uploaded_surface_mask.getvalue()
            mask = load_binary_electrode_mask(
                io.BytesIO(uploaded_surface_mask_bytes),
                threshold=mask_threshold,
            )
            component_labels, number_of_components = detect_electrode_components(mask)
            st.session_state["surface_mask_bytes"] = uploaded_surface_mask_bytes
            st.session_state["surface_binary_mask"] = mask
            st.session_state["surface_component_labels"] = component_labels
            st.session_state["surface_number_of_components"] = number_of_components
            st.session_state["surface_mask_threshold"] = mask_threshold
        except Exception as exc:
            sidebar.error(f"Could not read electrode mask: {exc}")
    elif "surface_binary_mask" in st.session_state:
        mask = st.session_state["surface_binary_mask"]
        component_labels = st.session_state["surface_component_labels"]
        number_of_components = st.session_state["surface_number_of_components"]
        uploaded_surface_mask_bytes = st.session_state.get("surface_mask_bytes")
        sidebar.caption("Using the mask saved in the current Streamlit session.")

    if mask is not None and component_labels is not None:
        sidebar.info(
            "Компоненты задают только геометрию электродов. "
            "Тип электрода и напряжение задаются вручную."
        )
        sidebar.write(f"Detected electrodes: {number_of_components}")
        role_labels = list(ROLE_LABEL_TO_CODE.keys())
        default_role_index = role_labels.index("GND / 0 V")
        for component_id in range(1, number_of_components + 1):
            section = sidebar.expander(
                f"Electrode {component_id}",
                expanded=component_id <= 4,
            )
            with section:
                name = section.text_input(
                    "Name",
                    value=f"electrode {component_id}",
                    key=f"surface_electrode_name_{component_id}",
                )
                role_label = section.selectbox(
                    "Role",
                    role_labels,
                    index=default_role_index,
                    key=f"surface_electrode_role_{component_id}",
                )
                role = ROLE_LABEL_TO_CODE[role_label]
                voltage = 0.0
                if role in {ROLE_DC, ROLE_CUSTOM}:
                    voltage = section.number_input(
                        "Static voltage [V]",
                        value=0.0,
                        key=f"surface_electrode_voltage_{component_id}",
                    )
                assignments.append(
                    ElectrodeAssignment(
                        region_id=component_id,
                        name=name,
                        role=role,
                        voltage=float(voltage),
                    )
                )
        if assignments and all(assignment.role == ROLE_GND for assignment in assignments):
            sidebar.warning(
                "All electrodes are GND. Select RF, DC, or CUSTOM to create a nonzero field."
            )

    surface_options = _build_surface_options_from_labeled_regions(
        source_kind="image_mask",
        surface_config=surface_config,
        mask=mask,
        component_labels=component_labels,
        assignments=assignments,
        number_of_components=number_of_components,
        uploaded_surface_mask=uploaded_surface_mask,
        uploaded_surface_mask_bytes=uploaded_surface_mask_bytes,
    )
    if surface_options["voltage_map"] is not None:
        st.session_state["surface_voltage_map"] = surface_options["voltage_map"]
        st.session_state["surface_rf_voltage_map_base"] = surface_options[
            "rf_voltage_map_base"
        ]
        st.session_state["surface_dc_voltage_map"] = surface_options["dc_voltage_map"]
        st.session_state["surface_computational_voltage_map"] = surface_options[
            "computational_voltage_map"
        ]
        st.session_state["surface_computational_mask"] = surface_options[
            "computational_mask"
        ]
        st.session_state["surface_downsample_metadata"] = surface_options[
            "downsample_metadata"
        ]
    return surface_options


def _definition_to_state_dict(definition: ElectrodeRegionDefinition) -> dict:
    """Convert a preset definition into Streamlit session-state data."""

    return {
        "name": definition.name,
        "expression": definition.expression,
        "role": definition.role,
        "voltage": float(definition.voltage),
        "rf_phase": float(definition.rf_phase),
    }


def _state_dict_to_definition(data: dict) -> ElectrodeRegionDefinition:
    """Convert one editable UI dictionary into a region definition."""

    return ElectrodeRegionDefinition(
        name=str(data.get("name", "electrode")),
        expression=str(data.get("expression", "False")),
        role=str(data.get("role", ROLE_GND)),
        voltage=float(data.get("voltage", 0.0)),
        rf_phase=float(data.get("rf_phase", 0.0)),
    )


def _load_function_geometry_preset(preset_name: str):
    """Replace editable function-electrode definitions with a preset."""

    presets = function_geometry_presets()
    st_defs = [_definition_to_state_dict(definition) for definition in presets[preset_name]]
    return st_defs


def render_function_geometry_controls(
    st,
    container,
    surface_config: SurfaceMaskConfig,
) -> dict:
    """Render Desmos-like function electrode controls and preview maps."""

    container.info(
        "Электроды задаются неравенствами по x и y. Поздние определения "
        "перекрывают ранние, если области пересекаются."
    )
    presets = function_geometry_presets()
    if "function_electrode_defs" not in st.session_state:
        first_preset_name = next(iter(presets))
        st.session_state["function_electrode_defs"] = _load_function_geometry_preset(
            first_preset_name
        )

    preset_cols = container.columns([2, 1, 1])
    preset_name = preset_cols[0].selectbox(
        "Preset",
        list(presets.keys()),
        key="function_geometry_preset",
    )
    if preset_cols[1].button("Загрузить пресет", key="load_function_preset"):
        st.session_state["function_electrode_defs"] = _load_function_geometry_preset(
            preset_name
        )
        st.rerun()
    if preset_cols[2].button("Добавить электрод", key="add_function_electrode"):
        st.session_state["function_electrode_defs"].append(
            {
                "name": f"electrode {len(st.session_state['function_electrode_defs']) + 1}",
                "expression": "abs(x) < 100e-6 and abs(y) < 100e-6",
                "role": ROLE_GND,
                "voltage": 0.0,
                "rf_phase": 0.0,
            }
        )
        st.rerun()

    definitions_state = st.session_state["function_electrode_defs"]
    if len(definitions_state) > 1 and container.button(
        "Удалить последний электрод",
        key="remove_function_electrode",
    ):
        definitions_state.pop()
        st.rerun()

    role_labels = list(ROLE_LABEL_TO_CODE.keys())
    for index, data in enumerate(definitions_state):
        section = container.expander(
            f"Electrode {index + 1}: {data.get('name', '')}",
            expanded=index < 4,
        )
        with section:
            cols = section.columns([2, 1, 1])
            data["name"] = cols[0].text_input(
                "Name",
                value=str(data.get("name", f"electrode {index + 1}")),
                key=f"function_name_{index}",
            )
            current_label = ROLE_CODE_TO_LABEL.get(str(data.get("role", ROLE_GND)), "GND / 0 V")
            role_label = cols[1].selectbox(
                "Role",
                role_labels,
                index=role_labels.index(current_label),
                key=f"function_role_{index}",
            )
            data["role"] = ROLE_LABEL_TO_CODE[role_label]
            previous_voltage = float(data.get("voltage", 0.0))
            if data["role"] in {ROLE_DC, ROLE_CUSTOM}:
                data["voltage"] = cols[2].number_input(
                    "Static V [V]",
                    value=previous_voltage,
                    key=f"function_voltage_{index}",
                )
            else:
                data["voltage"] = 0.0
                cols[2].caption("No static voltage")
            data["expression"] = section.text_area(
                "Expression",
                value=str(data.get("expression", "False")),
                key=f"function_expression_{index}",
                height=72,
            )

    definitions = [_state_dict_to_definition(data) for data in definitions_state]
    try:
        raster = rasterize_function_regions(
            definitions,
            x_size_m=surface_config.x_size_m,
            y_size_m=surface_config.y_size_m,
            nx_mask=int(st.session_state.get("function_nx_mask", 192)),
            ny_mask=int(st.session_state.get("function_ny_mask", 192)),
        )
        labels = raster.region_labels
        mask = labels > 0
        surface_options = _build_surface_options_from_labeled_regions(
            source_kind="function",
            surface_config=surface_config,
            mask=mask,
            component_labels=labels,
            assignments=raster.assignments,
            number_of_components=len(raster.assignments),
            overlap_warning=raster.overlap_warning,
        )
        if raster.overlap_warning:
            container.warning(raster.overlap_warning)
        if not np.any(mask):
            container.warning("Function definitions produced an empty electrode mask.")
        show_surface_mask_section(container, surface_options)
        return surface_options
    except Exception as exc:
        container.error("Could not rasterize function-defined electrodes.")
        container.exception(exc)
        return _build_surface_options_from_labeled_regions(
            source_kind="function",
            surface_config=surface_config,
            mask=None,
            component_labels=None,
            assignments=[],
            number_of_components=0,
        )


def _assignments_to_session_dict(assignments: list[ElectrodeAssignment]) -> dict:
    """Store assignment defaults in Streamlit session state."""

    return {
        str(assignment.region_id): {
            "name": assignment.name,
            "role": assignment.role,
            "voltage": float(assignment.voltage),
            "rf_phase": float(assignment.rf_phase),
        }
        for assignment in assignments
    }


def _render_canvas_assignment_controls(
    container,
    number_of_components: int,
    defaults_by_region_id: dict,
) -> list[ElectrodeAssignment]:
    """Render compact manual role controls for canvas components."""

    assignments = []
    role_labels = list(ROLE_LABEL_TO_CODE.keys())
    default_role_index = role_labels.index("GND / 0 V")
    for region_id in range(1, number_of_components + 1):
        defaults = defaults_by_region_id.get(str(region_id), {})
        section = container.expander(
            f"Компонент {region_id}",
            expanded=region_id <= 5,
        )
        with section:
            cols = section.columns([2, 1, 1])
            name = cols[0].text_input(
                "Имя",
                value=str(defaults.get("name", f"canvas electrode {region_id}")),
                key=f"canvas_electrode_name_{region_id}",
            )
            current_role = str(defaults.get("role", ROLE_GND))
            current_label = ROLE_CODE_TO_LABEL.get(current_role, "GND / 0 V")
            role_label = cols[1].selectbox(
                "Роль",
                role_labels,
                index=role_labels.index(current_label)
                if current_label in role_labels
                else default_role_index,
                key=f"canvas_electrode_role_{region_id}",
            )
            role = ROLE_LABEL_TO_CODE[role_label]
            voltage = 0.0
            if role in {ROLE_DC, ROLE_CUSTOM}:
                voltage = cols[2].number_input(
                    "Static voltage [V]",
                    value=float(defaults.get("voltage", 0.0)),
                    key=f"canvas_electrode_voltage_{region_id}",
                )
            else:
                cols[2].caption("0 V static")
            assignments.append(
                ElectrodeAssignment(
                    region_id=region_id,
                    name=name,
                    role=role,
                    voltage=float(voltage),
                    rf_phase=float(defaults.get("rf_phase", 0.0)),
                )
            )
    return assignments


def render_canvas_geometry_controls(st, container, expert_mode: bool) -> dict:
    """Render an interactive/fallback canvas electrode editor."""

    tabs = container.tabs(
        [
            "Рисование",
            "Компоненты",
            "Назначение электродов",
            "Карты RF/DC",
            "Расчёт поля",
        ]
    )
    drawing_tab, components_tab, assignment_tab, maps_tab, field_tab = tabs

    if "canvas_editor_version" not in st.session_state:
        st.session_state["canvas_editor_version"] = 0
    if "canvas_assignments" not in st.session_state:
        st.session_state["canvas_assignments"] = {}

    with drawing_tab:
        drawing_tab.info(
            "Нарисованные области задают только геометрию. Тип электрода "
            "и напряжение задаются ниже вручную."
        )
        uploaded_design = drawing_tab.file_uploader(
            "Загрузить дизайн JSON",
            type=["json"],
            key="canvas_design_upload",
        )
        if uploaded_design is not None and drawing_tab.button(
            "Применить дизайн JSON",
            key="apply_canvas_design_json",
        ):
            try:
                loaded_design = canvas_design_from_dict(
                    json.loads(uploaded_design.getvalue().decode("utf-8"))
                )
                st.session_state["canvas_confirmed_mask"] = loaded_design["binary_mask"]
                st.session_state["canvas_assignments"] = _assignments_to_session_dict(
                    loaded_design["assignments"]
                )
                st.session_state["canvas_loaded_x_size_m"] = loaded_design["x_size_m"]
                st.session_state["canvas_loaded_y_size_m"] = loaded_design["y_size_m"]
                st.session_state["canvas_loaded_resolution_px"] = loaded_design[
                    "canvas_resolution_px"
                ]
                st.success("Дизайн JSON загружен.")
                st.rerun()
            except Exception as exc:
                drawing_tab.error("Не удалось загрузить дизайн JSON.")
                drawing_tab.exception(exc)

        size_cols = drawing_tab.columns(3)
        x_size_m = size_cols[0].number_input(
            "Физический размер x [m]",
            value=float(st.session_state.get("canvas_loaded_x_size_m", 1.0e-3)),
            format="%.6e",
            key="canvas_x_size",
        )
        y_size_m = size_cols[1].number_input(
            "Физический размер y [m]",
            value=float(st.session_state.get("canvas_loaded_y_size_m", 1.0e-3)),
            format="%.6e",
            key="canvas_y_size",
        )
        canvas_resolution_px = int(
            size_cols[2].number_input(
                "Canvas resolution [px]",
                value=int(st.session_state.get("canvas_loaded_resolution_px", 256)),
                min_value=64,
                max_value=768,
                step=32,
                key="canvas_resolution_px",
            )
        )

        draw_cols = drawing_tab.columns(4)
        brush_size = int(
            draw_cols[0].slider(
                "Brush size",
                min_value=2,
                max_value=60,
                value=18,
                key="canvas_brush_size",
            )
        )
        drawing_mode_label = draw_cols[1].selectbox(
            "Drawing mode",
            ["free draw", "rectangle", "circle", "polygon"],
            key="canvas_drawing_mode",
        )
        drawing_modes = {
            "free draw": "freedraw",
            "rectangle": "rect",
            "circle": "circle",
            "polygon": "polygon",
        }
        min_component_area = int(
            draw_cols[2].number_input(
                "min area [px]",
                value=24,
                min_value=1,
                step=1,
                key="canvas_min_component_area",
            )
        )
        if draw_cols[3].button("Очистить canvas", key="clear_canvas"):
            st.session_state.pop("canvas_confirmed_mask", None)
            st.session_state.pop("canvas_current_mask", None)
            st.session_state["canvas_assignments"] = {}
            st.session_state["canvas_editor_version"] += 1
            st.rerun()

        canvas_image = None
        if st_canvas is not None:
            try:
                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 1)",
                    stroke_width=brush_size,
                    stroke_color="#000000",
                    background_color="#ffffff",
                    height=canvas_resolution_px,
                    width=canvas_resolution_px,
                    drawing_mode=drawing_modes[drawing_mode_label],
                    key=f"canvas_editor_{st.session_state['canvas_editor_version']}",
                )
                if canvas_result.image_data is not None:
                    canvas_image = canvas_result.image_data
            except Exception as exc:
                drawing_tab.warning(
                    "Canvas component failed. You can use PNG upload fallback below."
                )
                drawing_tab.exception(exc)
        else:
            drawing_tab.warning(
                "streamlit-drawable-canvas is not installed. "
                "Install requirements or upload a drawn PNG mask below."
            )

        fallback_png = drawing_tab.file_uploader(
            "Fallback: загрузить нарисованный PNG/JPG",
            type=["png", "jpg", "jpeg"],
            key="canvas_fallback_png",
        )
        if fallback_png is not None:
            canvas_image = np.asarray(Image.open(fallback_png).convert("RGBA"))

        if canvas_image is not None:
            raw_mask = canvas_image_to_binary_mask(canvas_image)
            st.session_state["canvas_current_mask"] = raw_mask
            drawing_tab.image(canvas_image.astype(np.uint8), caption="Canvas preview")
            if drawing_tab.button("Подтвердить геометрию", key="confirm_canvas_geometry"):
                st.session_state["canvas_confirmed_mask"] = raw_mask
                drawing_tab.success("Геометрия подтверждена.")
        elif "canvas_confirmed_mask" not in st.session_state:
            drawing_tab.info(
                "Электроды не найдены. Нарисуйте хотя бы одну область или загрузите PNG."
            )

    mask = st.session_state.get(
        "canvas_confirmed_mask",
        st.session_state.get("canvas_current_mask"),
    )
    if mask is None:
        return _build_surface_options_from_labeled_regions(
            source_kind="canvas",
            surface_config=SurfaceMaskConfig(),
            mask=None,
            component_labels=None,
            assignments=[],
            number_of_components=0,
        )

    labels, number_of_components, cleaned_mask = label_canvas_electrodes(
        mask,
        min_component_area=min_component_area,
    )
    build_regions_from_canvas_mask(cleaned_mask, min_component_area=1)

    with components_tab:
        if number_of_components == 0:
            components_tab.warning(
                "Электроды не найдены. Нарисуйте хотя бы одну область."
            )
        elif number_of_components > 20:
            components_tab.warning(
                "Найдено много мелких компонент. Увеличьте min area или очистите рисунок."
            )
        components_tab.write(f"Найдено компонент: {number_of_components}")
        comp_cols = components_tab.columns(2)
        comp_cols[0].image(cleaned_mask.astype(float), caption="Binary mask")
        labels_fig, labels_ax = plt.subplots(figsize=(4, 4))
        labels_image = labels_ax.imshow(labels, origin="upper", cmap="tab20")
        labels_ax.set_title("Labeled regions")
        labels_ax.set_axis_off()
        labels_fig.colorbar(labels_image, ax=labels_ax, fraction=0.046)
        comp_cols[1].pyplot(labels_fig)
        plt.close(labels_fig)

    with field_tab:
        field_tab.info(
            "Псевдопотенциал считается только по RF-полю. DC-электроды "
            "учитываются в полной динамике как статическое поле."
        )
        grid_cols = field_tab.columns(4)
        min_z_m = grid_cols[0].number_input(
            "min z [m]",
            value=2.0e-5,
            format="%.6e",
            key="canvas_min_z",
        )
        z_max_m = grid_cols[1].number_input(
            "z max [m]",
            value=1.0e-3,
            format="%.6e",
            key="canvas_z_max",
        )
        surface_nx = grid_cols[2].number_input(
            "field nx",
            value=25,
            min_value=3,
            step=1,
            key="canvas_surface_nx",
        )
        surface_ny = grid_cols[3].number_input(
            "field ny",
            value=25,
            min_value=3,
            step=1,
            key="canvas_surface_ny",
        )
        more_grid_cols = field_tab.columns(4)
        surface_nz = more_grid_cols[0].number_input(
            "field nz",
            value=15,
            min_value=3,
            step=1,
            key="canvas_surface_nz",
        )
        grid_mode_xy = more_grid_cols[1].selectbox(
            "Сетка x-y",
            ["uniform", "center_clustered_tanh", "edge_aware"],
            index=2,
            key="canvas_grid_mode_xy",
        )
        grid_mode_z = more_grid_cols[2].selectbox(
            "Сетка z",
            ["uniform", "near_surface_clustered"],
            index=1,
            key="canvas_grid_mode_z",
        )
        max_computational_mask_size = int(
            more_grid_cols[3].slider(
                "max comp mask",
                min_value=32,
                max_value=512,
                value=192,
                step=16,
                key="canvas_max_comp_mask",
            )
        )
        if max_computational_mask_size < 96:
            field_tab.warning(
                "Маска для расчёта грубая. Быстро, но поле около краёв "
                "электродов будет неточным."
            )
        max_active_pixels = DEFAULT_SURFACE_MAX_ACTIVE_PIXELS
        if expert_mode:
            expert_canvas_grid = field_tab.expander("Экспертные настройки")
            with expert_canvas_grid:
                max_active_pixels = expert_canvas_grid.number_input(
                    "max_active_pixels_for_direct_sum",
                    value=DEFAULT_SURFACE_MAX_ACTIVE_PIXELS,
                    min_value=1,
                    step=100,
                    key="canvas_max_active",
                )

    surface_config = SurfaceMaskConfig(
        x_size_m=x_size_m,
        y_size_m=y_size_m,
        z_max_m=z_max_m,
        min_z_m=min_z_m,
        nx=int(surface_nx),
        ny=int(surface_ny),
        nz=int(surface_nz),
        grid_mode_xy=grid_mode_xy,
        grid_mode_z=grid_mode_z,
        downsample_mask_for_computation=True,
        max_computational_mask_size=max_computational_mask_size,
        max_active_pixels_for_direct_sum=int(max_active_pixels),
    )

    assignments = []
    if number_of_components > 0:
        with assignment_tab:
            assignment_tab.info(
                "Компоненты задают только геометрию. RF/DC/GND/CUSTOM "
                "назначаются вручную."
            )
            assignments = _render_canvas_assignment_controls(
                assignment_tab,
                number_of_components,
                st.session_state.get("canvas_assignments", {}),
            )
            st.session_state["canvas_assignments"] = _assignments_to_session_dict(
                assignments
            )
    rf_map, dc_map = canvas_design_to_voltage_maps(labels, assignments)
    if not np.any(rf_map):
        maps_tab.warning(
            "RF-напряжение может быть задано, но RF-электроды не выбраны. "
            "Псевдопотенциал недоступен без RF-электродов."
        )

    surface_options = _build_surface_options_from_labeled_regions(
        source_kind="canvas",
        surface_config=surface_config,
        mask=cleaned_mask,
        component_labels=labels,
        assignments=assignments,
        number_of_components=number_of_components,
    )
    surface_options["canvas_binary_mask"] = cleaned_mask

    with maps_tab:
        if number_of_components > 0:
            show_surface_mask_section(maps_tab, surface_options)
            design = canvas_design_to_dict(
                x_size_m=x_size_m,
                y_size_m=y_size_m,
                canvas_resolution_px=canvas_resolution_px,
                binary_mask=cleaned_mask,
                assignments=assignments,
                notes="Canvas surface electrode design.",
            )
            maps_tab.download_button(
                "Скачать дизайн JSON",
                data=json.dumps(design, ensure_ascii=False, indent=2),
                file_name="canvas_electrode_design.json",
                mime="application/json",
                key="download_canvas_design_json",
            )

    return surface_options


def build_particle_controls(sidebar, defaults) -> tuple[str, dict]:
    """Build Russian particle controls and return physical particle values."""

    with sidebar.expander("Параметры частицы", expanded=True):
        particle_preset_name = sidebar.selectbox(
            "Пресет частицы",
            list(PARTICLE_PRESETS.keys()),
            index=2,
        )
        particle_preset = PARTICLE_PRESETS[particle_preset_name]
        derive_mass = sidebar.checkbox(
            "Вычислять массу по радиусу и плотности",
            value=bool(particle_preset.get("derive_mass", False)),
        )
        particle_charge = sidebar.number_input(
            "Заряд q [Кл]",
            value=float(particle_preset.get("particle_charge", defaults.particle_charge)),
            format="%.6e",
        )
        particle_radius = sidebar.number_input(
            "Радиус частицы [м]",
            value=float(particle_preset.get("particle_radius", 1.0e-6)),
            format="%.6e",
        )
        particle_density = sidebar.number_input(
            "Плотность частицы [кг/м³]",
            value=float(particle_preset.get("particle_density", 2200.0)),
            format="%.6e",
        )
        manual_mass_default = float(
            particle_preset.get("particle_mass", defaults.particle_mass)
        )
        manual_mass = sidebar.number_input(
            "Масса m [кг]",
            value=manual_mass_default,
            format="%.6e",
            disabled=derive_mass,
        )
        if derive_mass:
            particle_mass = particle_mass_from_radius_density(
                particle_radius,
                particle_density,
            )
            sidebar.metric("Вычисленная масса [кг]", f"{particle_mass:.3e}")
        else:
            particle_mass = manual_mass

        default_position = particle_preset.get(
            "initial_position",
            defaults.initial_position,
        )
        default_velocity = particle_preset.get(
            "initial_velocity",
            defaults.initial_velocity,
        )
        initial_position = (
            sidebar.number_input("x0 [м]", value=float(default_position[0]), format="%.6e"),
            sidebar.number_input("y0 [м]", value=float(default_position[1]), format="%.6e"),
            sidebar.number_input("z0 [м]", value=float(default_position[2]), format="%.6e"),
        )
        initial_velocity = (
            sidebar.number_input("vx0 [м/с]", value=float(default_velocity[0]), format="%.6e"),
            sidebar.number_input("vy0 [м/с]", value=float(default_velocity[1]), format="%.6e"),
            sidebar.number_input("vz0 [м/с]", value=float(default_velocity[2]), format="%.6e"),
        )

    return particle_preset_name, {
        "particle_mass": particle_mass,
        "particle_charge": particle_charge,
        "particle_radius": particle_radius,
        "particle_density": particle_density,
        "initial_position": initial_position,
        "initial_velocity": initial_velocity,
    }


def build_environment_controls(
    sidebar,
    particle_radius_m: float,
    particle_mass_kg: float,
) -> dict:
    """Build Russian gas/friction controls and return damping diagnostics."""

    mode_labels = {
        "Вакуум: трение выключено": "vacuum",
        "Воздух: автоматическая модель трения": "air_auto",
        "Газ при заданном давлении": "pressure_gas",
        "Пользовательское трение gamma": "custom",
    }
    with sidebar.expander("Среда и трение", expanded=True):
        sidebar.info(DRAG_WARNING_TEXT)
        mode_label = sidebar.selectbox("Режим среды", list(mode_labels.keys()))
        pressure_pa = sidebar.number_input("Давление [Па]", value=101325.0, format="%.6e")
        temperature_k = sidebar.number_input("Температура [К]", value=293.15)
        gas_viscosity_pa_s = sidebar.number_input(
            "Вязкость газа [Па·с]",
            value=1.8e-5,
            format="%.6e",
        )
        gas_molecular_mass_kg = sidebar.number_input(
            "Молекулярная масса газа [кг]",
            value=4.81e-26,
            format="%.6e",
        )
        custom_gamma_kg_s = sidebar.number_input(
            "Пользовательский gamma [кг/с]",
            value=1.0e-15,
            format="%.6e",
        )

    env_config = EnvironmentConfig(
        environment_mode=mode_labels[mode_label],
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
        gas_viscosity_pa_s=gas_viscosity_pa_s,
        gas_molecular_mass_kg=gas_molecular_mass_kg,
        custom_gamma_kg_s=custom_gamma_kg_s,
    )
    report = compute_damping_gamma(
        particle_radius_m,
        particle_mass_kg,
        env_config,
    )
    report["mode_label"] = mode_label
    report["environment_config"] = env_config
    return report


def render_particle_card(container, defaults) -> tuple[str, dict]:
    """Render the compact particle card and return particle parameters."""

    particle_preset_name = container.selectbox(
        "Пресет частицы",
        list(PARTICLE_PRESETS.keys()),
        index=2,
        key="dash_particle_preset",
    )
    particle_preset = PARTICLE_PRESETS[particle_preset_name]

    left, right = container.columns(2)
    with left:
        particle_charge = left.number_input(
            "Заряд q [Кл]",
            value=float(particle_preset.get("particle_charge", defaults.particle_charge)),
            format="%.6e",
            key="dash_particle_charge",
        )
        derive_mass = left.checkbox(
            "Вычислять массу по радиусу и плотности",
            value=bool(particle_preset.get("derive_mass", False)),
            key="dash_derive_mass",
        )
    with right:
        particle_radius = right.number_input(
            "Радиус [м]",
            value=float(particle_preset.get("particle_radius", 1.0e-6)),
            format="%.6e",
            key="dash_particle_radius",
        )
        particle_density = right.number_input(
            "Плотность [кг/м³]",
            value=float(particle_preset.get("particle_density", 2200.0)),
            format="%.6e",
            key="dash_particle_density",
        )

    manual_mass = container.number_input(
        "Масса m [кг]",
        value=float(particle_preset.get("particle_mass", defaults.particle_mass)),
        format="%.6e",
        disabled=derive_mass,
        key="dash_particle_mass",
    )
    if derive_mass:
        particle_mass = particle_mass_from_radius_density(particle_radius, particle_density)
        container.metric("Вычисленная масса [кг]", f"{particle_mass:.3e}")
    else:
        particle_mass = manual_mass

    return particle_preset_name, {
        "particle_mass": particle_mass,
        "particle_charge": particle_charge,
        "particle_radius": particle_radius,
        "particle_density": particle_density,
    }


def render_environment_card(container, particle_radius_m: float, particle_mass_kg: float) -> dict:
    """Render compact gas/friction controls and return damping diagnostics."""

    mode_labels = {
        "Вакуум: трение выключено": "vacuum",
        "Воздух: автоматическая модель": "air_auto",
        "Газ при заданном давлении": "pressure_gas",
        "Пользовательское gamma": "custom",
    }
    mode_label = container.selectbox(
        "Режим среды",
        list(mode_labels.keys()),
        key="dash_environment_mode",
    )
    env_mode = mode_labels[mode_label]

    common_cols = container.columns(2)
    pressure_pa = common_cols[0].number_input(
        "Давление [Па]",
        value=101325.0,
        format="%.6e",
        disabled=(env_mode == "vacuum"),
        key="dash_pressure",
    )
    temperature_k = common_cols[1].number_input(
        "Температура [К]",
        value=293.15,
        disabled=(env_mode == "vacuum"),
        key="dash_temperature",
    )

    extra = container.expander("Дополнительно", expanded=False)
    with extra:
        gas_viscosity_pa_s = extra.number_input(
            "Вязкость газа [Па·с]",
            value=1.8e-5,
            format="%.6e",
            disabled=(env_mode == "vacuum"),
            key="dash_viscosity",
        )
        gas_molecular_mass_kg = extra.number_input(
            "Молекулярная масса газа [кг]",
            value=4.81e-26,
            format="%.6e",
            disabled=(env_mode == "vacuum"),
            key="dash_molecular_mass",
        )
        custom_gamma_kg_s = extra.number_input(
            "Пользовательский gamma [кг/с]",
            value=1.0e-15,
            format="%.6e",
            disabled=(env_mode != "custom"),
            key="dash_custom_gamma",
        )

    env_config = EnvironmentConfig(
        environment_mode=env_mode,
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
        gas_viscosity_pa_s=gas_viscosity_pa_s,
        gas_molecular_mass_kg=gas_molecular_mass_kg,
        custom_gamma_kg_s=custom_gamma_kg_s,
    )
    report = compute_damping_gamma(particle_radius_m, particle_mass_kg, env_config)
    report["mode_label"] = mode_label
    report["environment_config"] = env_config

    metrics = container.columns(3)
    metrics[0].metric("gamma [кг/с]", f"{report['gamma_kg_s']:.3e}")
    metrics[1].metric("Kn", _format_optional_float(report["knudsen_number"]))
    metrics[2].metric("m/gamma [с]", _format_optional_float(report["damping_time_s"]))
    container.caption(DRAG_WARNING_TEXT)
    return report


def render_voltage_card(container, defaults) -> dict:
    """Render compact voltage controls."""

    mode_label = container.selectbox(
        "Режим питания",
        ["RF", "Статическое"],
        key="dash_voltage_mode",
    )
    cols = container.columns(2)
    dc_voltage = cols[0].number_input(
        "DC scale [В]",
        value=0.0,
        key="dash_dc_voltage",
    )
    voltage_amplitude = cols[1].number_input(
        "Масштаб поля / static [В]",
        value=float(defaults.voltage_amplitude),
        key="dash_voltage_amplitude",
    )
    rf_cols = container.columns(2)
    rf_voltage = rf_cols[0].number_input(
        "RF-амплитуда [В]",
        value=20.0,
        key="dash_rf_voltage",
    )
    rf_frequency_hz = rf_cols[1].number_input(
        "RF-частота [Гц]",
        value=3.0e4,
        min_value=0.0,
        key="dash_rf_frequency",
    )
    container.info(
        "Если поле уже рассчитано для физических напряжений, не масштабируйте "
        "его второй раз без необходимости."
    )
    return {
        "use_time_dependent_voltage": mode_label == "RF",
        "voltage_amplitude": voltage_amplitude,
        "dc_voltage": dc_voltage,
        "rf_voltage": rf_voltage,
        "rf_frequency_hz": rf_frequency_hz,
    }


def render_initial_conditions_card(container, defaults, expert_mode: bool) -> dict:
    """Render initial conditions and simulation-time controls."""

    default_position = PARTICLE_PRESETS["Тестовая частица"]["initial_position"]
    default_velocity = PARTICLE_PRESETS["Тестовая частица"]["initial_velocity"]
    pos_cols = container.columns(3)
    initial_position = (
        pos_cols[0].number_input("x0 [м]", value=float(default_position[0]), format="%.6e", key="dash_x0"),
        pos_cols[1].number_input("y0 [м]", value=float(default_position[1]), format="%.6e", key="dash_y0"),
        pos_cols[2].number_input("z0 [м]", value=float(default_position[2]), format="%.6e", key="dash_z0"),
    )
    vel_cols = container.columns(3)
    initial_velocity = (
        vel_cols[0].number_input("vx0 [м/с]", value=float(default_velocity[0]), format="%.6e", key="dash_vx0"),
        vel_cols[1].number_input("vy0 [м/с]", value=float(default_velocity[1]), format="%.6e", key="dash_vy0"),
        vel_cols[2].number_input("vz0 [м/с]", value=float(default_velocity[2]), format="%.6e", key="dash_vz0"),
    )
    sim_cols = container.columns(3)
    simulation_duration = sim_cols[0].number_input(
        "Время расчёта [с]",
        value=2.0e-3,
        format="%.6e",
        key="dash_sim_time",
    )
    time_step = sim_cols[1].number_input(
        "Шаг записи / max step [с]",
        value=float(defaults.time_step),
        format="%.6e",
        key="dash_time_step",
    )
    confinement_radius_threshold = sim_cols[2].number_input(
        "Порог локализации [м]",
        value=float(defaults.confinement_radius_threshold),
        format="%.6e",
        key="dash_confinement_radius",
    )

    ode_rtol = defaults.ode_rtol
    ode_atol = defaults.ode_atol
    if expert_mode:
        solver_extra = container.expander("Экспертные настройки solver", expanded=False)
        with solver_extra:
            ode_rtol = solver_extra.number_input("ode_rtol", value=float(defaults.ode_rtol), format="%.6e", key="dash_ode_rtol")
            ode_atol = solver_extra.number_input("ode_atol", value=float(defaults.ode_atol), format="%.6e", key="dash_ode_atol")

    return {
        "initial_position": initial_position,
        "initial_velocity": initial_velocity,
        "simulation_duration": simulation_duration,
        "time_step": time_step,
        "confinement_radius_threshold": confinement_radius_threshold,
        "ode_rtol": ode_rtol,
        "ode_atol": ode_atol,
    }


def render_field_source_card(st, container, expert_mode: bool) -> tuple[str, str, object, dict | None, tuple[int, int, int]]:
    """Render source-specific field controls in the main dashboard."""

    source_label = container.selectbox(
        "Источник поля",
        [
            SOURCE_SURFACE,
            SOURCE_FUNCTIONS,
            SOURCE_CANVAS,
            SOURCE_FIELD_NPZ,
            SOURCE_POTENTIAL_NPZ,
            SOURCE_LEGACY_FEM,
        ],
        key="dash_source_label",
    )
    field_source = FIELD_SOURCE_TO_INTERNAL[source_label]
    uploaded_field_file = None
    surface_options = None
    mesh_cells = DEFAULT_CONFIG.mesh_cells

    if source_label == SOURCE_LEGACY_FEM:
        container.warning(
            "Legacy FEM оставлен для отладки пайплайна. Для surface-геометрии "
            "лучше использовать рисунок электродов или импорт .npz."
        )
        if expert_mode:
            mesh_cols = container.columns(3)
            mesh_cells = (
                int(mesh_cols[0].number_input("legacy nx", value=10, min_value=2, step=1, key="dash_mesh_nx")),
                int(mesh_cols[1].number_input("legacy ny", value=10, min_value=2, step=1, key="dash_mesh_ny")),
                int(mesh_cols[2].number_input("legacy nz", value=10, min_value=2, step=1, key="dash_mesh_nz")),
            )

    elif source_label in {SOURCE_FIELD_NPZ, SOURCE_POTENTIAL_NPZ}:
        uploaded_field_file = container.file_uploader(
            "Файл поля или потенциала .npz",
            type=["npz"],
            key="dash_npz_upload",
        )
        if source_label == SOURCE_POTENTIAL_NPZ:
            container.info("Для потенциала приложение вычислит E = -grad(phi).")
        show_field_validation_section(container, field_source, uploaded_field_file)

    elif source_label == SOURCE_CANVAS:
        canvas_card = render_card(
            container,
            "Редактор электродов",
            "Нарисуйте surface-электроды, затем вручную назначьте RF/DC/GND/CUSTOM роли.",
        )
        surface_options = render_canvas_geometry_controls(
            st,
            canvas_card,
            expert_mode,
        )

    elif source_label == SOURCE_FUNCTIONS:
        geometry_tab, grid_tab = container.tabs(["Электроды", "Область расчёта"])
        with grid_tab:
            size_cols = grid_tab.columns(2)
            x_size_m = size_cols[0].number_input("Размер x [m]", value=1.0e-3, format="%.6e", key="function_x_size")
            y_size_m = size_cols[1].number_input("Размер y [m]", value=1.0e-3, format="%.6e", key="function_y_size")
            mask_cols = grid_tab.columns(2)
            nx_mask = mask_cols[0].number_input("mask nx", value=192, min_value=16, step=16, key="function_nx_mask")
            ny_mask = mask_cols[1].number_input("mask ny", value=192, min_value=16, step=16, key="function_ny_mask")
            z_cols = grid_tab.columns(2)
            min_z_m = z_cols[0].number_input("min z [m]", value=2.0e-5, format="%.6e", key="function_min_z")
            z_max_m = z_cols[1].number_input("z max [m]", value=1.0e-3, format="%.6e", key="function_z_max")
            n_cols = grid_tab.columns(3)
            surface_nx = n_cols[0].number_input("field nx", value=25, min_value=3, step=1, key="function_surface_nx")
            surface_ny = n_cols[1].number_input("field ny", value=25, min_value=3, step=1, key="function_surface_ny")
            surface_nz = n_cols[2].number_input("field nz", value=15, min_value=3, step=1, key="function_surface_nz")
            mode_cols = grid_tab.columns(2)
            grid_mode_xy = mode_cols[0].selectbox(
                "Сетка x-y",
                ["uniform", "center_clustered_tanh", "edge_aware"],
                index=2,
                key="function_grid_mode_xy",
            )
            grid_mode_z = mode_cols[1].selectbox(
                "Сетка z",
                ["uniform", "near_surface_clustered"],
                index=1,
                key="function_grid_mode_z",
            )
            downsample_mask_for_computation = grid_tab.checkbox(
                "Уменьшать маску для расчёта",
                value=True,
                key="function_downsample_mask",
            )
            max_computational_mask_size = grid_tab.slider(
                "Максимальный размер маски для расчёта",
                min_value=32,
                max_value=512,
                value=192,
                step=16,
                key="function_max_comp_mask",
            )
            max_active_pixels = DEFAULT_SURFACE_MAX_ACTIVE_PIXELS
            if expert_mode:
                expert_grid = grid_tab.expander("Экспертные настройки сетки")
                with expert_grid:
                    max_active_pixels = expert_grid.number_input(
                        "max_active_pixels_for_direct_sum",
                        value=DEFAULT_SURFACE_MAX_ACTIVE_PIXELS,
                        min_value=1,
                        step=100,
                        key="function_max_active",
                    )
        surface_config = SurfaceMaskConfig(
            x_size_m=x_size_m,
            y_size_m=y_size_m,
            z_max_m=z_max_m,
            min_z_m=min_z_m,
            nx=int(surface_nx),
            ny=int(surface_ny),
            nz=int(surface_nz),
            grid_mode_xy=grid_mode_xy,
            grid_mode_z=grid_mode_z,
            downsample_mask_for_computation=downsample_mask_for_computation,
            max_computational_mask_size=int(max_computational_mask_size),
            max_active_pixels_for_direct_sum=int(max_active_pixels),
        )
        with geometry_tab:
            surface_options = render_function_geometry_controls(
                st,
                geometry_tab,
                surface_config,
            )

    else:
        image_tab, mask_tab, grid_tab, voltage_tab = container.tabs(
            ["Изображение", "Маска", "Сетка", "Питание электродов"]
        )
        with image_tab:
            uploaded_surface_mask = image_tab.file_uploader(
                "Загрузить PNG/JPG",
                type=["png", "jpg", "jpeg"],
                key="dash_surface_upload",
            )
            image_tab.info(
                "Светлые связные области считаются электродами; тёмный фон = 0 В."
            )
            mask_threshold = image_tab.slider(
                "Порог бинаризации",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                key="dash_mask_threshold",
            )
        with grid_tab:
            size_cols = grid_tab.columns(2)
            x_size_m = size_cols[0].number_input("Размер x [м]", value=1.0e-3, format="%.6e", key="dash_x_size")
            y_size_m = size_cols[1].number_input("Размер y [м]", value=1.0e-3, format="%.6e", key="dash_y_size")
            z_cols = grid_tab.columns(2)
            min_z_m = z_cols[0].number_input("min z [м]", value=2.0e-5, format="%.6e", key="dash_min_z")
            z_max_m = z_cols[1].number_input("z max [м]", value=1.0e-3, format="%.6e", key="dash_z_max")
            n_cols = grid_tab.columns(3)
            surface_nx = n_cols[0].number_input("nx", value=25, min_value=3, step=1, key="dash_surface_nx")
            surface_ny = n_cols[1].number_input("ny", value=25, min_value=3, step=1, key="dash_surface_ny")
            surface_nz = n_cols[2].number_input("nz", value=15, min_value=3, step=1, key="dash_surface_nz")
            mode_cols = grid_tab.columns(2)
            grid_mode_xy = mode_cols[0].selectbox(
                "Сетка x-y",
                ["uniform", "center_clustered_tanh", "edge_aware"],
                index=2,
                key="dash_grid_mode_xy",
            )
            grid_mode_z = mode_cols[1].selectbox(
                "Сетка z",
                ["uniform", "near_surface_clustered"],
                index=1,
                key="dash_grid_mode_z",
            )
            downsample_mask_for_computation = grid_tab.checkbox(
                "Уменьшать маску для расчёта",
                value=True,
                key="dash_downsample_mask",
            )
            max_computational_mask_size = grid_tab.slider(
                "Максимальный размер маски для расчёта",
                min_value=32,
                max_value=512,
                value=192,
                step=16,
                key="dash_max_comp_mask",
            )
            cluster_strength_xy = 2.5
            cluster_strength_z = 2.5
            edge_refinement_radius_m = 4.0e-5
            edge_refinement_points_per_edge = 5
            max_edge_grid_points = 120
            min_grid_spacing_m = 2.0e-6
            max_active_pixels = DEFAULT_SURFACE_MAX_ACTIVE_PIXELS
            if expert_mode:
                grid_extra = grid_tab.expander("Экспертные настройки сетки")
                with grid_extra:
                    cluster_strength_xy = grid_extra.number_input("Сгущение x-y", value=2.5, min_value=0.1, key="dash_cluster_xy")
                    cluster_strength_z = grid_extra.number_input("Сгущение z", value=2.5, min_value=0.1, key="dash_cluster_z")
                    edge_refinement_radius_m = grid_extra.number_input("Радиус edge-уточнения [м]", value=4.0e-5, format="%.6e", key="dash_edge_radius")
                    edge_refinement_points_per_edge = grid_extra.number_input("Точек на edge-пиксель", value=5, min_value=1, step=1, key="dash_edge_points")
                    max_edge_grid_points = grid_extra.number_input("Максимум edge-пикселей", value=120, min_value=1, step=1, key="dash_max_edge")
                    min_grid_spacing_m = grid_extra.number_input("Минимальный шаг сетки [м]", value=2.0e-6, format="%.6e", key="dash_min_spacing")
                    max_active_pixels = grid_extra.number_input("max_active_pixels_for_direct_sum", value=DEFAULT_SURFACE_MAX_ACTIVE_PIXELS, min_value=1, step=100, key="dash_max_active")

        surface_config = SurfaceMaskConfig(
            x_size_m=x_size_m,
            y_size_m=y_size_m,
            z_max_m=z_max_m,
            min_z_m=min_z_m,
            nx=int(surface_nx),
            ny=int(surface_ny),
            nz=int(surface_nz),
            mask_threshold=mask_threshold,
            grid_mode_xy=grid_mode_xy,
            grid_mode_z=grid_mode_z,
            cluster_strength_xy=cluster_strength_xy,
            cluster_strength_z=cluster_strength_z,
            edge_refinement_radius_m=edge_refinement_radius_m,
            edge_refinement_points_per_edge=int(edge_refinement_points_per_edge),
            max_edge_grid_points=int(max_edge_grid_points),
            min_grid_spacing_m=min_grid_spacing_m,
            downsample_mask_for_computation=downsample_mask_for_computation,
            max_computational_mask_size=int(max_computational_mask_size),
            max_active_pixels_for_direct_sum=int(max_active_pixels),
        )
        surface_options = build_surface_options_from_sidebar_manual_roles(
            st,
            voltage_tab,
            uploaded_surface_mask,
            surface_config,
            mask_threshold,
        )
        with mask_tab:
            show_surface_mask_section(mask_tab, surface_options)

    return field_source, source_label, uploaded_field_file, surface_options, mesh_cells


def build_dashboard_inputs(st, workflow_mode: str, expert_mode: bool) -> dict:
    """Render main-page card inputs and return all selected run options."""

    defaults = DEFAULT_CONFIG
    effective_mode = WORKFLOW_CHECK if workflow_mode == WORKFLOW_EXPERT else workflow_mode

    source_card = render_card(
        st,
        "1. Геометрия и поле",
        "Выберите рисунок электродов, импорт .npz или legacy FEM.",
    )
    field_source, source_label, uploaded_field_file, surface_options, mesh_cells = (
        render_field_source_card(st, source_card, expert_mode)
    )

    top_cols = st.columns(2)
    voltage_card = render_card(top_cols[0], "2. Питание ловушки")
    voltage_values = render_voltage_card(voltage_card, defaults)

    particle_card = render_card(top_cols[1], "3. Частица")
    particle_preset_name, particle_values = render_particle_card(particle_card, defaults)

    lower_cols = st.columns(2)
    environment_card = render_card(lower_cols[0], "4. Среда")
    environment_report = render_environment_card(
        environment_card,
        particle_radius_m=particle_values["particle_radius"],
        particle_mass_kg=particle_values["particle_mass"],
    )

    initial_card = render_card(
        lower_cols[1],
        "5. Начальные условия и время расчёта",
        "Раньше это называлось параметрами динамики.",
    )
    initial_values = render_initial_conditions_card(initial_card, defaults, expert_mode)

    sweep_defaults = {
        "sweep_voltages": "5,10,20",
        "sweep_frequencies": "5000,30000,100000",
    }
    if effective_mode == WORKFLOW_SEARCH:
        scan_card = render_card(st, "Диапазоны RF")
        scan_cols = scan_card.columns(3)
        voltage_text = scan_cols[0].text_input(
            "RF voltage values [В]",
            value=sweep_defaults["sweep_voltages"],
            key="dash_sweep_voltages",
        )
        frequency_text = scan_cols[1].text_input(
            "RF frequency values [Гц]",
            value=sweep_defaults["sweep_frequencies"],
            key="dash_sweep_frequencies",
        )
        max_simulations = scan_cols[2].number_input(
            "Лимит расчётов",
            value=25,
            min_value=1,
            step=1,
            key="dash_max_simulations",
        )
    else:
        voltage_text = sweep_defaults["sweep_voltages"]
        frequency_text = sweep_defaults["sweep_frequencies"]
        max_simulations = 25

    config = with_config_overrides(
        defaults,
        particle_mass=particle_values["particle_mass"],
        particle_charge=particle_values["particle_charge"],
        particle_radius=particle_values["particle_radius"],
        particle_density=particle_values["particle_density"],
        damping_coefficient=environment_report["gamma_kg_s"],
        initial_position=initial_values["initial_position"],
        initial_velocity=initial_values["initial_velocity"],
        voltage_amplitude=voltage_values["voltage_amplitude"],
        use_time_dependent_voltage=voltage_values["use_time_dependent_voltage"],
        dc_voltage=voltage_values["dc_voltage"],
        rf_voltage=voltage_values["rf_voltage"],
        rf_angular_frequency=2.0 * np.pi * voltage_values["rf_frequency_hz"],
        simulation_time=(0.0, initial_values["simulation_duration"]),
        time_step=initial_values["time_step"],
        max_time_step=initial_values["time_step"],
        ode_rtol=initial_values["ode_rtol"],
        ode_atol=initial_values["ode_atol"],
        confinement_radius_threshold=initial_values["confinement_radius_threshold"],
        mesh_cells=mesh_cells,
    )
    if voltage_values["use_time_dependent_voltage"] and voltage_values["rf_frequency_hz"] > 0.0:
        config = make_rf_case_config(
            config,
            rf_voltage=voltage_values["rf_voltage"],
            rf_angular_frequency=2.0 * np.pi * voltage_values["rf_frequency_hz"],
        )

    plot_quality = "Better quality" if expert_mode else "Fast preview"
    return {
        "config": config,
        "workflow_mode": effective_mode,
        "particle_preset_name": particle_preset_name,
        "plot_quality": plot_quality,
        "field_source": field_source,
        "source_label": source_label,
        "uploaded_field_file": uploaded_field_file,
        "surface_options": surface_options,
        "environment_report": environment_report,
        "voltage_text": voltage_text,
        "frequency_text": frequency_text,
        "max_simulations": int(max_simulations),
    }


def _format_optional_float(value: float) -> str:
    """Format finite/infinite diagnostic values compactly."""

    if np.isinf(value):
        return "∞"
    if np.isnan(value):
        return "—"
    return f"{value:.3e}"


def config_with_field_grid_domain(config, field_grid):
    """Use imported grid extents as the rectangular simulation domain."""

    domain_size = (
        float(field_grid.x_grid[-1] - field_grid.x_grid[0]),
        float(field_grid.y_grid[-1] - field_grid.y_grid[0]),
        float(field_grid.z_grid[-1] - field_grid.z_grid[0]),
    )
    return with_config_overrides(config, domain_size=domain_size)


def grid_position_from_index(field_grid, index: tuple[int, int, int]) -> tuple[float, float, float]:
    """Return the physical grid position for an ``(ix, iy, iz)`` index."""

    ix, iy, iz = index
    return (
        float(field_grid.x_grid[ix]),
        float(field_grid.y_grid[iy]),
        float(field_grid.z_grid[iz]),
    )


def compute_field_grid_diagnostics(field_grid, config) -> dict:
    """Compute simple diagnostics for a structured electric-field grid."""

    field = np.asarray(field_grid.electric_field_grid, dtype=float)
    field_magnitude = np.linalg.norm(field, axis=-1)
    finite_magnitude = np.where(np.isfinite(field_magnitude), field_magnitude, np.inf)
    min_index = tuple(int(i) for i in np.unravel_index(np.argmin(finite_magnitude), finite_magnitude.shape))
    min_position = grid_position_from_index(field_grid, min_index)

    target_z = float(config.initial_position[2])
    center_index = (
        int(np.argmin(np.abs(field_grid.x_grid - 0.0))),
        int(np.argmin(np.abs(field_grid.y_grid - 0.0))),
        int(np.argmin(np.abs(field_grid.z_grid - target_z))),
    )
    center_position = grid_position_from_index(field_grid, center_index)

    finite_values = field_magnitude[np.isfinite(field_magnitude)]
    max_field = float(np.max(finite_values)) if finite_values.size else np.nan

    diagnostics = {
        "source": getattr(field_grid, "source_description", None) or "FieldGrid",
        "grid_shape": tuple(int(v) for v in field_magnitude.shape),
        "min_field_index": min_index,
        "min_field_position": min_position,
        "min_field_magnitude": float(field_magnitude[min_index]),
        "center_index": center_index,
        "center_position": center_position,
        "center_field_magnitude": float(field_magnitude[center_index]),
        "max_field_magnitude": max_field,
        "pseudopotential": None,
    }

    if (
        config.use_time_dependent_voltage
        and abs(config.rf_voltage) > 0.0
        and abs(config.rf_angular_frequency) > 0.0
    ):
        rf_field_grid = getattr(field_grid, "rf_field_grid", field_grid)
        pseudopotential = compute_pseudopotential_from_rf_field_grid(
            rf_field_grid,
            particle_charge=config.particle_charge,
            particle_mass=config.particle_mass,
            rf_voltage=config.rf_voltage,
            rf_angular_frequency=config.rf_angular_frequency,
        )
        finite_pseudopotential = np.where(
            np.isfinite(pseudopotential),
            pseudopotential,
            np.inf,
        )
        pseudo_min_index = tuple(
            int(i)
            for i in np.unravel_index(
                np.argmin(finite_pseudopotential),
                finite_pseudopotential.shape,
            )
        )
        finite_pseudo_values = pseudopotential[np.isfinite(pseudopotential)]
        diagnostics["pseudopotential"] = {
            "min_index": pseudo_min_index,
            "min_position": grid_position_from_index(rf_field_grid, pseudo_min_index),
            "min_value_J": float(pseudopotential[pseudo_min_index]),
            "max_value_J": (
                float(np.max(finite_pseudo_values))
                if finite_pseudo_values.size
                else np.nan
            ),
        }

    return diagnostics


def show_field_grid_diagnostics(st, field_grid, config):
    """Display computed field, RF-null, and pseudopotential diagnostics."""

    diagnostics = compute_field_grid_diagnostics(field_grid, config)
    st.subheader("Диагностика поля")
    st.success(
        f"Сетка поля {diagnostics['grid_shape']} рассчитана/загружена из "
        f"{diagnostics['source']}."
    )

    rows = [
        {
            "quantity": "min |E_base| / кандидат RF-null",
            "x [m]": diagnostics["min_field_position"][0],
            "y [m]": diagnostics["min_field_position"][1],
            "z [m]": diagnostics["min_field_position"][2],
            "value": diagnostics["min_field_magnitude"],
            "units": "В/м на базовое поле",
        },
        {
            "quantity": "точка около x=y=0 и z0",
            "x [m]": diagnostics["center_position"][0],
            "y [m]": diagnostics["center_position"][1],
            "z [m]": diagnostics["center_position"][2],
            "value": diagnostics["center_field_magnitude"],
            "units": "В/м на базовое поле",
        },
        {
            "quantity": "max |E_base| на сетке",
            "x [m]": np.nan,
            "y [m]": np.nan,
            "z [m]": np.nan,
            "value": diagnostics["max_field_magnitude"],
            "units": "В/м на базовое поле",
        },
    ]
    st.dataframe(rows, use_container_width=True)

    surface_metadata = getattr(field_grid, "surface_computation_metadata", None)
    if surface_metadata is not None:
        st.write("Вычислительное уменьшение surface-маски")
        st.dataframe(
            [
                {
                    "original shape": str(surface_metadata["original_shape"]),
                    "computational shape": str(surface_metadata["computational_shape"]),
                    "active pixels before": surface_metadata["active_pixels_before"],
                    "active pixels after": surface_metadata["active_pixels_after"],
                    "active voltage pixels after": surface_metadata.get(
                        "active_voltage_pixels_after",
                        surface_metadata["active_pixels_after"],
                    ),
                    "pixel area [m^2]": surface_metadata.get("pixel_area_m2", np.nan),
                }
            ],
            use_container_width=True,
        )

    if diagnostics["pseudopotential"] is None:
        st.info(
            "Псевдопотенциал показывается только в RF-режиме при ненулевых "
            "RF-напряжении и RF-частоте."
        )
        return

    pseudopotential = diagnostics["pseudopotential"]
    st.subheader("Диагностика псевдопотенциала")
    st.write(
        "Упрощённая оценка: U_pseudo = Q^2 |Vrf E_RF_base|^2 / (4 m Omega^2). "
        "Псевдопотенциал считается только по RF-полю; DC-электроды входят "
        "в полную динамику как статическое поле."
    )
    st.dataframe(
        [
            {
                "quantity": "кандидат минимума псевдопотенциала",
                "x [m]": pseudopotential["min_position"][0],
                "y [m]": pseudopotential["min_position"][1],
                "z [m]": pseudopotential["min_position"][2],
                "value [J]": pseudopotential["min_value_J"],
            },
            {
                "quantity": "максимум псевдопотенциала на сетке",
                "x [m]": np.nan,
                "y [m]": np.nan,
                "z [m]": np.nan,
                "value [J]": pseudopotential["max_value_J"],
            },
        ],
        use_container_width=True,
    )


def load_ui_field_source(
    st,
    config,
    field_source: str,
    uploaded_field_file,
    surface_options=None,
):
    """Load the selected field source for a single trajectory run."""

    if field_source == "Built-in placeholder FEM":
        with st.spinner("Считаем демо / legacy FEM поле, нормированное на 1 В..."):
            fem_result = solve_laplace(config)
            E_base = make_E_at_position(fem_result)
        return fem_result, E_base, config

    if field_source in {
        "Surface electrode mask, Poisson kernel",
        "Function-defined surface electrodes",
        "Canvas-drawn surface electrodes",
    }:
        if surface_options is None or surface_options["mask"] is None:
            st.warning("Define or upload a surface-electrode geometry before computing the field.")
            return None, None, config

        surface_config = surface_options["config"]
        mask = surface_options["mask"]
        component_labels = surface_options["component_labels"]
        voltage_map = surface_options.get("voltage_map")
        if voltage_map is None:
            voltage_map = build_voltage_map_from_components(
                component_labels,
                surface_options["electrode_potentials"],
            )
        if config.initial_position[2] < surface_config.min_z_m:
            st.warning(
                "Начальная z-координата частицы ниже минимума surface-сетки. "
                "Задайте z0 >= min_z_m, потому что поле не вычисляется при z = 0."
            )

        if surface_options.get("uses_rf_dc_separation"):
            rf_voltage_map_base = surface_options.get("rf_voltage_map_base")
            dc_voltage_map = surface_options.get("dc_voltage_map")
            if rf_voltage_map_base is None or dc_voltage_map is None:
                st.warning("Surface RF/DC maps are not ready.")
                return None, None, config
            has_rf_electrodes = bool(np.any(rf_voltage_map_base != 0.0))
            has_static_electrodes = bool(np.any(dc_voltage_map != 0.0))
            if (
                config.use_time_dependent_voltage
                and abs(config.rf_voltage) > 0.0
                and not has_rf_electrodes
            ):
                st.warning(
                    "RF-напряжение задано, но RF-электроды не выбраны."
                )
            if not has_rf_electrodes and has_static_electrodes:
                st.info(
                    "Псевдопотенциал недоступен без RF-электродов; "
                    "DC/GND электроды останутся статическим полем."
                )
            with st.spinner("Computing separated RF and DC surface fields..."):
                try:
                    rf_field_grid = compute_surface_field_grid(
                        rf_voltage_map_base,
                        surface_config,
                        mask=mask,
                    )
                    dc_field_grid = compute_surface_field_grid(
                        dc_voltage_map,
                        surface_config,
                        mask=mask,
                    )
                except ValueError as exc:
                    show_surface_computation_error(st, exc)
                    return None, None, config
                rf_amplitude = config.rf_voltage if config.use_time_dependent_voltage else 0.0
                field_grid = combine_field_grids_for_preview(
                    rf_field_grid,
                    dc_field_grid,
                    rf_amplitude=rf_amplitude,
                )
                field_grid.surface_computation_metadata = getattr(
                    rf_field_grid,
                    "surface_computation_metadata",
                    surface_options.get("downsample_metadata"),
                )
                st.session_state["surface_field_grid"] = field_grid
                st.session_state["surface_rf_field_grid"] = rf_field_grid
                st.session_state["surface_dc_field_grid"] = dc_field_grid
                st.session_state["surface_voltage_map"] = voltage_map
                st.session_state["surface_rf_voltage_map_base"] = rf_voltage_map_base
                st.session_state["surface_dc_voltage_map"] = dc_voltage_map
                st.session_state["surface_component_labels"] = component_labels
                st.session_state["surface_binary_mask"] = mask
                st.session_state["surface_downsample_metadata"] = getattr(
                    field_grid,
                    "surface_computation_metadata",
                    surface_options.get("downsample_metadata"),
                )
                imported_config = config_with_field_grid_domain(config, field_grid)
                E_rf_base = make_E_at_position_from_field_grid(rf_field_grid)
                E_dc_static = make_E_at_position_from_field_grid(dc_field_grid)
                E_base = TwoChannelElectricField(
                    E_rf_base=E_rf_base,
                    E_dc_static=E_dc_static,
                    rf_amplitude=rf_amplitude,
                    rf_angular_frequency=config.rf_angular_frequency,
                    rf_phase=0.0,
                )

            st.info(
                "Surface field uses separated RF/DC channels: "
                "E_total(t,r)=E_dc(r)+Vrf*cos(Omega*t) E_RF_base(r)."
            )
            return field_grid, E_base, imported_config

        with st.spinner("Считаем поле surface-электродов методом Poisson-kernel..."):
            try:
                field_grid = compute_surface_field_grid(
                    voltage_map,
                    surface_config,
                    mask=mask,
                )
            except ValueError as exc:
                show_surface_computation_error(st, exc)
                return None, None, config
            st.session_state["surface_field_grid"] = field_grid
            st.session_state["surface_voltage_map"] = voltage_map
            st.session_state["surface_component_labels"] = component_labels
            st.session_state["surface_binary_mask"] = mask
            st.session_state["surface_downsample_metadata"] = getattr(
                field_grid,
                "surface_computation_metadata",
                surface_options.get("downsample_metadata"),
            )
            imported_config = config_with_field_grid_domain(config, field_grid)
            E_base = make_E_at_position_from_field_grid(field_grid)

        st.info("Используется surface-поле из Dirichlet half-space Poisson-kernel.")
        return field_grid, E_base, imported_config

    if uploaded_field_file is None:
        st.warning("Загрузите .npz файл перед расчётом с импортированным полем.")
        return None, None, config

    with st.spinner("Загружаем импортированное поле..."):
        if field_source == "Uploaded electric field grid .npz":
            field_grid = load_field_grid_npz(uploaded_field_file)
        else:
            field_grid = load_potential_grid_npz(uploaded_field_file)

        imported_config = config_with_field_grid_domain(config, field_grid)
        E_base = make_E_at_position_from_field_grid(field_grid)

    st.info("Импортированное поле используется как базовое и масштабируется выбранным DC/RF.")
    return field_grid, E_base, imported_config


def load_uploaded_field_for_validation(field_source: str, uploaded_field_file):
    """Load an uploaded file just for validation display."""

    if uploaded_field_file is None:
        return None

    if field_source == "Uploaded electric field grid .npz":
        return load_field_grid_npz(uploaded_field_file)
    if field_source == "Uploaded potential grid .npz":
        return load_potential_grid_npz(uploaded_field_file)
    return None


def show_field_validation_section(st, field_source: str, uploaded_field_file):
    """Show a compact validation report for uploaded field data."""

    if field_source in {
        "Built-in placeholder FEM",
        "Surface electrode mask, Poisson kernel",
    }:
        return

    st.subheader("Проверка импортированного поля")
    if uploaded_field_file is None:
        st.info("Загрузите .npz файл поля или потенциала, чтобы увидеть validation report.")
        return

    try:
        field_grid = load_uploaded_field_for_validation(field_source, uploaded_field_file)
        report = validate_field_grid_report(field_grid)
        symmetry_report = estimate_symmetry_checks(field_grid)
    except Exception as exc:
        st.error(f"Не удалось проверить загруженный файл: {exc}")
        return

    if report["valid"]:
        st.success("Загруженная сетка поля прошла базовую проверку.")
    else:
        st.error("В загруженной сетке поля есть ошибки.")

    if report["warnings"] or symmetry_report["warnings"]:
        for warning in report["warnings"] + symmetry_report["warnings"]:
            st.warning(warning)

    with st.expander("Validation report"):
        st.json(report)
        st.json({"symmetry_checks": symmetry_report})


def show_surface_mask_section(st, surface_options):
    """Show mask, components, voltage map, and observation-grid preview."""

    if surface_options is None:
        return

    st.subheader("Рисунок электродов")
    st.warning(
        "Surface-режим использует упрощённую Dirichlet half-space Poisson-kernel "
        "модель. Диэлектрик, конечная толщина электродов и точная физика зазоров "
        "пока не учитываются."
    )
    st.info(
        "Для предпросмотра используется исходное изображение, но для расчёта "
        "маска может быть уменьшена. Физический размер ловушки не меняется; "
        "меняется только число пикселей в численном интеграле."
    )

    mask = surface_options["mask"]
    labels = surface_options["component_labels"]
    if mask is None or labels is None:
        st.info("Загрузите PNG/JPG маску, чтобы увидеть компоненты и карту напряжений.")
        return

    voltage_map = surface_options.get("voltage_map")
    if voltage_map is None:
        voltage_map = build_voltage_map_from_components(
            labels,
            surface_options["electrode_potentials"],
        )
    if surface_options.get("overlap_warning"):
        st.warning(surface_options["overlap_warning"])
    if surface_options.get("assignments"):
        st.info(
            "Компоненты задают только геометрию. RF/DC/GND/CUSTOM роли "
            "задаются вручную и используются для отдельных RF и DC карт."
        )
        st.dataframe(
            _assignment_rows(surface_options["assignments"]),
            use_container_width=True,
        )
    config = surface_options["config"]
    computational_mask = surface_options.get("computational_mask")
    computational_voltage_map = surface_options.get("computational_voltage_map")
    downsample_metadata = surface_options.get("downsample_metadata")
    if computational_mask is None or downsample_metadata is None:
        (
            computational_voltage_map,
            computational_mask,
            downsample_metadata,
        ) = prepare_surface_voltage_map_for_computation(
            voltage_map,
            config,
            mask=mask,
        )
    x_grid, y_grid, _z_grid = make_surface_observation_grid(
        config,
        mask=computational_mask,
    )

    preview_top = st.columns(2)
    preview_bottom = st.columns(2)
    preview_top[0].image(mask.astype(float), caption="Бинарная маска электродов")

    labels_fig, labels_ax = plt.subplots(figsize=(4, 4))
    labels_image = labels_ax.imshow(labels, origin="upper", cmap="tab20")
    labels_ax.set_title("Связные компоненты")
    labels_ax.set_axis_off()
    labels_fig.colorbar(labels_image, ax=labels_ax, fraction=0.046)
    preview_top[1].pyplot(labels_fig)
    plt.close(labels_fig)

    voltage_fig, voltage_ax = plt.subplots(figsize=(4, 4))
    voltage_image = voltage_ax.imshow(voltage_map, origin="upper", cmap="coolwarm")
    voltage_ax.set_title("Карта напряжений [В]")
    voltage_ax.set_axis_off()
    voltage_fig.colorbar(voltage_image, ax=voltage_ax, fraction=0.046)
    preview_bottom[0].pyplot(voltage_fig)
    plt.close(voltage_fig)

    rf_voltage_map_base = surface_options.get("rf_voltage_map_base")
    dc_voltage_map = surface_options.get("dc_voltage_map")
    if rf_voltage_map_base is not None and dc_voltage_map is not None:
        with st.expander("RF/DC voltage maps"):
            role_cols = st.columns(2)
            rf_fig, rf_ax = plt.subplots(figsize=(4, 4))
            rf_image = rf_ax.imshow(rf_voltage_map_base, origin="upper", cmap="viridis")
            rf_ax.set_title("RF base map [1 on RF electrodes]")
            rf_ax.set_axis_off()
            rf_fig.colorbar(rf_image, ax=rf_ax, fraction=0.046)
            role_cols[0].pyplot(rf_fig)
            plt.close(rf_fig)

            dc_fig, dc_ax = plt.subplots(figsize=(4, 4))
            dc_image = dc_ax.imshow(dc_voltage_map, origin="upper", cmap="coolwarm")
            dc_ax.set_title("DC/static map [V]")
            dc_ax.set_axis_off()
            dc_fig.colorbar(dc_image, ax=dc_ax, fraction=0.046)
            role_cols[1].pyplot(dc_fig)
            plt.close(dc_fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    stride = max(1, int(np.ceil(max(len(x_grid), len(y_grid)) / 45)))
    ax.scatter(
        xx[::stride, ::stride].ravel(),
        yy[::stride, ::stride].ravel(),
        s=4,
        alpha=0.6,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Предпросмотр сетки наблюдения")
    preview_bottom[1].pyplot(fig)
    plt.close(fig)

    st.write(
        f"Найдено электродов: {surface_options['number_of_components']}; "
        f"сетка поля: {len(x_grid)} x {len(y_grid)} x {config.nz}"
    )
    st.dataframe(
        [
            {
                "original mask shape": str(downsample_metadata["original_shape"]),
                "computational mask shape": str(
                    downsample_metadata["computational_shape"]
                ),
                "active pixels before": downsample_metadata["active_pixels_before"],
                "active pixels after": downsample_metadata["active_pixels_after"],
                "active fraction after": downsample_metadata["active_fraction_after"],
                "pixel area [m^2]": (
                    config.x_size_m
                    / downsample_metadata["computational_shape"][1]
                    * config.y_size_m
                    / downsample_metadata["computational_shape"][0]
                ),
            }
        ],
        use_container_width=True,
    )
    warn_about_surface_active_pixels(
        st,
        downsample_metadata,
        int(getattr(config, "max_active_pixels_for_direct_sum", DEFAULT_SURFACE_MAX_ACTIVE_PIXELS)),
    )
    smallest_computational_side = min(downsample_metadata["computational_shape"])
    if smallest_computational_side < 64:
        st.warning(
            "Маска для расчёта сильно уменьшена. Быстро, но грубо. Для более "
            "точного поля увеличьте размер computational mask."
        )

    with st.expander("Computational mask preview"):
        comp_cols = st.columns(2)
        comp_cols[0].image(
            computational_mask.astype(float),
            caption="Расчётная бинарная маска",
        )
        comp_fig, comp_ax = plt.subplots(figsize=(4, 4))
        comp_image = comp_ax.imshow(
            computational_voltage_map,
            origin="upper",
            cmap="coolwarm",
        )
        comp_ax.set_title("Расчётная карта напряжений [В]")
        comp_ax.set_axis_off()
        comp_fig.colorbar(comp_image, ax=comp_ax, fraction=0.046)
        comp_cols[1].pyplot(comp_fig)
        plt.close(comp_fig)


def plot_potential_slice_better(fem_result, z_value: float = 0.0):
    """Return a smoother contour plot of the base potential."""

    if getattr(fem_result, "potential_grid", None) is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(
            0.5,
            0.5,
            "No potential grid available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    k = int(np.argmin(np.abs(fem_result.z_grid - z_value)))
    x_mesh, y_mesh = np.meshgrid(fem_result.x_grid, fem_result.y_grid, indexing="ij")
    potential_slice = fem_result.potential_grid[:, :, k]

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(
        x_mesh,
        y_mesh,
        potential_slice,
        levels=40,
        cmap="coolwarm",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Base potential at z = {fem_result.z_grid[k]:.2e} m")
    fig.colorbar(contour, ax=ax, label="Base potential [V per 1 V drive]")
    return fig


def plot_electric_field_slice_better(fem_result, z_value: float = 0.0):
    """Return a clearer contour/quiver plot of the base electric field."""

    k = int(np.argmin(np.abs(fem_result.z_grid - z_value)))
    x_mesh, y_mesh = np.meshgrid(fem_result.x_grid, fem_result.y_grid, indexing="ij")
    field = fem_result.electric_field_grid[:, :, k, :]
    ex = field[:, :, 0]
    ey = field[:, :, 1]
    ez = field[:, :, 2]
    field_magnitude = np.sqrt(ex**2 + ey**2 + ez**2)

    stride = max(1, int(np.ceil(max(len(fem_result.x_grid), len(fem_result.y_grid)) / 14)))

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(
        x_mesh,
        y_mesh,
        field_magnitude,
        levels=40,
        cmap="viridis",
    )
    ax.quiver(
        x_mesh[::stride, ::stride],
        y_mesh[::stride, ::stride],
        ex[::stride, ::stride],
        ey[::stride, ::stride],
        color="white",
        pivot="mid",
        scale=None,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Base electric field at z = {fem_result.z_grid[k]:.2e} m")
    fig.colorbar(contour, ax=ax, label="Base electric field magnitude [V/m per V]")
    return fig


def make_standard_figures(
    fem_result,
    particle_result,
    plot_quality: str = "Fast preview",
) -> list[tuple[str, plt.Figure]]:
    """Create figures displayed and saved by the UI."""

    figures = []

    if plot_quality == "Better quality":
        fig1 = plot_potential_slice_better(fem_result)
    else:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        plot_potential_slice(fem_result, ax=ax1)
    figures.append(("potential_slice.png", fig1))

    if plot_quality == "Better quality":
        fig2 = plot_electric_field_slice_better(fem_result)
    else:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        plot_electric_field_slice(fem_result, ax=ax2)
    figures.append(("electric_field_slice.png", fig2))

    fig3 = plt.figure(figsize=(6, 5))
    ax3 = fig3.add_subplot(111, projection="3d")
    plot_trajectory_3d(particle_result, ax=ax3)
    figures.append(("trajectory_3d.png", fig3))

    fig4, ax4 = plt.subplots(figsize=(7, 4))
    plot_coordinates_vs_time(particle_result, ax=ax4)
    figures.append(("coordinates_vs_time.png", fig4))

    fig5, ax5 = plt.subplots(figsize=(7, 4))
    plot_radius_vs_time(particle_result, ax=ax5)
    figures.append(("radius_vs_time.png", fig5))

    fig6, ax6 = plt.subplots(figsize=(7, 4))
    plot_speed_vs_time(particle_result, ax=ax6)
    figures.append(("speed_vs_time.png", fig6))

    return figures


def save_figures(figures: list[tuple[str, plt.Figure]], output_dir: Path):
    """Save UI figures into one output directory."""

    for filename, figure in figures:
        figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")


def save_particle_result_csv(particle_result, path: Path):
    """Save trajectory samples to CSV."""

    data = np.column_stack(
        (
            particle_result.t,
            particle_result.positions,
            particle_result.velocities,
            particle_result.speed,
        )
    )
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="t,x,y,z,vx,vy,vz,speed",
        comments="",
    )


def save_metrics_dict_csv(metrics: dict, status: str, path: Path):
    """Save one metrics dictionary to CSV."""

    row = {"status": status, **metrics}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_field_result_to_npz(field_result, path: Path):
    """Export either a FemResult or FieldGrid-like result to .npz."""

    if hasattr(field_result, "mesh") and hasattr(field_result, "basis"):
        save_fem_field_to_npz(field_result, path)
    else:
        save_field_grid_to_npz(field_result, path)


def min_field_candidate_on_boundary(field_grid, diagnostics: dict) -> bool:
    """Return True if the min-|E| candidate lies on the grid boundary."""

    index = diagnostics["min_field_index"]
    shape = diagnostics["grid_shape"]
    return any(component == 0 or component == size - 1 for component, size in zip(index, shape))


def show_environment_report(st, environment_report: dict):
    """Display gas/friction diagnostics."""

    st.subheader("Среда и трение")
    st.info(DRAG_WARNING_TEXT)
    st.dataframe(
        [
            {
                "режим": environment_report.get("mode_label", environment_report["regime"]),
                "модель": environment_report["regime"],
                "средняя длина свободного пробега [м]": environment_report[
                    "mean_free_path_m"
                ],
                "число Кнудсена": environment_report["knudsen_number"],
                "gamma [кг/с]": environment_report["gamma_kg_s"],
                "время затухания m/gamma [с]": environment_report["damping_time_s"],
            }
        ],
        use_container_width=True,
    )


def show_localization_summary(st, particle_result, config, field_result):
    """Show a localization-style conclusion and supporting metrics."""

    metrics = compute_trajectory_metrics(particle_result)
    status_code = classify_localization_status(
        particle_result,
        config,
        config.confinement_radius_threshold,
    )
    status_label = localization_status_label(status_code)
    final_time = (
        particle_result.exit_time
        if particle_result.left_domain
        else float(particle_result.t[-1])
    )
    diagnostics = compute_field_grid_diagnostics(field_result, config)
    min_position = np.asarray(diagnostics["min_field_position"], dtype=float)
    final_position = np.asarray(particle_result.positions[-1], dtype=float)
    distance_to_min_field = float(np.linalg.norm(final_position - min_position))

    st.subheader("Диагностика локализации")
    st.warning(PHYSICS_WARNING_TEXT)
    cols = st.columns(6)
    cols[0].metric("статус", status_label)
    cols[1].metric("final time [с]", f"{final_time:.3e}")
    cols[2].metric("max r [м]", f"{metrics['max_radius']:.3e}")
    cols[3].metric("final r [м]", f"{metrics['final_radius']:.3e}")
    cols[4].metric("final speed [м/с]", f"{metrics['final_speed']:.3e}")
    cols[5].metric("до min |E| [м]", f"{distance_to_min_field:.3e}")

    if particle_result.left_domain:
        st.error(f"Частица вылетела при t = {particle_result.exit_time:.3e} с.")
    elif status_code == "localized_like":
        st.success("Траектория выглядит локализованной в рамках выбранного порога.")
    else:
        st.warning("Результат неясен: частица не вылетела, но радиус слишком велик.")

    if min_field_candidate_on_boundary(field_result, diagnostics):
        st.warning(
            "Минимум |E| найден на границе расчётной области. Это может означать, "
            "что область расчёта слишком мала или настоящего внутреннего RF-null "
            "не найдено."
        )

    return status_code, status_label, metrics


def run_parameter_search_with_field(
    E_base,
    base_config,
    rf_voltages: np.ndarray,
    rf_frequencies_hz: np.ndarray,
    max_simulations: int,
) -> tuple[list[dict], np.ndarray, dict[str, np.ndarray]]:
    """Run RF voltage-frequency search using one already-built field."""

    total = len(rf_voltages) * len(rf_frequencies_hz)
    if total > max_simulations:
        raise ValueError(
            f"Слишком много расчётов: {total}. Уменьшите сетку или увеличьте лимит."
        )

    status_map = np.zeros((len(rf_frequencies_hz), len(rf_voltages)), dtype=int)
    metric_maps = {
        "max_radius": np.full_like(status_map, np.nan, dtype=float),
        "final_radius": np.full_like(status_map, np.nan, dtype=float),
        "final_speed": np.full_like(status_map, np.nan, dtype=float),
    }
    rows = []

    for frequency_index, rf_frequency_hz in enumerate(rf_frequencies_hz):
        omega = 2.0 * np.pi * float(rf_frequency_hz)
        for voltage_index, rf_voltage in enumerate(rf_voltages):
            case_config = make_rf_case_config(
                with_config_overrides(base_config, use_time_dependent_voltage=True),
                rf_voltage=float(rf_voltage),
                rf_angular_frequency=omega,
            )
            E_case = E_base
            if isinstance(E_base, TwoChannelElectricField):
                E_case = TwoChannelElectricField(
                    E_rf_base=E_base.E_rf_base,
                    E_dc_static=E_base.E_dc_static,
                    rf_amplitude=float(rf_voltage),
                    rf_angular_frequency=omega,
                    rf_phase=E_base.rf_phase,
                )
            particle_result = simulate_particle(E_case, case_config)
            status_code = classify_localization_status(
                particle_result,
                case_config,
                case_config.confinement_radius_threshold,
            )
            metrics = compute_trajectory_metrics(particle_result)
            status_map[frequency_index, voltage_index] = {
                "escaped": 0,
                "unclear": 1,
                "localized_like": 2,
            }[status_code]
            for metric_name in metric_maps:
                metric_maps[metric_name][frequency_index, voltage_index] = metrics[metric_name]
            rows.append(
                {
                    "rf_voltage": float(rf_voltage),
                    "rf_frequency_hz": float(rf_frequency_hz),
                    "status": status_code,
                    "status_label": localization_status_label(status_code),
                    "final_time": (
                        particle_result.exit_time
                        if particle_result.left_domain
                        else float(particle_result.t[-1])
                    ),
                    **metrics,
                }
            )

    return rows, status_map, metric_maps


def plot_localization_map(status_map, rf_voltages, rf_frequencies_hz):
    """Plot localization-like / escaped / unclear RF search map."""

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(
        status_map,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=2,
        extent=[
            float(rf_voltages[0]),
            float(rf_voltages[-1]),
            float(rf_frequencies_hz[0]),
            float(rf_frequencies_hz[-1]),
        ],
        cmap="RdYlGn",
    )
    ax.set_xlabel("RF-амплитуда [В]")
    ax.set_ylabel("RF-частота [Гц]")
    ax.set_title("Карта локализации")
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(["вылет", "неясно", "локализация"])
    return fig


def plot_metric_heatmap(metric_map, rf_voltages, rf_frequencies_hz, title, colorbar_label):
    """Plot one numeric RF-search metric heatmap."""

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(
        metric_map,
        origin="lower",
        aspect="auto",
        extent=[
            float(rf_voltages[0]),
            float(rf_voltages[-1]),
            float(rf_frequencies_hz[0]),
            float(rf_frequencies_hz[-1]),
        ],
        cmap="viridis",
    )
    ax.set_xlabel("RF-амплитуда [В]")
    ax.set_ylabel("RF-частота [Гц]")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    return fig


def plot_pseudopotential_slice(field_grid, config, z_value: float | None = None):
    """Plot a simple RF pseudopotential slice if RF parameters are nonzero."""

    if (
        not config.use_time_dependent_voltage
        or abs(config.rf_voltage) <= 0.0
        or abs(config.rf_angular_frequency) <= 0.0
    ):
        return None

    rf_field_grid = getattr(field_grid, "rf_field_grid", field_grid)
    pseudopotential = compute_pseudopotential_from_rf_field_grid(
        rf_field_grid,
        particle_charge=config.particle_charge,
        particle_mass=config.particle_mass,
        rf_voltage=config.rf_voltage,
        rf_angular_frequency=config.rf_angular_frequency,
    )
    if z_value is None:
        z_value = float(config.initial_position[2])
    k = int(np.argmin(np.abs(rf_field_grid.z_grid - z_value)))
    x_mesh, y_mesh = np.meshgrid(rf_field_grid.x_grid, rf_field_grid.y_grid, indexing="ij")

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(
        x_mesh,
        y_mesh,
        pseudopotential[:, :, k],
        levels=40,
        cmap="magma",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [м]")
    ax.set_ylabel("y [м]")
    ax.set_title(f"Псевдопотенциал RF при z = {rf_field_grid.z_grid[k]:.2e} м")
    fig.colorbar(contour, ax=ax, label="U_pseudo [Дж]")
    return fig


def apply_experiment_to_app_session(st, experiment_data: dict, run_after_load: bool = False):
    """Apply a preset or loaded experiment JSON to Streamlit session state."""

    experiment = experiment_config_from_dict(experiment_data)
    report = apply_experiment_config_to_session_state(
        experiment,
        st.session_state,
    )
    geometry_config = experiment.to_dict().get("geometry_config", {})
    if "function_definitions" in geometry_config:
        st.session_state["function_electrode_defs"] = geometry_config[
            "function_definitions"
        ]
    canvas_design = geometry_config.get("canvas_design")
    if canvas_design is not None:
        try:
            loaded_canvas = canvas_design_from_dict(canvas_design)
            st.session_state["canvas_confirmed_mask"] = loaded_canvas["binary_mask"]
            st.session_state["canvas_assignments"] = {
                str(assignment.region_id): {
                    "name": assignment.name,
                    "role": assignment.role,
                    "voltage": assignment.voltage,
                    "rf_phase": assignment.rf_phase,
                }
                for assignment in loaded_canvas["assignments"]
            }
            st.session_state["canvas_loaded_x_size_m"] = loaded_canvas["x_size_m"]
            st.session_state["canvas_loaded_y_size_m"] = loaded_canvas["y_size_m"]
            st.session_state["canvas_loaded_resolution_px"] = loaded_canvas[
                "canvas_resolution_px"
            ]
        except Exception as exc:
            report.setdefault("warnings", []).append(
                f"Canvas design could not be applied: {exc}"
            )
    if run_after_load:
        st.session_state["run_quick_demo_after_load"] = True
        st.session_state["quick_demo_resolution_warning"] = True
    st.session_state["last_loaded_experiment_title"] = experiment.title
    return report


def queue_experiment_for_next_run(st, experiment_data: dict, run_after_load: bool = False):
    """Apply an experiment on the next run, before widget keys are created."""

    st.session_state["pending_experiment_data"] = experiment_data
    st.session_state["pending_experiment_run_after_load"] = bool(run_after_load)


def render_landing_dashboard(st):
    """Show a clean first-open dashboard with workflow cards."""

    def choose_workflow(mode: str):
        set_pending_workflow_mode(st, mode)
        st.rerun()

    if (
        "last_field_result" in st.session_state
        or "last_particle_result" in st.session_state
        or "last_parameter_search_results" in st.session_state
    ):
        return

    st.subheader("Выберите сценарий расчёта")
    cards = st.columns(3)
    scenarios = [
        (
            "Проверить локализацию",
            "Один набор параметров: поле, частица, среда, RF. На выходе — "
            "траектория и диагностика.",
        ),
        (
            "Подобрать RF-параметры",
            "Скан по напряжению и частоте. На выходе — карта областей, где "
            "частица не вылетает.",
        ),
        (
            "Посмотреть поле / экспорт",
            "Рассчитать или загрузить поле, посмотреть срезы и сохранить .npz.",
        ),
    ]
    scenario_modes = [WORKFLOW_CHECK, WORKFLOW_SEARCH, WORKFLOW_FIELD_ONLY]
    for column, mode, (title, text) in zip(cards, scenario_modes, scenarios):
        with column.container(border=True):
            st.markdown(f"#### {title}")
            st.markdown(f'<div class="landing-card">{text}</div>', unsafe_allow_html=True)
            st.button(
                "Открыть сценарий",
                key=f"landing_open_{dashboard_mode_to_action(mode)}",
                on_click=choose_workflow,
                args=(mode,),
            )
    action_cols = st.columns(2)
    if action_cols[0].button("Запустить быстрый демо-расчёт", type="primary"):
        preset = get_experiment_preset("Быстрый старт: 4 RF электрода")
        queue_experiment_for_next_run(st, preset, run_after_load=True)
        st.rerun()
    if action_cols[1].button("Открыть пресеты"):
        st.session_state["show_experiment_tools_hint"] = True
        st.rerun()
    st.caption(
        "Быстрый старт: выберите сценарий слева, загрузите рисунок электродов "
        "или .npz, задайте RF-напряжение/частоту и нажмите кнопку запуска."
    )


def render_overview_tab(st, config, environment_report):
    """Render overview result metrics without an empty wall of plots."""

    st.subheader("Обзор")
    if "last_particle_result" not in st.session_state:
        st.info("После запуска расчёта здесь появится статус локализации и ключевые метрики.")
        if "last_field_result" in st.session_state:
            st.success("Поле уже рассчитано или загружено.")
            show_field_grid_diagnostics(st, st.session_state["last_field_result"], config)
        return

    particle_result = st.session_state["last_particle_result"]
    metrics = st.session_state.get("last_trajectory_metrics", compute_trajectory_metrics(particle_result))
    status = st.session_state.get("last_trajectory_status", "unclear")
    render_status_badge(st, status)
    cols = st.columns(5)
    cols[0].metric("max radius [м]", f"{metrics['max_radius']:.3e}")
    cols[1].metric("final radius [м]", f"{metrics['final_radius']:.3e}")
    cols[2].metric("final speed [м/с]", f"{metrics['final_speed']:.3e}")
    cols[3].metric("gamma [кг/с]", f"{environment_report['gamma_kg_s']:.3e}")
    cols[4].metric("m/gamma [с]", _format_optional_float(environment_report["damping_time_s"]))

    if "last_field_result" in st.session_state:
        diagnostics = compute_field_grid_diagnostics(st.session_state["last_field_result"], config)
        st.dataframe(
            [
                {
                    "min |E| x [м]": diagnostics["min_field_position"][0],
                    "min |E| y [м]": diagnostics["min_field_position"][1],
                    "min |E| z [м]": diagnostics["min_field_position"][2],
                    "min |E| [В/м]": diagnostics["min_field_magnitude"],
                }
            ],
            use_container_width=True,
        )


def render_launch_card(
    st,
    workflow_mode: str,
    field_source: str,
    surface_options: dict | None,
    voltage_text: str,
    frequency_text: str,
):
    """Render the compact launch card and return button states."""

    card = render_card(st, "6. Запуск")
    active_pixels = "—"
    field_grid_size = "—"
    if field_source in {
        "Surface electrode mask, Poisson kernel",
        "Function-defined surface electrodes",
        "Canvas-drawn surface electrodes",
    } and surface_options:
        metadata = surface_options.get("downsample_metadata")
        config = surface_options.get("config")
        if metadata is not None:
            active_pixels = str(metadata.get("active_voltage_pixels_after", metadata["active_pixels_after"]))
        if config is not None:
            field_grid_size = f"{config.nx} × {config.ny} × {config.nz}"

    simulations = 1
    if workflow_mode == WORKFLOW_SEARCH:
        try:
            simulations = len(parse_comma_separated_floats(voltage_text)) * len(
                parse_comma_separated_floats(frequency_text)
            )
        except Exception:
            simulations = 0

    metric_cols = card.columns(3)
    metric_cols[0].metric("Сетка поля", field_grid_size)
    metric_cols[1].metric("Активных пикселей", active_pixels)
    metric_cols[2].metric("Расчётов", simulations)

    if workflow_mode == WORKFLOW_CHECK:
        return card.button("Запустить расчёт", type="primary"), False, False
    if workflow_mode == WORKFLOW_SEARCH:
        return False, card.button("Запустить подбор RF-параметров", type="primary"), False
    if workflow_mode == WORKFLOW_FIELD_ONLY:
        return False, False, card.button("Построить поле / экспорт", type="primary")
    card.info("В экспертном режиме используйте вкладку «Экспертное» или выберите рабочий сценарий.")
    return False, False, False


def run_single_simulation(
    st,
    config,
    plot_quality: str,
    field_source: str,
    uploaded_field_file,
    surface_options=None,
):
    """Run one FEM solve and particle trajectory from the UI."""

    output_dir = create_ui_output_dir(config.output_dir)

    try:
        field_result, E_base, config = load_ui_field_source(
            st,
            config,
            field_source,
            uploaded_field_file,
            surface_options,
        )
    except Exception as exc:
        st.error("Расчёт поля не удался.")
        st.exception(exc)
        return

    if field_result is None:
        return

    st.session_state["last_field_result"] = field_result
    show_field_grid_diagnostics(st, field_result, config)

    try:
        with st.spinner("Интегрируем траекторию частицы..."):
            particle_result = simulate_particle(E_base, config)
    except Exception as exc:
        st.error("Расчёт динамики частицы не удался.")
        st.exception(exc)
        return

    st.session_state["last_particle_result"] = particle_result

    status_code, status_label, metrics = show_localization_summary(
        st,
        particle_result,
        config,
        field_result,
    )
    st.session_state["last_trajectory_metrics"] = metrics
    st.session_state["last_trajectory_status"] = status_code

    figures = make_standard_figures(field_result, particle_result, plot_quality)
    save_figures(figures, output_dir)
    save_particle_result_csv(particle_result, output_dir / "trajectory.csv")
    save_metrics_dict_csv(metrics, status_label, output_dir / "metrics.csv")
    save_field_result_to_npz(field_result, output_dir / "field_grid.npz")
    st.session_state["last_output_dir"] = str(output_dir)
    st.session_state["last_exported_files"] = {
        "trajectory.csv": str(output_dir / "trajectory.csv"),
        "metrics.csv": str(output_dir / "metrics.csv"),
        "field_grid.npz": str(output_dir / "field_grid.npz"),
    }

    st.subheader("Графики")
    for _filename, figure in figures:
        st.pyplot(figure)
        plt.close(figure)

    st.success(f"Результаты сохранены в: {output_dir}")


def run_quick_sweep(st, config, voltage_text: str, frequency_text: str):
    """Run a small RF sweep from comma-separated UI inputs."""

    rf_voltages = parse_comma_separated_floats(voltage_text)
    rf_frequencies_hz = parse_comma_separated_floats(frequency_text)
    rf_angular_frequencies = 2.0 * np.pi * rf_frequencies_hz
    output_dir = create_ui_output_dir(config.output_dir)

    sweep_config = with_config_overrides(config, use_time_dependent_voltage=True)

    with st.spinner("Running quick RF sweep..."):
        results, survival_map, confinement_map = run_parameter_sweep(
            rf_voltages=rf_voltages,
            rf_angular_frequencies=rf_angular_frequencies,
            base_config=sweep_config,
            simulation_time=sweep_config.simulation_time,
            confinement_radius_threshold=sweep_config.confinement_radius_threshold,
        )

    st.subheader("Sweep results")
    st.dataframe(results, use_container_width=True)
    csv_path = output_dir / "ui_sweep_results.csv"
    save_results_csv(results, csv_path)

    survival_fig = plot_binary_map(
        survival_map,
        rf_voltages,
        rf_angular_frequencies,
        title="UI RF sweep survival map",
        zero_label="escaped",
        one_label="survived",
    )
    confinement_fig = plot_binary_map(
        confinement_map,
        rf_voltages,
        rf_angular_frequencies,
        title="UI RF sweep confinement map",
        zero_label="not confined",
        one_label="confined",
    )
    survival_path = output_dir / "ui_stability_map.png"
    confinement_path = output_dir / "ui_confinement_map.png"
    survival_fig.savefig(survival_path, dpi=180, bbox_inches="tight")
    confinement_fig.savefig(confinement_path, dpi=180, bbox_inches="tight")
    st.session_state["last_output_dir"] = str(output_dir)
    st.session_state["last_exported_files"] = {
        "ui_sweep_results.csv": str(csv_path),
        "ui_stability_map.png": str(survival_path),
        "ui_confinement_map.png": str(confinement_path),
    }

    st.subheader("Survival map")
    st.pyplot(survival_fig)
    plt.close(survival_fig)

    st.subheader("Confinement map")
    st.pyplot(confinement_fig)
    plt.close(confinement_fig)

    st.success(f"Saved UI outputs to: {output_dir}")


def run_field_only_workflow(
    st,
    config,
    plot_quality: str,
    field_source: str,
    uploaded_field_file,
    surface_options=None,
):
    """Build/load field, show diagnostics and slices, and export .npz."""

    output_dir = create_ui_output_dir(config.output_dir)
    try:
        field_result, _E_base, config = load_ui_field_source(
            st,
            config,
            field_source,
            uploaded_field_file,
            surface_options,
        )
    except Exception as exc:
        st.error("Расчёт или загрузка поля не удались.")
        st.exception(exc)
        return

    if field_result is None:
        return

    st.session_state["last_field_result"] = field_result
    show_field_grid_diagnostics(st, field_result, config)

    st.subheader("Срезы поля")
    potential_fig = (
        plot_potential_slice_better(field_result)
        if plot_quality == "Better quality"
        else None
    )
    if potential_fig is None:
        potential_fig, ax = plt.subplots(figsize=(6, 5))
        plot_potential_slice(field_result, ax=ax)
    st.pyplot(potential_fig)
    plt.close(potential_fig)

    field_fig = (
        plot_electric_field_slice_better(field_result)
        if plot_quality == "Better quality"
        else None
    )
    if field_fig is None:
        field_fig, ax = plt.subplots(figsize=(6, 5))
        plot_electric_field_slice(field_result, ax=ax)
    st.pyplot(field_fig)
    plt.close(field_fig)

    export_path = output_dir / "current_field_grid.npz"
    save_field_result_to_npz(field_result, export_path)
    st.session_state["last_output_dir"] = str(output_dir)
    st.session_state["last_exported_files"] = {
        "field_grid.npz": str(export_path),
    }
    st.success(f"Поле экспортировано в: {export_path}")


def run_parameter_search_workflow(
    st,
    config,
    field_source: str,
    uploaded_field_file,
    surface_options,
    voltage_text: str,
    frequency_text: str,
    max_simulations: int,
):
    """Run RF parameter search without recomputing the selected field each time."""

    rf_voltages = parse_comma_separated_floats(voltage_text)
    rf_frequencies_hz = parse_comma_separated_floats(frequency_text)
    total = len(rf_voltages) * len(rf_frequencies_hz)
    st.write(f"Будет выполнено расчётов: {total}")
    if total > max_simulations:
        st.warning("Слишком большая сетка подбора: уменьшите число точек.")
        return

    output_dir = create_ui_output_dir(config.output_dir)
    try:
        field_result, E_base, config = load_ui_field_source(
            st,
            config,
            field_source,
            uploaded_field_file,
            surface_options,
        )
        if field_result is None:
            return
        with st.spinner("Запускаем подбор RF-параметров..."):
            rows, status_map, metric_maps = run_parameter_search_with_field(
                E_base,
                config,
                rf_voltages,
                rf_frequencies_hz,
                max_simulations=max_simulations,
            )
    except Exception as exc:
        st.error("Подбор параметров не удался.")
        st.exception(exc)
        return

    st.session_state["last_parameter_search_results"] = rows

    csv_path = output_dir / "parameter_search_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    st.session_state["last_output_dir"] = str(output_dir)
    st.session_state["last_exported_files"] = {
        "parameter_search_results.csv": str(csv_path),
    }

    st.subheader("Результаты подбора")
    st.dataframe(rows, use_container_width=True)

    localization_fig = plot_localization_map(status_map, rf_voltages, rf_frequencies_hz)
    localization_fig.savefig(output_dir / "localization_map.png", dpi=180, bbox_inches="tight")
    st.pyplot(localization_fig)
    plt.close(localization_fig)

    for metric_name, title, label in [
        ("max_radius", "max radius", "max r [м]"),
        ("final_radius", "final radius", "final r [м]"),
        ("final_speed", "final speed", "final speed [м/с]"),
    ]:
        fig = plot_metric_heatmap(
            metric_maps[metric_name],
            rf_voltages,
            rf_frequencies_hz,
            title,
            label,
        )
        fig.savefig(output_dir / f"{metric_name}_map.png", dpi=180, bbox_inches="tight")
        st.pyplot(fig)
        plt.close(fig)

    localized_rows = [row for row in rows if row["status"] == "localized_like"]
    localized_rows.sort(key=lambda row: row["max_radius"])
    st.subheader("Подходящие параметры")
    if localized_rows:
        st.dataframe(localized_rows[:5], use_container_width=True)
    else:
        st.info("На этой сетке не найдено точек, похожих на локализацию.")
    st.success(f"Результаты подбора сохранены в: {csv_path}")


def _collect_ui_experiment(
    st,
    config,
    workflow_mode: str,
    source_label: str,
    surface_options,
    environment_report: dict,
    voltage_text: str,
    frequency_text: str,
    max_simulations: int,
):
    """Collect the current dashboard state as an ExperimentConfig."""

    return collect_current_experiment_config_from_ui(
        config=config,
        workflow_mode=workflow_mode,
        geometry_source=source_label,
        surface_options=surface_options,
        environment_report=environment_report,
        parameter_search_config={
            "sweep_voltages": voltage_text,
            "sweep_frequencies": frequency_text,
            "max_simulations": int(max_simulations),
        },
        session_state=dict(st.session_state),
        title=st.session_state.get(
            "experiment_title",
            "Streamlit charged-particle trap experiment",
        ),
        description=st.session_state.get("experiment_description", ""),
        notes=st.session_state.get("experiment_notes", ""),
    )


def render_experiment_tools(
    st,
    config,
    workflow_mode: str,
    source_label: str,
    surface_options,
    environment_report: dict,
    voltage_text: str,
    frequency_text: str,
    max_simulations: int,
):
    """Render presets, experiment JSON save/load, reports, and ZIP export."""

    st.subheader("Пресеты и отчёт")
    if st.session_state.get("show_experiment_tools_hint"):
        st.info("Пресеты открыты здесь. Выберите карточку и загрузите настройки.")
        st.session_state["show_experiment_tools_hint"] = False

    preset_tab, config_tab, report_tab, zip_tab = st.tabs(
        ["Пресеты", "Сохранить/загрузить", "Отчёт", "ZIP export"]
    )

    with preset_tab:
        st.write("Встроенные пресеты используют лёгкие сетки и placeholder-параметры.")
        presets = built_in_experiment_presets()
        for title, preset in presets.items():
            with st.container(border=True):
                st.markdown(f"#### {title}")
                st.write(preset.get("description", ""))
                expected = preset.get("diagnostics_config", {}).get(
                    "expected_thing_to_look_at",
                    "",
                )
                if expected:
                    st.caption(expected)
                cols = st.columns(2)
                if cols[0].button("Загрузить пресет", key=f"load_preset_{title}"):
                    queue_experiment_for_next_run(st, preset)
                    st.rerun()
                if cols[1].button(
                    "Загрузить и запустить",
                    key=f"load_run_preset_{title}",
                ):
                    queue_experiment_for_next_run(st, preset, run_after_load=True)
                    st.rerun()

    initial_experiment = _collect_ui_experiment(
        st,
        config,
        workflow_mode,
        source_label,
        surface_options,
        environment_report,
        voltage_text,
        frequency_text,
        max_simulations,
    )
    experiment = initial_experiment
    experiment_json = json.dumps(
        initial_experiment.to_dict(),
        indent=2,
        ensure_ascii=False,
    )

    with config_tab:
        st.text_input(
            "Название эксперимента",
            value=initial_experiment.title,
            key="experiment_title",
        )
        st.text_area(
            "Описание",
            value=st.session_state.get("experiment_description", ""),
            key="experiment_description",
            height=80,
        )
        st.text_area(
            "Заметки",
            value=st.session_state.get("experiment_notes", ""),
            key="experiment_notes",
            height=80,
        )
        experiment = _collect_ui_experiment(
            st,
            config,
            workflow_mode,
            source_label,
            surface_options,
            environment_report,
            voltage_text,
            frequency_text,
            max_simulations,
        )
        experiment_json = json.dumps(
            experiment.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
        st.download_button(
            "Скачать experiment_config.json",
            data=experiment_json,
            file_name="experiment_config.json",
            mime="application/json",
            key="download_experiment_config",
        )
        uploaded_config = st.file_uploader(
            "Загрузить experiment_config.json",
            type=["json"],
            key="upload_experiment_config",
        )
        if uploaded_config is not None and st.button("Применить experiment_config.json"):
            try:
                data = json.loads(uploaded_config.getvalue().decode("utf-8"))
                validation = validate_experiment_config(data)
                if not validation["valid"]:
                    st.error("; ".join(validation["errors"]))
                else:
                    if validation["warnings"]:
                        st.warning("; ".join(validation["warnings"]))
                    queue_experiment_for_next_run(st, data)
                    st.rerun()
            except Exception as exc:
                st.error("Не удалось загрузить experiment_config.json.")
                st.exception(exc)

    metrics = st.session_state.get("last_trajectory_metrics", {})
    if metrics and "last_trajectory_status" in st.session_state:
        metrics = {"status": st.session_state["last_trajectory_status"], **metrics}
    field_diagnostics = {}
    if "last_field_result" in st.session_state:
        try:
            field_diagnostics = compute_field_grid_diagnostics(
                st.session_state["last_field_result"],
                config,
            )
        except Exception:
            field_diagnostics = {}
    exported_files = list(st.session_state.get("last_exported_files", {}).keys())
    report_md = build_markdown_report(
        experiment_config=experiment,
        metrics=metrics,
        field_diagnostics=field_diagnostics,
        environment_report=environment_report,
        exported_files=exported_files,
    )
    report_html = build_html_report(
        experiment_config=experiment,
        metrics=metrics,
        field_diagnostics=field_diagnostics,
        environment_report=environment_report,
        exported_files=exported_files,
    )

    with report_tab:
        st.download_button(
            "Скачать report.md",
            data=report_md,
            file_name="report.md",
            mime="text/markdown",
            key="download_report_md",
        )
        st.download_button(
            "Скачать report.html",
            data=report_html,
            file_name="report.html",
            mime="text/html",
            key="download_report_html",
        )
        with st.expander("Предпросмотр Markdown"):
            st.markdown(report_md)

    with zip_tab:
        include_field_grid = st.checkbox(
            "Включить FieldGrid .npz в ZIP",
            value=False,
            key="include_field_grid_in_zip",
        )
        if st.button("Скачать пакет эксперимента ZIP"):
            output_dir = create_ui_output_dir(config.output_dir)
            zip_path = output_dir / "experiment_package.zip"
            extra_files = st.session_state.get("last_exported_files", {}).copy()
            if (
                surface_options
                and surface_options.get("source_kind") == "canvas"
                and surface_options.get("canvas_binary_mask") is not None
            ):
                try:
                    canvas_config = surface_options["config"]
                    canvas_mask = surface_options["canvas_binary_mask"]
                    canvas_design = canvas_design_to_dict(
                        x_size_m=canvas_config.x_size_m,
                        y_size_m=canvas_config.y_size_m,
                        canvas_resolution_px=int(max(canvas_mask.shape)),
                        binary_mask=canvas_mask,
                        assignments=surface_options.get("assignments", []),
                        rf_amplitude=config.rf_voltage,
                        rf_frequency_hz=config.rf_angular_frequency / (2.0 * np.pi),
                        notes="Exported from Streamlit experiment package.",
                    )
                    extra_files["canvas_design.json"] = json.dumps(
                        canvas_design,
                        ensure_ascii=False,
                        indent=2,
                    ).encode("utf-8")
                except Exception as exc:
                    st.warning(f"Canvas design JSON was not added to ZIP: {exc}")
            export_experiment_zip(
                zip_path,
                experiment_config=experiment,
                report_md=report_md,
                report_html=report_html,
                extra_files=extra_files,
                include_field_grid=include_field_grid,
            )
            st.session_state["last_experiment_zip"] = str(zip_path)
            st.success(f"ZIP создан: {zip_path}")
        if "last_experiment_zip" in st.session_state:
            zip_path = Path(st.session_state["last_experiment_zip"])
            if zip_path.exists():
                st.download_button(
                    "Скачать готовый ZIP",
                    data=zip_path.read_bytes(),
                    file_name="experiment_package.zip",
                    mime="application/zip",
                    key="download_experiment_zip",
                )


def export_builtin_fem_field_from_ui(st, config):
    """Save the built-in normalized FEM field in the documented .npz format."""

    output_dir = create_ui_output_dir(config.output_dir)
    export_path = output_dir / "builtin_fem_field_grid.npz"
    with st.spinner("Solving built-in FEM field for export..."):
        fem_result = solve_laplace(config)
        save_fem_field_to_npz(fem_result, export_path)
    st.success(f"Saved reference field-grid file to: {export_path}")


def mathieu_voltage_components(config) -> tuple[float, float]:
    """Return the voltage components actually represented in the UI config."""

    if config.use_time_dependent_voltage:
        return config.dc_voltage, config.rf_voltage

    # In static mode, the particle dynamics ignores dc_voltage/rf_voltage and
    # uses voltage_amplitude.  For a Mathieu-style point, represent that as a
    # pure DC case with mathieu_q = 0.
    return config.voltage_amplitude, 0.0


def mathieu_parameters_outside_standard_view(mathieu_parameters: dict) -> bool:
    """Return True when the current a/q values are outside the usual first view."""

    for axis in ("x", "y"):
        mathieu_a_value = mathieu_parameters.get(f"mathieu_a_{axis}", np.nan)
        mathieu_q_value = mathieu_parameters.get(f"mathieu_q_{axis}", np.nan)
        if abs(mathieu_a_value) > 0.5 or abs(mathieu_q_value) > 1.2:
            return True
    return False


def run_mathieu_analysis(st, config):
    """Show the local FEM effective Mathieu point and stability diagram."""

    st.subheader("Mathieu stability")
    st.write(
        "This diagram uses the classic dimensionless a-q_mathieu plane for a "
        "quadrupole-like model. The RF sweep maps are direct voltage-frequency "
        "numerical experiments; this plot is a local dimensionless stability "
        "analysis."
    )
    st.caption("mathieu_q is the dimensionless Mathieu parameter, not particle_charge.")
    auto_zoom = st.checkbox(
        "Auto zoom to current Mathieu point",
        value=False,
        help=(
            "Leave unchecked to keep the standard first-stability-region view. "
            "Check it only when you want to inspect a far-away point."
        ),
    )

    with st.spinner("Solving FEM field and fitting local curvature..."):
        fem_result = solve_laplace(config)
        curvature = estimate_potential_curvature_near_center(fem_result)

    dc_voltage, rf_voltage = mathieu_voltage_components(config)
    try:
        mathieu_parameters = compute_effective_mathieu_parameters_from_curvature(
            curvature,
            particle_charge=config.particle_charge,
            particle_mass=config.particle_mass,
            dc_voltage=dc_voltage,
            rf_voltage=rf_voltage,
            omega=config.rf_angular_frequency,
        )
        point_status = "current"
    except Exception as exc:
        st.warning(f"Could not compute Mathieu parameters: {exc}")
        mathieu_parameters = {
            "mathieu_a_x": np.nan,
            "mathieu_q_x": np.nan,
            "mathieu_a_y": np.nan,
            "mathieu_q_y": np.nan,
        }
        point_status = "unavailable"

    if mathieu_parameters_outside_standard_view(mathieu_parameters):
        st.warning(
            "Current point is outside the standard first-stability-region view. "
            "This usually means the selected DC/RF/frequency parameters are not "
            "in the usual Mathieu stability range."
        )

    st.write(
        "The FEM values below are effective local estimates from a quadratic "
        "fit near the trap center, not exact global stability parameters."
    )
    st.dataframe(
        [
            {
                "kx [1/m^2 per V]": curvature["kx"],
                "ky [1/m^2 per V]": curvature["ky"],
                "kz [1/m^2 per V]": curvature["kz"],
                "fit_radius [m]": curvature["fit_radius"],
                "points_used": curvature["number_of_points_used"],
                "rms_residual [V]": curvature["rms_residual"],
            }
        ],
        use_container_width=True,
    )
    st.dataframe(
        [
            {
                **mathieu_parameters,
                "dc_voltage_used [V]": dc_voltage,
                "rf_voltage_used [V]": rf_voltage,
                "rf_frequency_used [Hz]": config.rf_angular_frequency / (2.0 * np.pi),
            }
        ],
        use_container_width=True,
    )

    point = {
        **mathieu_parameters,
        "status": point_status,
        "label": "current UI point",
    }
    fig, _ax = plot_mathieu_stability_diagram(points=point, auto_zoom=auto_zoom)
    st.pyplot(fig)
    plt.close(fig)


def main():
    """Run the Streamlit app."""

    import streamlit as st

    st.set_page_config(
        page_title="Surface electrode trap dashboard",
        layout="wide",
    )
    render_dashboard_style(st)
    st.title("Моделирование поверхностной электродной ловушки")
    st.markdown(
        '<div class="dashboard-kicker">Геометрия электродов → поле → движение '
        'частицы → диагностика локализации</div>',
        unsafe_allow_html=True,
    )
    st.warning(WARNING_TEXT)

    pending_experiment = st.session_state.pop("pending_experiment_data", None)
    if pending_experiment is not None:
        run_after_load = bool(
            st.session_state.pop("pending_experiment_run_after_load", False)
        )
        apply_experiment_to_app_session(
            st,
            pending_experiment,
            run_after_load=run_after_load,
        )

    st.sidebar.header("Что сделать?")
    initialize_workflow_state(st)
    workflow_mode = st.sidebar.radio(
        "Сценарий",
        DASHBOARD_WORKFLOW_MODES,
        key=WORKFLOW_WIDGET_KEY,
        label_visibility="collapsed",
    )
    workflow_mode = _normalize_workflow_mode(workflow_mode)
    st.session_state[APP_WORKFLOW_MODE_KEY] = workflow_mode
    expert_toggle = st.sidebar.checkbox("Показать экспертные настройки")
    expert_mode = expert_toggle or workflow_mode == WORKFLOW_EXPERT
    st.session_state["expert_mode_enabled"] = expert_mode
    st.sidebar.divider()
    st.sidebar.subheader("Статус проекта")
    st.sidebar.caption("Локальный учебно-исследовательский прототип.")
    st.sidebar.caption("Физика не валидирована для реальной ловушки.")

    render_landing_dashboard(st)
    st.info(WORKFLOW_DESCRIPTIONS[workflow_mode])
    if st.session_state.pop("quick_demo_resolution_warning", False):
        st.warning("Быстрый демо-расчёт использует грубую сетку, чтобы запускаться быстро.")

    dashboard = build_dashboard_inputs(st, workflow_mode, expert_mode)
    config = dashboard["config"]
    field_source = dashboard["field_source"]
    uploaded_field_file = dashboard["uploaded_field_file"]
    surface_options = dashboard["surface_options"]
    plot_quality = dashboard["plot_quality"]
    environment_report = dashboard["environment_report"]
    voltage_text = dashboard["voltage_text"]
    frequency_text = dashboard["frequency_text"]
    max_simulations = dashboard["max_simulations"]

    run_check, run_search, run_field_only = render_launch_card(
        st,
        dashboard["workflow_mode"],
        field_source,
        surface_options,
        voltage_text,
        frequency_text,
    )

    if run_check:
        run_single_simulation(
            st,
            config,
            plot_quality,
            field_source,
            uploaded_field_file,
            surface_options,
        )
    elif run_search:
        run_parameter_search_workflow(
            st,
            config,
            field_source,
            uploaded_field_file,
            surface_options,
            voltage_text,
            frequency_text,
            int(max_simulations),
        )
    elif run_field_only:
        run_field_only_workflow(
            st,
            config,
            plot_quality,
            field_source,
            uploaded_field_file,
            surface_options,
        )
    elif st.session_state.pop("run_quick_demo_after_load", False):
        run_single_simulation(
            st,
            config,
            plot_quality,
            field_source,
            uploaded_field_file,
            surface_options,
        )

    overview_tab, field_tab, pseudo_tab, trajectory_tab, diagnostics_tab, experiments_tab, export_tab = st.tabs(
        ["Обзор", "Поле", "Псевдопотенциал", "Траектория", "Диагностика", "Эксперименты", "Экспорт"]
    )

    with overview_tab:
        render_overview_tab(st, config, environment_report)
        if "last_parameter_search_results" in st.session_state:
            st.subheader("Последний подбор RF-параметров")
            st.dataframe(
                st.session_state["last_parameter_search_results"],
                use_container_width=True,
            )

    with field_tab:
        st.subheader("Поле")
        if "last_field_result" in st.session_state:
            field_result = st.session_state["last_field_result"]
            show_field_grid_diagnostics(st, field_result, config)
            fig = plot_potential_slice_better(field_result) if plot_quality == "Better quality" else None
            if fig is None:
                fig, ax = plt.subplots(figsize=(6, 5))
                plot_potential_slice(field_result, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
            fig = plot_electric_field_slice_better(field_result) if plot_quality == "Better quality" else None
            if fig is None:
                fig, ax = plt.subplots(figsize=(6, 5))
                plot_electric_field_slice(field_result, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Сначала рассчитайте или загрузите поле.")

    with pseudo_tab:
        st.subheader("Псевдопотенциал")
        if "last_field_result" not in st.session_state:
            st.info("Псевдопотенциал появится после расчёта поля.")
        else:
            fig = plot_pseudopotential_slice(st.session_state["last_field_result"], config)
            if fig is None:
                st.warning("RF voltage или RF frequency равны нулю, поэтому псевдопотенциал не строится.")
            else:
                st.pyplot(fig)
                plt.close(fig)

    with trajectory_tab:
        st.subheader("Траектория")
        if "last_particle_result" in st.session_state:
            particle_result = st.session_state["last_particle_result"]
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
            plot_trajectory_3d(particle_result, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
            for plotter in [plot_coordinates_vs_time, plot_radius_vs_time, plot_speed_vs_time]:
                fig, ax = plt.subplots(figsize=(7, 4))
                plotter(particle_result, ax=ax)
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("Траектория появится после запуска расчёта локализации.")

    with diagnostics_tab:
        st.subheader("Диагностика")
        st.warning(PHYSICS_WARNING_TEXT)
        show_environment_report(st, environment_report)
        if "last_trajectory_metrics" in st.session_state:
            st.json(
                {
                    "status": st.session_state.get("last_trajectory_status"),
                    "metrics": st.session_state["last_trajectory_metrics"],
                }
            )
        if expert_mode:
            st.divider()
            run_mathieu_analysis(st, config)

    with experiments_tab:
        render_experiment_tools(
            st,
            config,
            dashboard["workflow_mode"],
            dashboard["source_label"],
            surface_options,
            environment_report,
            voltage_text,
            frequency_text,
            max_simulations,
        )

    with export_tab:
        st.subheader("Экспорт")
        st.write(
            "Расчёты автоматически сохраняют FieldGrid .npz, trajectory.csv, "
            "metrics.csv или parameter_search_results.csv в timestamp-папки results/."
        )
        if st.button("Экспортировать legacy FEM поле .npz"):
            export_builtin_fem_field_from_ui(st, config)
        if "last_field_result" in st.session_state and st.button("Экспортировать текущее поле .npz"):
            output_dir = create_ui_output_dir(config.output_dir)
            export_path = output_dir / "current_field_grid.npz"
            save_field_result_to_npz(st.session_state["last_field_result"], export_path)
            st.success(f"Текущее поле сохранено в: {export_path}")


if __name__ == "__main__":
    main()
