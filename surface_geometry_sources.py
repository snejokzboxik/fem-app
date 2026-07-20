"""Surface-electrode geometry sources and RF/DC voltage assignment.

This module keeps geometry separate from electrical role assignment.  A region
can come from an uploaded image component or from a mathematical inequality;
its role then decides whether it contributes to the RF map, DC/static map, or
ground.
"""

from __future__ import annotations

import ast
import base64
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage

from field_data import FieldGrid


ROLE_RF = "RF"
ROLE_DC = "DC"
ROLE_GND = "GND"
ROLE_CUSTOM = "CUSTOM"
ELECTRODE_ROLES = (ROLE_RF, ROLE_DC, ROLE_GND, ROLE_CUSTOM)


@dataclass(frozen=True)
class ElectrodeRegionDefinition:
    """One mathematical electrode region definition."""

    name: str
    expression: str
    role: str = ROLE_GND
    voltage: float = 0.0
    rf_phase: float = 0.0


@dataclass(frozen=True)
class ElectrodeAssignment:
    """Electrical role assigned to a labeled electrode region."""

    region_id: int
    name: str
    role: str = ROLE_GND
    voltage: float = 0.0
    rf_phase: float = 0.0


@dataclass
class RasterizedElectrodeGeometry:
    """Labeled surface-electrode rasterization."""

    region_labels: np.ndarray
    assignments: list[ElectrodeAssignment]
    x_centers: np.ndarray
    y_centers: np.ndarray
    overlap_pixels: int = 0
    overlap_warning: str | None = None


class TwoChannelElectricField:
    """Callable electric field for separated RF and DC surface fields."""

    expects_time = True

    def __init__(
        self,
        E_rf_base: Callable,
        E_dc_static: Callable,
        rf_amplitude: float,
        rf_angular_frequency: float,
        rf_phase: float = 0.0,
    ):
        self.E_rf_base = E_rf_base
        self.E_dc_static = E_dc_static
        self.rf_amplitude = float(rf_amplitude)
        self.rf_angular_frequency = float(rf_angular_frequency)
        self.rf_phase = float(rf_phase)

    def __call__(self, t: float, position):
        rf_scale = self.rf_amplitude * np.cos(
            self.rf_angular_frequency * t + self.rf_phase
        )
        return self.E_dc_static(position) + rf_scale * self.E_rf_base(position)


def evaluate_region_expression_safe(expression: str, x, y):
    """Evaluate a whitelisted vectorized expression over x/y arrays."""

    tree = ast.parse(expression, mode="eval")
    evaluator = _SafeExpressionEvaluator({"x": x, "y": y})
    result = evaluator.visit(tree.body)
    result = np.asarray(result)
    if result.shape == ():
        result = np.full_like(np.asarray(x, dtype=float), bool(result), dtype=bool)
    if result.shape != np.asarray(x).shape:
        raise ValueError("Expression result must match the x/y grid shape.")
    return result.astype(bool)


def rasterize_function_regions(
    definitions: list[ElectrodeRegionDefinition],
    x_size_m: float,
    y_size_m: float,
    nx_mask: int,
    ny_mask: int,
) -> RasterizedElectrodeGeometry:
    """Rasterize function-defined electrode regions on an image-like grid."""

    if nx_mask < 1 or ny_mask < 1:
        raise ValueError("nx_mask and ny_mask must be positive.")
    x_centers = -0.5 * x_size_m + (np.arange(nx_mask) + 0.5) * x_size_m / nx_mask
    y_centers = 0.5 * y_size_m - (np.arange(ny_mask) + 0.5) * y_size_m / ny_mask
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)

    labels = np.zeros((ny_mask, nx_mask), dtype=int)
    assignments = []
    overlap_pixels = 0
    for index, definition in enumerate(definitions, start=1):
        region_mask = evaluate_region_expression_safe(definition.expression, x_grid, y_grid)
        overlap_pixels += int(np.count_nonzero(region_mask & (labels != 0)))
        labels[region_mask] = index
        assignments.append(
            ElectrodeAssignment(
                region_id=index,
                name=definition.name or f"electrode {index}",
                role=definition.role,
                voltage=definition.voltage,
                rf_phase=definition.rf_phase,
            )
        )

    warning = None
    if overlap_pixels > 0:
        warning = (
            "Некоторые области электродов пересекаются. Более поздние "
            "определения перекрывают более ранние."
        )
    return RasterizedElectrodeGeometry(
        region_labels=labels,
        assignments=assignments,
        x_centers=x_centers,
        y_centers=y_centers,
        overlap_pixels=overlap_pixels,
        overlap_warning=warning,
    )


def build_regions_from_function_definitions(
    definitions: list[ElectrodeRegionDefinition],
    x_size_m: float,
    y_size_m: float,
    nx_mask: int,
    ny_mask: int,
) -> RasterizedElectrodeGeometry:
    """Compatibility wrapper for function-defined regions."""

    return rasterize_function_regions(definitions, x_size_m, y_size_m, nx_mask, ny_mask)


def build_regions_from_uploaded_mask(
    component_labels: np.ndarray,
    assignments_by_region_id: dict[int, ElectrodeAssignment] | None = None,
) -> RasterizedElectrodeGeometry:
    """Build region assignments from connected components in an uploaded mask."""

    labels = np.asarray(component_labels, dtype=int)
    assignments = []
    for region_id in range(1, int(np.max(labels)) + 1):
        default = ElectrodeAssignment(region_id, f"electrode {region_id}", ROLE_GND)
        assignments.append(
            assignments_by_region_id.get(region_id, default)
            if assignments_by_region_id
            else default
        )
    return RasterizedElectrodeGeometry(
        region_labels=labels,
        assignments=assignments,
        x_centers=np.arange(labels.shape[1]),
        y_centers=np.arange(labels.shape[0]),
    )


def canvas_image_to_binary_mask(
    canvas_image,
    darkness_threshold: float = 0.85,
    alpha_threshold: float = 0.05,
) -> np.ndarray:
    """Convert an RGBA/RGB canvas image into a boolean electrode mask.

    The Streamlit canvas uses a light background and dark strokes by default.
    Pixels are treated as drawn electrodes when they are visible and dark.
    """

    image = np.asarray(canvas_image)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError("canvas_image must have shape (height, width, 3 or 4).")
    image = image.astype(float)
    if np.nanmax(image) > 1.0:
        image = image / 255.0
    rgb = image[..., :3]
    alpha = image[..., 3] if image.shape[-1] == 4 else np.ones(image.shape[:2])
    luminance = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    visible = alpha > float(alpha_threshold)
    dark = luminance < float(darkness_threshold)
    return np.asarray(visible & dark, dtype=bool)


def clean_binary_mask(
    mask: np.ndarray,
    min_component_area: int = 8,
) -> np.ndarray:
    """Remove tiny connected components from a binary canvas mask."""

    mask = np.asarray(mask, dtype=bool)
    if min_component_area <= 1:
        return mask.copy()
    structure = np.ones((3, 3), dtype=int)
    labels, number_of_components = ndimage.label(mask, structure=structure)
    cleaned = np.zeros_like(mask, dtype=bool)
    for component_id in range(1, int(number_of_components) + 1):
        component = labels == component_id
        if int(np.count_nonzero(component)) >= int(min_component_area):
            cleaned[component] = True
    return cleaned


def label_canvas_electrodes(
    mask: np.ndarray,
    min_component_area: int = 8,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Clean and label connected electrode regions from a canvas mask."""

    cleaned_mask = clean_binary_mask(mask, min_component_area=min_component_area)
    structure = np.ones((3, 3), dtype=int)
    labels, number_of_components = ndimage.label(cleaned_mask, structure=structure)
    return labels.astype(int), int(number_of_components), cleaned_mask


def build_regions_from_canvas_mask(
    mask: np.ndarray,
    min_component_area: int = 8,
    assignments_by_region_id: dict[int, ElectrodeAssignment] | None = None,
) -> RasterizedElectrodeGeometry:
    """Build labeled canvas regions with default/manual assignments."""

    labels, number_of_components, _cleaned_mask = label_canvas_electrodes(
        mask,
        min_component_area=min_component_area,
    )
    assignments = []
    for region_id in range(1, number_of_components + 1):
        default = ElectrodeAssignment(region_id, f"canvas electrode {region_id}", ROLE_GND)
        assignments.append(
            assignments_by_region_id.get(region_id, default)
            if assignments_by_region_id
            else default
        )
    return RasterizedElectrodeGeometry(
        region_labels=labels,
        assignments=assignments,
        x_centers=np.arange(labels.shape[1]),
        y_centers=np.arange(labels.shape[0]),
    )


def canvas_design_to_voltage_maps(
    region_labels: np.ndarray,
    assignments: list[ElectrodeAssignment],
) -> tuple[np.ndarray, np.ndarray]:
    """Build RF/DC voltage maps for a canvas design."""

    return build_voltage_maps_from_assignments(region_labels, assignments)


def _assignment_to_dict(assignment: ElectrodeAssignment) -> dict:
    """Serialize an electrode assignment for design JSON."""

    return {
        "region_id": int(assignment.region_id),
        "name": assignment.name,
        "role": assignment.role,
        "voltage": float(assignment.voltage),
        "rf_phase": float(assignment.rf_phase),
    }


def _assignment_from_dict(data: dict) -> ElectrodeAssignment:
    """Deserialize one electrode assignment from design JSON."""

    return ElectrodeAssignment(
        region_id=int(data["region_id"]),
        name=str(data.get("name", f"electrode {data['region_id']}")),
        role=str(data.get("role", ROLE_GND)),
        voltage=float(data.get("voltage", 0.0)),
        rf_phase=float(data.get("rf_phase", 0.0)),
    )


def _encode_binary_mask(mask: np.ndarray) -> dict:
    """Encode a boolean mask compactly for JSON."""

    mask = np.asarray(mask, dtype=bool)
    packed = np.packbits(mask.reshape(-1).astype(np.uint8))
    return {
        "encoding": "base64_packbits",
        "shape": [int(mask.shape[0]), int(mask.shape[1])],
        "data": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _decode_binary_mask(encoded: dict) -> np.ndarray:
    """Decode a boolean mask from JSON data."""

    if encoded.get("encoding") != "base64_packbits":
        raise ValueError("Unsupported mask encoding.")
    shape = tuple(int(v) for v in encoded["shape"])
    raw = base64.b64decode(encoded["data"].encode("ascii"))
    packed = np.frombuffer(raw, dtype=np.uint8)
    flat = np.unpackbits(packed)[: shape[0] * shape[1]]
    return flat.reshape(shape).astype(bool)


def canvas_design_to_dict(
    *,
    x_size_m: float,
    y_size_m: float,
    canvas_resolution_px: int,
    binary_mask: np.ndarray,
    assignments: list[ElectrodeAssignment],
    rf_amplitude: float | None = None,
    rf_frequency_hz: float | None = None,
    notes: str = "",
) -> dict:
    """Create a JSON-serializable canvas electrode design snapshot."""

    return {
        "version": 1,
        "geometry_source_type": "canvas",
        "x_size_m": float(x_size_m),
        "y_size_m": float(y_size_m),
        "canvas_resolution_px": int(canvas_resolution_px),
        "binary_mask": _encode_binary_mask(binary_mask),
        "assignments": [_assignment_to_dict(assignment) for assignment in assignments],
        "rf_amplitude": None if rf_amplitude is None else float(rf_amplitude),
        "rf_frequency_hz": None if rf_frequency_hz is None else float(rf_frequency_hz),
        "notes": str(notes),
    }


def canvas_design_from_dict(data: dict) -> dict:
    """Load a canvas electrode design snapshot from a dictionary."""

    if data.get("geometry_source_type") != "canvas":
        raise ValueError("Design JSON is not a canvas geometry design.")
    return {
        "version": int(data.get("version", 1)),
        "x_size_m": float(data["x_size_m"]),
        "y_size_m": float(data["y_size_m"]),
        "canvas_resolution_px": int(data["canvas_resolution_px"]),
        "binary_mask": _decode_binary_mask(data["binary_mask"]),
        "assignments": [
            _assignment_from_dict(assignment)
            for assignment in data.get("assignments", [])
        ],
        "rf_amplitude": data.get("rf_amplitude"),
        "rf_frequency_hz": data.get("rf_frequency_hz"),
        "notes": str(data.get("notes", "")),
    }


def build_voltage_maps_from_assignments(
    region_labels: np.ndarray,
    assignments: list[ElectrodeAssignment],
    default_dc_voltage: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized RF and physical DC/static voltage maps."""

    labels = np.asarray(region_labels, dtype=int)
    rf_voltage_map_base = np.zeros_like(labels, dtype=float)
    dc_voltage_map = np.zeros_like(labels, dtype=float)
    for assignment in assignments:
        if assignment.role not in ELECTRODE_ROLES:
            raise ValueError(f"Unknown electrode role: {assignment.role}")
        region_mask = labels == int(assignment.region_id)
        if assignment.role == ROLE_RF:
            rf_voltage_map_base[region_mask] = 1.0
        elif assignment.role == ROLE_DC:
            dc_voltage_map[region_mask] = float(
                assignment.voltage if assignment.voltage is not None else default_dc_voltage
            )
        elif assignment.role == ROLE_CUSTOM:
            dc_voltage_map[region_mask] = float(assignment.voltage)
        elif assignment.role == ROLE_GND:
            dc_voltage_map[region_mask] = 0.0
    return rf_voltage_map_base, dc_voltage_map


def combine_field_grids_for_preview(
    rf_field_grid: FieldGrid,
    dc_field_grid: FieldGrid,
    rf_amplitude: float,
) -> FieldGrid:
    """Create a display FieldGrid from DC plus RF-at-cosine-maximum fields."""

    potential_grid = None
    if rf_field_grid.potential_grid is not None and dc_field_grid.potential_grid is not None:
        potential_grid = dc_field_grid.potential_grid + rf_amplitude * rf_field_grid.potential_grid
    combined = FieldGrid(
        x_grid=rf_field_grid.x_grid,
        y_grid=rf_field_grid.y_grid,
        z_grid=rf_field_grid.z_grid,
        electric_field_grid=(
            dc_field_grid.electric_field_grid
            + rf_amplitude * rf_field_grid.electric_field_grid
        ),
        potential_grid=potential_grid,
        source_description="Surface RF/DC separated field preview",
    )
    combined.rf_field_grid = rf_field_grid
    combined.dc_field_grid = dc_field_grid
    combined.surface_rf_dc_separated = True
    return combined


def compute_pseudopotential_from_rf_field_grid(
    rf_field_grid: FieldGrid,
    particle_charge: float,
    particle_mass: float,
    rf_voltage: float,
    rf_angular_frequency: float,
) -> np.ndarray:
    """Compute pseudopotential from RF field only."""

    if rf_angular_frequency == 0.0:
        raise ValueError("rf_angular_frequency must be nonzero.")
    field_magnitude = np.linalg.norm(rf_field_grid.electric_field_grid, axis=-1)
    return (
        particle_charge**2
        * (rf_voltage * field_magnitude) ** 2
        / (4.0 * particle_mass * rf_angular_frequency**2)
    )


def function_geometry_presets() -> dict[str, list[ElectrodeRegionDefinition]]:
    """Return simple function-defined electrode presets."""

    return {
        "Четыре прямоугольных электрода": [
            ElectrodeRegionDefinition(
                "RF left",
                "abs(x + 250e-6) < 80e-6 and abs(y) < 250e-6",
                ROLE_RF,
            ),
            ElectrodeRegionDefinition(
                "RF right",
                "abs(x - 250e-6) < 80e-6 and abs(y) < 250e-6",
                ROLE_RF,
            ),
            ElectrodeRegionDefinition(
                "DC top",
                "abs(y - 250e-6) < 80e-6 and abs(x) < 250e-6",
                ROLE_DC,
                1.0,
            ),
            ElectrodeRegionDefinition(
                "GND bottom",
                "abs(y + 250e-6) < 80e-6 and abs(x) < 250e-6",
                ROLE_GND,
            ),
        ],
        "Параболические электроды": [
            ElectrodeRegionDefinition(
                "RF parabola top",
                "y > 0.2*(x/1e-4)**2 + 120e-6 and y < 0.2*(x/1e-4)**2 + 220e-6",
                ROLE_RF,
            ),
            ElectrodeRegionDefinition(
                "DC parabola bottom",
                "y < -0.2*(x/1e-4)**2 - 120e-6 and y > -0.2*(x/1e-4)**2 - 220e-6",
                ROLE_DC,
                -1.0,
            ),
        ],
        "Кольцевой электрод": [
            ElectrodeRegionDefinition(
                "RF ring",
                "(x**2 + y**2) > (150e-6)**2 and (x**2 + y**2) < (250e-6)**2",
                ROLE_RF,
            ),
            ElectrodeRegionDefinition(
                "DC center",
                "(x**2 + y**2) < (90e-6)**2",
                ROLE_DC,
                1.0,
            ),
        ],
        "Асимметричный пример": [
            ElectrodeRegionDefinition("RF disk", "(x**2 + y**2) < (180e-6)**2", ROLE_RF),
            ElectrodeRegionDefinition("Custom right", "x > 180e-6 and abs(y) < 120e-6", ROLE_CUSTOM, 2.0),
        ],
        "RF + DC compensation electrodes": [
            ElectrodeRegionDefinition("RF rail +", "abs(x - 220e-6) < 70e-6 and abs(y) < 320e-6", ROLE_RF),
            ElectrodeRegionDefinition("RF rail -", "abs(x + 220e-6) < 70e-6 and abs(y) < 320e-6", ROLE_RF),
            ElectrodeRegionDefinition("DC comp top", "abs(y - 280e-6) < 60e-6 and abs(x) < 180e-6", ROLE_DC, 0.5),
            ElectrodeRegionDefinition("DC comp bottom", "abs(y + 280e-6) < 60e-6 and abs(x) < 180e-6", ROLE_DC, -0.5),
        ],
    }


class _SafeExpressionEvaluator(ast.NodeVisitor):
    """Strict AST evaluator for vectorized x/y expressions."""

    _functions = {
        "abs": np.abs,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
    }
    _constants = {
        "pi": np.pi,
        "e": np.e,
    }

    def __init__(self, variables: dict[str, np.ndarray]):
        self.variables = variables

    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ValueError("Only numeric and boolean constants are allowed.")

    def visit_Name(self, node):
        if node.id in self.variables:
            return self.variables[node.id]
        if node.id in self._constants:
            return self._constants[node.id]
        raise ValueError(f"Unknown name: {node.id}")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        operators = {
            ast.Add: np.add,
            ast.Sub: np.subtract,
            ast.Mult: np.multiply,
            ast.Div: np.divide,
            ast.Pow: np.power,
        }
        for operator_type, function in operators.items():
            if isinstance(node.op, operator_type):
                return function(left, right)
        raise ValueError("Unsupported binary operator.")

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.Not):
            return np.logical_not(operand)
        raise ValueError("Unsupported unary operator.")

    def visit_BoolOp(self, node):
        values = [self.visit(value) for value in node.values]
        if isinstance(node.op, ast.And):
            result = values[0]
            for value in values[1:]:
                result = np.logical_and(result, value)
            return result
        if isinstance(node.op, ast.Or):
            result = values[0]
            for value in values[1:]:
                result = np.logical_or(result, value)
            return result
        raise ValueError("Unsupported boolean operator.")

    def visit_Compare(self, node):
        left = self.visit(node.left)
        comparisons = []
        for operator, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            comparisons.append(_compare_values(left, right, operator))
            left = right
        result = comparisons[0]
        for comparison in comparisons[1:]:
            result = np.logical_and(result, comparison)
        return result

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct whitelisted function calls are allowed.")
        if node.func.id not in self._functions:
            raise ValueError(f"Function is not allowed: {node.func.id}")
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed.")
        args = [self.visit(arg) for arg in node.args]
        return self._functions[node.func.id](*args)


def _compare_values(left, right, operator):
    """Evaluate one comparison node."""

    if isinstance(operator, ast.Lt):
        return left < right
    if isinstance(operator, ast.LtE):
        return left <= right
    if isinstance(operator, ast.Gt):
        return left > right
    if isinstance(operator, ast.GtE):
        return left >= right
    if isinstance(operator, ast.Eq):
        return left == right
    if isinstance(operator, ast.NotEq):
        return left != right
    raise ValueError("Unsupported comparison operator.")
