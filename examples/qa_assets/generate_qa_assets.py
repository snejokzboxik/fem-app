"""Generate deterministic QA masks and sample JSON files."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ASSET_FILENAMES = [
    "mask_four_rectangles.png",
    "mask_ring.png",
    "mask_asymmetric_rf_dc.png",
    "mask_many_small_components.png",
    "mask_black_electrodes_on_white.png",
    "sample_canvas_design.json",
    "sample_experiment_config.json",
]


def _save_mask(mask: np.ndarray, path: Path) -> None:
    """Save a boolean/uint8 mask as an 8-bit grayscale PNG."""

    image = np.asarray(mask, dtype=np.uint8)
    if image.max() <= 1:
        image = image * 255
    Image.fromarray(image, mode="L").save(path)


def _four_rectangles(size: int = 512) -> np.ndarray:
    """Return four separated white rectangular electrodes on black background."""

    mask = np.zeros((size, size), dtype=np.uint8)
    mask[150:362, 72:140] = 255
    mask[150:362, 372:440] = 255
    mask[72:140, 150:362] = 255
    mask[372:440, 150:362] = 255
    return mask


def _ring(size: int = 512) -> np.ndarray:
    """Return a white ring with a black center hole."""

    yy, xx = np.indices((size, size))
    cx = cy = (size - 1) / 2.0
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.where((radius > 108) & (radius < 178), 255, 0).astype(np.uint8)


def _asymmetric_rf_dc(size: int = 512) -> np.ndarray:
    """Return several asymmetric electrodes for manual role assignment."""

    mask = np.zeros((size, size), dtype=np.uint8)
    mask[80:420, 70:130] = 255
    mask[125:350, 380:445] = 255
    mask[70:130, 180:300] = 255
    mask[355:430, 190:330] = 255
    mask[210:285, 215:295] = 255
    return mask


def _many_small_components(size: int = 512) -> np.ndarray:
    """Return normal electrodes plus deterministic tiny components."""

    mask = _four_rectangles(size)
    rng = np.random.default_rng(20260705)
    for y, x in rng.integers(30, size - 30, size=(45, 2)):
        mask[y : y + 3, x : x + 3] = 255
    return mask


def _black_electrodes_on_white(size: int = 512) -> np.ndarray:
    """Return black electrodes on white background to test polarity handling."""

    image = np.full((size, size), 255, dtype=np.uint8)
    image[150:362, 72:140] = 0
    image[150:362, 372:440] = 0
    image[90:155, 185:327] = 0
    image[357:422, 185:327] = 0
    return image


def _write_sample_canvas_design(path: Path) -> None:
    """Write a small canvas design JSON using the app's real format."""

    from surface_geometry_sources import (
        ROLE_DC,
        ROLE_GND,
        ROLE_RF,
        ElectrodeAssignment,
        canvas_design_to_dict,
    )

    mask = np.zeros((96, 96), dtype=bool)
    mask[24:72, 12:22] = True
    mask[24:72, 74:84] = True
    mask[12:22, 34:62] = True
    mask[74:84, 34:62] = True
    assignments = [
        ElectrodeAssignment(1, "RF слева", ROLE_RF),
        ElectrodeAssignment(2, "RF справа", ROLE_RF),
        ElectrodeAssignment(3, "DC сверху", ROLE_DC, 0.5),
        ElectrodeAssignment(4, "GND снизу", ROLE_GND, 0.0),
    ]
    design = canvas_design_to_dict(
        x_size_m=1.0e-3,
        y_size_m=1.0e-3,
        canvas_resolution_px=96,
        binary_mask=mask,
        assignments=assignments,
        rf_amplitude=20.0,
        rf_frequency_hz=3.0e4,
        notes="QA пример canvas-дизайна.",
    )
    path.write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_sample_experiment_config(path: Path) -> None:
    """Write a sample experiment config JSON if the preset helper exists."""

    from experiment_presets import get_experiment_preset

    preset = get_experiment_preset("Быстрый старт: 4 RF электрода")
    preset["title"] = "QA sample experiment config"
    preset["description"] = (
        "Пример experiment_config.json для ручной проверки загрузки настроек."
    )
    path.write_text(json.dumps(preset, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_qa_assets(output_dir: str | Path | None = None) -> list[Path]:
    """Generate all QA assets and return their paths."""

    output = Path(output_dir) if output_dir is not None else Path(__file__).parent
    output.mkdir(parents=True, exist_ok=True)

    generated = []
    masks = {
        "mask_four_rectangles.png": _four_rectangles(),
        "mask_ring.png": _ring(),
        "mask_asymmetric_rf_dc.png": _asymmetric_rf_dc(),
        "mask_many_small_components.png": _many_small_components(),
        "mask_black_electrodes_on_white.png": _black_electrodes_on_white(),
    }
    for filename, mask in masks.items():
        path = output / filename
        _save_mask(mask, path)
        generated.append(path)

    canvas_path = output / "sample_canvas_design.json"
    _write_sample_canvas_design(canvas_path)
    generated.append(canvas_path)

    experiment_path = output / "sample_experiment_config.json"
    _write_sample_experiment_config(experiment_path)
    generated.append(experiment_path)

    return generated


def main() -> int:
    """CLI entry point."""

    paths = generate_qa_assets()
    print("QA-ассеты созданы:")
    for path in paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
