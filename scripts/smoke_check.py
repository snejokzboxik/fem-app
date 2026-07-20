"""Lightweight project smoke check for QA materials."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

QA_DOCS = [
    "docs/qa/README_FOR_TESTER.md",
    "docs/qa/QUICK_START_FOR_TESTER.md",
    "docs/qa/TEST_PLAN_FULL.md",
    "docs/qa/BUG_REPORT_TEMPLATE.md",
    "docs/qa/TEST_REPORT_TEMPLATE.md",
    "docs/qa/KNOWN_LIMITATIONS.md",
    "docs/qa/manual_test_checklist.csv",
]

QA_ASSETS = [
    "examples/qa_assets/generate_qa_assets.py",
    "examples/qa_assets/README_TEST_ASSETS.md",
    "examples/qa_assets/mask_four_rectangles.png",
    "examples/qa_assets/mask_ring.png",
    "examples/qa_assets/mask_asymmetric_rf_dc.png",
    "examples/qa_assets/mask_many_small_components.png",
    "examples/qa_assets/mask_black_electrodes_on_white.png",
    "examples/qa_assets/sample_canvas_design.json",
    "examples/qa_assets/sample_experiment_config.json",
]

LAUNCHER_FILES = [
    "launcher.py",
    "run_app_windows.bat",
    "run_app_windows.ps1",
    "scripts/build_windows_launcher.ps1",
    "docs/LAUNCH_WINDOWS.md",
]

REGRESSION_FILES = [
    "scripts/run_regression_scenarios.py",
]

KEY_MODULES = [
    "app",
    "config",
    "field_data",
    "field_interpolation",
    "particle_dynamics",
    "surface_geometry_sources",
    "surface_superposition",
]


def _load_asset_generator():
    """Load the QA asset generator without requiring package metadata."""

    generator_path = PROJECT_ROOT / "examples/qa_assets/generate_qa_assets.py"
    spec = importlib.util.spec_from_file_location("generate_qa_assets", generator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Не удалось загрузить generate_qa_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _missing(paths: list[str]) -> list[str]:
    """Return paths that do not exist relative to the project root."""

    return [path for path in paths if not (PROJECT_ROOT / path).exists()]


def _assert_ascii_file(relative_path: str) -> None:
    """Raise a readable error if a text file is not ASCII-decodable."""

    path = PROJECT_ROOT / relative_path
    try:
        path.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{relative_path} must be ASCII-only.") from exc


def smoke_check(generate_missing_assets: bool = True) -> dict:
    """Run lightweight checks and return a Russian summary dictionary."""

    imported_modules = []
    for module_name in KEY_MODULES:
        importlib.import_module(module_name)
        imported_modules.append(module_name)

    missing_docs = _missing(QA_DOCS)
    if missing_docs:
        raise FileNotFoundError(
            "Не найдены QA-документы: " + ", ".join(missing_docs)
        )

    missing_launchers = _missing(LAUNCHER_FILES)
    if missing_launchers:
        raise FileNotFoundError(
            "Не найдены файлы запуска: " + ", ".join(missing_launchers)
        )
    _assert_ascii_file("run_app_windows.bat")

    missing_regression_files = _missing(REGRESSION_FILES)
    if missing_regression_files:
        raise FileNotFoundError(
            "Regression scripts are missing: " + ", ".join(missing_regression_files)
        )

    missing_assets = _missing(QA_ASSETS)
    if missing_assets and generate_missing_assets:
        generator = _load_asset_generator()
        generator.generate_qa_assets(PROJECT_ROOT / "examples/qa_assets")
        missing_assets = _missing(QA_ASSETS)
    if missing_assets:
        raise FileNotFoundError("Не найдены QA-ассеты: " + ", ".join(missing_assets))

    return {
        "status": "ok",
        "imported_modules": imported_modules,
        "qa_docs_checked": len(QA_DOCS),
        "qa_assets_checked": len(QA_ASSETS),
        "launcher_files_checked": len(LAUNCHER_FILES),
        "regression_files_checked": len(REGRESSION_FILES),
    }


def main() -> int:
    """CLI entry point with Russian output for a manual tester."""

    try:
        report = smoke_check(generate_missing_assets=True)
    except Exception as exc:
        print("Smoke check не прошёл.")
        print(f"Ошибка: {exc}")
        return 1

    print("Smoke check прошёл.")
    print(f"Проверено QA-документов: {report['qa_docs_checked']}")
    print(f"Проверено QA-ассетов: {report['qa_assets_checked']}")
    print(f"Проверено файлов запуска: {report['launcher_files_checked']}")
    print("Ключевые модули импортируются:")
    for module_name in report["imported_modules"]:
        print(f"- {module_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
