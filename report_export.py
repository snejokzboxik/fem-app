"""Human-readable report and ZIP export helpers."""

from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path
import zipfile

from experiment_config import ExperimentConfig, _json_safe


def build_results_summary(
    *,
    experiment_config: ExperimentConfig | dict,
    metrics: dict | None = None,
    field_diagnostics: dict | None = None,
    environment_report: dict | None = None,
    exported_files: list[str] | None = None,
) -> dict:
    """Build a compact report summary dictionary."""

    config = (
        experiment_config.to_dict()
        if isinstance(experiment_config, ExperimentConfig)
        else _json_safe(experiment_config)
    )
    return {
        "title": config.get("title", "Untitled experiment"),
        "description": config.get("description", ""),
        "created_at": config.get("created_at", datetime.now().isoformat()),
        "geometry_source": config.get("geometry_source", ""),
        "workflow_mode": config.get("workflow_mode", ""),
        "electrode_assignments": config.get("electrode_assignments", []),
        "rf_config": config.get("rf_config", {}),
        "dc_config": config.get("dc_config", {}),
        "particle_config": config.get("particle_config", {}),
        "environment_config": environment_report or config.get("environment_config", {}),
        "dynamics_config": config.get("dynamics_config", {}),
        "metrics": metrics or {},
        "field_diagnostics": field_diagnostics or {},
        "exported_files": exported_files or [],
        "notes": config.get("notes", ""),
        "warnings": [
            "Surface fields use a simplified Dirichlet half-space Poisson-kernel model.",
            "Pseudopotential diagnostics use RF field only; DC fields affect full dynamics.",
            "Mathieu parameters for arbitrary surface geometry are local effective estimates.",
        ],
    }


def _markdown_table(rows: list[tuple[str, object]]) -> str:
    """Return a two-column Markdown table."""

    lines = ["| Quantity | Value |", "|---|---|"]
    for key, value in rows:
        lines.append(f"| {key} | `{value}` |")
    return "\n".join(lines)


def build_markdown_report(
    *,
    experiment_config: ExperimentConfig | dict,
    metrics: dict | None = None,
    field_diagnostics: dict | None = None,
    environment_report: dict | None = None,
    exported_files: list[str] | None = None,
) -> str:
    """Build a human-readable Markdown experiment report."""

    summary = build_results_summary(
        experiment_config=experiment_config,
        metrics=metrics,
        field_diagnostics=field_diagnostics,
        environment_report=environment_report,
        exported_files=exported_files,
    )
    metrics = summary["metrics"]
    field = summary["field_diagnostics"]
    environment = summary["environment_config"]
    lines = [
        f"# {summary['title']}",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Experiment",
        summary["description"],
        "",
        _markdown_table(
            [
                ("Workflow mode", summary["workflow_mode"]),
                ("Geometry source", summary["geometry_source"]),
                ("RF settings", summary["rf_config"]),
                ("DC settings", summary["dc_config"]),
            ]
        ),
        "",
        "## Electrode Roles",
    ]
    assignments = summary["electrode_assignments"]
    if assignments:
        lines.extend(
            [
                "| Region | Name | Role | Voltage [V] |",
                "|---:|---|---|---:|",
            ]
        )
        for assignment in assignments:
            lines.append(
                f"| {assignment.get('region_id')} | {assignment.get('name')} | "
                f"{assignment.get('role')} | {assignment.get('voltage', 0.0)} |"
            )
    else:
        lines.append("No electrode assignments were captured.")
    lines.extend(
        [
            "",
            "## Particle And Environment",
            _markdown_table(
                [
                    ("Particle", summary["particle_config"]),
                    ("Environment", environment),
                    ("Dynamics", summary["dynamics_config"]),
                ]
            ),
            "",
            "## Results",
            _markdown_table(
                [
                    ("Localization status", metrics.get("status", "n/a")),
                    ("max radius [m]", metrics.get("max_radius", "n/a")),
                    ("final radius [m]", metrics.get("final_radius", "n/a")),
                    ("final speed [m/s]", metrics.get("final_speed", "n/a")),
                    ("min |E| candidate", field.get("min_field_position", "n/a")),
                    ("min |E| [V/m]", field.get("min_field_magnitude", "n/a")),
                    (
                        "pseudopotential minimum [J]",
                        (field.get("pseudopotential") or {}).get("min_value_J", "n/a"),
                    ),
                    ("gamma [kg/s]", environment.get("gamma_kg_s", "n/a")),
                    ("damping time [s]", environment.get("damping_time_s", "n/a")),
                ]
            ),
        ]
    )
    if summary["notes"]:
        lines.extend(["", "## Notes", summary["notes"]])
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in summary["warnings"])
    lines.extend(["", "## Exported Files"])
    if summary["exported_files"]:
        lines.extend(f"- `{filename}`" for filename in summary["exported_files"])
    else:
        lines.append("No external files listed.")
    return "\n".join(lines) + "\n"


def build_html_report(**kwargs) -> str:
    """Build a simple HTML report from the Markdown report text."""

    markdown = build_markdown_report(**kwargs)
    escaped = html.escape(markdown)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Charged particle trap report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:960px;margin:32px auto;"
        "line-height:1.5;} pre{white-space:pre-wrap;background:#f6f8fa;padding:16px;"
        "border-radius:6px;}</style></head><body><pre>"
        f"{escaped}</pre></body></html>"
    )


def export_experiment_zip(
    zip_path,
    *,
    experiment_config: ExperimentConfig | dict,
    report_md: str,
    report_html: str | None = None,
    extra_files: dict[str, str | Path | bytes] | None = None,
    include_field_grid: bool = False,
) -> Path:
    """Export a reproducible experiment package as a ZIP file."""

    config = (
        experiment_config.to_dict()
        if isinstance(experiment_config, ExperimentConfig)
        else _json_safe(experiment_config)
    )
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "experiment_config.json",
            json.dumps(config, indent=2, ensure_ascii=False),
        )
        archive.writestr("report.md", report_md)
        if report_html is not None:
            archive.writestr("report.html", report_html)
        archive.writestr(
            "README_EXPERIMENT.txt",
            "This ZIP contains a reproducible charged_particle_trap UI experiment.\n"
            "Large FieldGrid arrays are included only when explicitly requested.\n",
        )
        for archive_name, source in (extra_files or {}).items():
            if archive_name.endswith("field_grid.npz") and not include_field_grid:
                continue
            if isinstance(source, bytes):
                archive.writestr(archive_name, source)
            else:
                source_path = Path(source)
                if source_path.exists():
                    archive.write(source_path, arcname=archive_name)
    return zip_path
