"""Compare two structured field-grid NPZ files.

Example:

    python compare_fields.py --reference data/builtin_fem_field_grid.npz \
        --candidate data/external_field_grid.npz

For this first validation tool, the coordinate grids must match exactly.  Later
versions can add interpolation between different grids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from field_data import load_field_grid_npz, load_potential_grid_npz
from field_validation import (
    compare_field_grids,
    compare_potential_grids,
    estimate_symmetry_checks,
    grids_match_exactly,
    validate_field_grid,
)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Validate and compare two field or potential .npz files."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("comparison_report.json"),
        help="Path for the saved JSON report.",
    )
    return parser.parse_args()


def load_any_field_npz(path: Path):
    """Load either a field-grid or potential-grid NPZ file."""

    with np.load(path, allow_pickle=False) as data:
        has_field = "electric_field_grid" in data.files
        has_potential = "potential_grid" in data.files

    if has_field:
        return load_field_grid_npz(path)
    if has_potential:
        return load_potential_grid_npz(path)

    raise ValueError(
        f"{path} must contain either electric_field_grid or potential_grid."
    )


def save_report(report: dict, output_path: Path):
    """Save a JSON report with readable indentation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def print_validation_summary(name: str, report: dict):
    """Print a compact validation summary."""

    print(f"\n{name} validation")
    print("-" * 40)
    print(f"valid: {report['valid']}")
    print(f"grid_shape: {report['grid_shape']}")
    print(f"electric_field_shape: {report['electric_field_shape']}")
    print(f"has_potential: {report['has_potential']}")
    if report["errors"]:
        print("errors:")
        for error in report["errors"]:
            print(f"  - {error}")
    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")


def print_error_metrics(title: str, metrics: dict):
    """Print common comparison metrics."""

    print(f"\n{title}")
    print("-" * 40)
    print(f"mean_abs_error: {metrics['mean_abs_error']:.6e}")
    print(f"max_abs_error: {metrics['max_abs_error']:.6e}")
    print(f"relative_l2_error: {metrics['relative_l2_error']:.6e}")


def main() -> int:
    """Run the validation and comparison workflow."""

    args = parse_args()

    reference = load_any_field_npz(args.reference)
    candidate = load_any_field_npz(args.candidate)

    reference_report = validate_field_grid(reference)
    candidate_report = validate_field_grid(candidate)

    report = {
        "reference_path": str(args.reference),
        "candidate_path": str(args.candidate),
        "reference_validation": reference_report,
        "candidate_validation": candidate_report,
        "grids_match_exactly": grids_match_exactly(reference, candidate),
        "potential_comparison": None,
        "field_comparison": None,
        "reference_symmetry": estimate_symmetry_checks(reference),
        "candidate_symmetry": estimate_symmetry_checks(candidate),
        "errors": [],
    }

    print_validation_summary("Reference", reference_report)
    print_validation_summary("Candidate", candidate_report)

    if not reference_report["valid"] or not candidate_report["valid"]:
        report["errors"].append("Validation failed; comparison was skipped.")
        save_report(report, args.output)
        print(f"\nSaved report to: {args.output}")
        return 1

    if not report["grids_match_exactly"]:
        message = (
            "Coordinate grids do not match exactly. This first version does not "
            "interpolate between grids."
        )
        report["errors"].append(message)
        save_report(report, args.output)
        print(f"\nError: {message}")
        print(f"Saved report to: {args.output}")
        return 1

    if reference.potential_grid is not None and candidate.potential_grid is not None:
        report["potential_comparison"] = compare_potential_grids(reference, candidate)
        print_error_metrics(
            "Potential comparison",
            report["potential_comparison"],
        )
    else:
        print("\nPotential comparison skipped: one file has no potential_grid.")

    report["field_comparison"] = compare_field_grids(reference, candidate)
    print_error_metrics("Electric-field comparison", report["field_comparison"])

    save_report(report, args.output)
    print(f"\nSaved report to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
