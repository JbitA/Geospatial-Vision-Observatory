#!/usr/bin/env python3
"""One-command real-data, multi-seed training and publication workflow."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geo_vision.ml.data import dataset_signature  # noqa: E402
from geo_vision.ml.schema import SHOWCASE_SPLIT  # noqa: E402


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_status(command: list[str]) -> int:
    """Run a command and return its status without weakening unexpected-failure handling."""

    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_optional(command: list[str], *, label: str) -> bool:
    """Run non-runtime evidence tooling without turning an optional check into a runtime outage."""

    status = run_status(command)
    if status != 0:
        print(
            f"WARNING: optional {label} step exited with code {status}; runtime results are retained.",
            file=sys.stderr,
        )
        return False
    return True


def validation_seed_selection(
    python: str,
    seeds: list[int],
    *,
    device: str,
    quick: bool,
    data_root: Path,
) -> int:
    candidate_root = Path(".showcase/candidates")
    candidate_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float | int]] = []
    for seed in seeds:
        report_dir = candidate_root / f"seed-{seed}"
        command = [
            python,
            "-m",
            "geo_vision.ml.train",
            "--device",
            device,
            "--seed",
            str(seed),
            "--validation-only",
            "--data-root",
            str(data_root),
            "--reports-dir",
            str(report_dir),
        ]
        if quick:
            command += [
                "--epochs",
                "3",
                "--patience",
                "2",
                "--base-channels",
                "12",
                "--batch-size",
                "4",
            ]
        run(command)
        evaluation = json.loads((report_dir / "evaluation.json").read_text())
        validation = evaluation["validation"]
        results.append(
            {
                "seed": seed,
                "macro_iou": float(validation["macro_iou"]),
                "weighted_iou": float(validation["weighted_iou"]),
                "dataset_signature": str(evaluation["dataset_signature"]),
                "experiment_signature": str(evaluation["experiment_signature"]),
                "splits": evaluation["splits"],
                "patch_counts": evaluation["patch_counts"],
            }
        )
    selected = max(results, key=lambda row: (row["macro_iou"], row["weighted_iou"]))
    selection = {
        "schema_version": "1.0",
        "selection_dataset": "validation AOIs only",
        "external_test_used_for_selection": False,
        "candidates": results,
        "selected_seed": selected["seed"],
        "selection_rule": "maximum validation macro IoU, then weighted IoU",
    }
    selection_path = (
        Path(".showcase/quick-seed-selection.json")
        if quick
        else Path("reports/landcover/seed-selection.json")
    )
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return int(selected["seed"])


def assert_selection_dataset_unchanged(
    data_root: Path,
    selection_path: Path = Path("reports/landcover/seed-selection.json"),
) -> None:
    selection = json.loads(selection_path.read_text())
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("seed-selection evidence is missing candidates")
    recorded = {row.get("dataset_signature") for row in candidates if isinstance(row, dict)}
    selection_dirs = [
        data_root / key for key in (*SHOWCASE_SPLIT.train, *SHOWCASE_SPLIT.validation)
    ]
    current = dataset_signature(selection_dirs)
    if recorded != {current}:
        raise RuntimeError(
            "train/validation data changed after seed selection; discard selection evidence and rerun"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--quick", action="store_true", help="short non-publishable integration run"
    )
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/curated"),
        help="curated raster root; Windows launcher defaults this outside synced source folders",
    )
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--seeds",
        default="20260823,20260824,20260825",
        help="comma-separated validation-selection seeds",
    )
    args = parser.parse_args()
    python = sys.executable
    try:
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    except ValueError as error:
        parser.error(f"seeds must be comma-separated integers: {error}")
    if not seeds:
        parser.error("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        parser.error("seed values must be unique")
    if args.quick:
        seeds = seeds[:1]
    elif len(seeds) < 3:
        parser.error("publication training requires at least three unique selection seeds")

    data_root = args.data_root.expanduser().resolve()
    selection_prepare = [
        python,
        "scripts/prepare_training_data.py",
        "--selection-only",
        "--output",
        str(data_root),
    ]
    if args.force_data:
        selection_prepare.append("--force")
    run(selection_prepare)

    selected_seed = validation_seed_selection(
        python,
        seeds,
        device=args.device,
        quick=args.quick,
        data_root=data_root,
    )
    if args.quick:
        # Diagnostic mode deliberately stops after validation-only training. It never loads the
        # frozen external holdout, exports a publication bundle, or mutates README/results evidence.
        pass
    else:
        # External holdout acquisition is deferred until seed selection is irreversibly recorded.
        # Train/validation scenes are hash-verified and reused; only missing external scenes download.
        run([python, "scripts/prepare_training_data.py", "--output", str(data_root)])
        assert_selection_dataset_unchanged(data_root)
        train = [
            python,
            "-m",
            "geo_vision.ml.train",
            "--device",
            args.device,
            "--seed",
            str(selected_seed),
            "--data-root",
            str(data_root),
        ]
        if args.force_train:
            train.append("--force")
        run(train)
        showcase_device = args.device if args.device in {"cpu", "cuda", "mps"} else "cpu"
        run(
            [
                python,
                "scripts/build_showcase.py",
                "--device",
                showcase_device,
                "--data-root",
                str(data_root),
            ]
        )

        # Scientific thresholds are publication metadata, not runtime prerequisites. The gate still
        # fails hard for integrity/provenance violations, while --report-only converts a clean
        # research-threshold miss into a recorded non-eligible result with exit code zero.
        run(
            [
                python,
                "scripts/release_gate.py",
                "--data-root",
                str(data_root),
                "--report-only",
            ]
        )

        # SBOM and source-build outputs are valuable release evidence but are deliberately advisory
        # in the local Windows runtime path. Strict CI/package commands enforce them separately.
        if module_available("pip_audit"):
            run_optional(
                [
                    python,
                    "-m",
                    "pip_audit",
                    "--format",
                    "cyclonedx-json",
                    "--output",
                    "reports/sbom.cdx.json",
                ],
                label="SBOM generation",
            )
        else:
            print("INFO: optional SBOM generation skipped; pip-audit is not installed.")
        if module_available("build"):
            run_optional([python, "-m", "build"], label="source/wheel build")
        else:
            print("INFO: optional package build skipped; build is not installed.")

    if args.quick:
        print("\nQuick integration run completed successfully.")
        return

    gate = json.loads(Path("reports/landcover/release-gate.json").read_text())
    if gate.get("eligible_for_github_showcase") is True:
        print(
            "\nShowcase runtime completed: real data, validation-selected trained bundle, "
            "external metrics and research-showcase eligibility are recorded."
        )
    else:
        print(
            "\nShowcase runtime completed successfully. The model is recorded as non-eligible "
            "for the optional research-showcase thresholds; this does not block local use. "
            "See reports/landcover/release-gate.json.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
