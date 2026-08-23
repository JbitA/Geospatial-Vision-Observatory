#!/usr/bin/env python3
"""Fail closed unless measured showcase evidence satisfies declared scientific/integrity gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from geo_vision.ml.data import dataset_signature
from geo_vision.ml.integrity import SHA256_PATTERN, verify_bundle
from geo_vision.ml.schema import CLASS_NAMES, SHOWCASE_SPLIT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _expected_splits() -> dict[str, list[str]]:
    return {
        "train": list(SHOWCASE_SPLIT.train),
        "validation": list(SHOWCASE_SPLIT.validation),
        "external_test": list(SHOWCASE_SPLIT.external_test),
    }


def _verify_curated_split(
    data_root: Path, dataset_policy: dict[str, Any], failures: list[str]
) -> None:
    split_path = data_root / "training-split.json"
    if not split_path.is_file():
        failures.append("training-split.json is missing")
        return
    try:
        split = _json(split_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"training-split.json is invalid: {error}")
        return

    expected = _expected_splits()
    expected_assignment = {
        key: split_name for split_name, keys in expected.items() for key in keys
    }
    if split.get("schema_version") != "1.0":
        failures.append("training-split schema is unsupported")
    if split.get("temporal_window") != dataset_policy.get("temporal_window"):
        failures.append("curation temporal window differs from showcase policy")
    split_cloud = _finite_number(split.get("max_cloud_percent"))
    policy_cloud = _finite_number(dataset_policy.get("max_cloud_percent"))
    if split_cloud is None or policy_cloud is None or split_cloud != policy_cloud:
        failures.append("curation cloud threshold differs from showcase policy")
    if split.get("cloud_quality_basis") != dataset_policy.get("cloud_quality_basis"):
        failures.append("curation cloud-quality basis differs from showcase policy")
    if split.get("max_pixels") != dataset_policy.get("max_pixels"):
        failures.append("curation pixel bound differs from showcase policy")

    rows = split.get("assignment")
    if not isinstance(rows, list):
        failures.append("training-split assignment is invalid")
        return
    observed: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("aoi"), str):
            failures.append("training-split contains an invalid assignment row")
            continue
        key = row["aoi"]
        if key in observed:
            failures.append(f"training-split assigns AOI more than once: {key}")
            continue
        split_name = row.get("split")
        if not isinstance(split_name, str):
            failures.append(f"training-split has no split for {key}")
            continue
        observed[key] = split_name
        scene_manifest_path = data_root / key / "manifest.json"
        if not scene_manifest_path.is_file():
            failures.append(f"curated manifest missing: {key}")
            continue
        actual_manifest_sha = _sha256(scene_manifest_path)
        if row.get("manifest_sha256") != actual_manifest_sha:
            failures.append(f"training-split manifest hash mismatch: {key}")
            continue
        try:
            scene = _json(scene_manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"curated manifest invalid for {key}: {error}")
            continue
        if scene.get("aoi", {}).get("key") != key:
            failures.append(f"curated AOI identity mismatch: {key}")
        sentinel = scene.get("sentinel2", {})
        if not isinstance(sentinel, dict):
            failures.append(f"Sentinel metadata invalid: {key}")
            continue
        if sentinel.get("collection") != "sentinel-2-c1-l2a":
            failures.append(f"unexpected Sentinel collection: {key}")
        requested_start = str(sentinel.get("requested_start", ""))[:10]
        requested_end = str(sentinel.get("requested_end", ""))[:10]
        temporal = dataset_policy.get("temporal_window", [])
        if len(temporal) == 2 and [requested_start, requested_end] != list(temporal):
            failures.append(f"Sentinel request window differs from policy: {key}")
        if sentinel.get("cloud_threshold_required") is not True:
            failures.append(f"Sentinel scene was not curated in strict cloud-threshold mode: {key}")
        cloud = _finite_number(sentinel.get("cloud_cover"))
        maximum_cloud = _finite_number(dataset_policy.get("max_cloud_percent"))
        aoi_obscured = _finite_number(sentinel.get("aoi_scl_obscured_percent"))
        if cloud is None or cloud < 0.0 or cloud > 100.0:
            failures.append(f"Sentinel cloud-cover metadata invalid: {key}")
        if sentinel.get("cloud_quality_basis") != "aoi_scl_obscured_percent":
            failures.append(f"Sentinel AOI cloud-quality basis invalid: {key}")
        if aoi_obscured is None or aoi_obscured < 0.0 or aoi_obscured > 100.0:
            failures.append(f"Sentinel AOI SCL obscured fraction invalid: {key}")
        elif maximum_cloud is None or aoi_obscured > maximum_cloud:
            failures.append(
                f"Sentinel AOI exceeds obscured-pixel threshold: {key} ({aoi_obscured}%)"
            )
        integrity = scene.get("integrity", {})
        if not isinstance(integrity, dict) or integrity.get("max_pixels") != dataset_policy.get(
            "max_pixels"
        ):
            failures.append(f"curated pixel bound differs from policy: {key}")
            continue
        outputs = integrity.get("outputs")
        if not isinstance(outputs, dict):
            failures.append(f"curated output hash set is missing: {key}")
            continue
        required_outputs = {"sentinel2_multispectral.tif", "worldcover_2021_on_sentinel.tif"}
        if not required_outputs.issubset(outputs):
            failures.append(f"curated required outputs are missing from manifest: {key}")
        for raw_name, metadata in outputs.items():
            if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
                failures.append(f"unsafe curated output path in {key}")
                continue
            if not isinstance(metadata, dict):
                failures.append(f"invalid curated output metadata in {key}: {raw_name}")
                continue
            path = data_root / key / raw_name
            expected_sha = metadata.get("sha256")
            expected_bytes = metadata.get("bytes")
            if (
                not path.is_file()
                or path.is_symlink()
                or not isinstance(expected_sha, str)
                or SHA256_PATTERN.fullmatch(expected_sha) is None
                or not isinstance(expected_bytes, int)
                or isinstance(expected_bytes, bool)
                or path.stat().st_size != expected_bytes
                or _sha256(path) != expected_sha
            ):
                failures.append(f"curated output integrity mismatch: {key}/{raw_name}")

    if observed != expected_assignment:
        failures.append("training-split AOI assignment differs from the frozen showcase split")


def _verify_seed_selection(
    report: dict[str, Any],
    policy: dict[str, Any],
    failures: list[str],
    *,
    expected_dataset_signature: str | None = None,
) -> None:
    if report.get("schema_version") != "1.0":
        failures.append("seed-selection schema is unsupported")
    if report.get("selection_dataset") != "validation AOIs only":
        failures.append("seed-selection evidence has an unexpected selection dataset")
    if report.get("selection_rule") != "maximum validation macro IoU, then weighted IoU":
        failures.append("seed-selection evidence has an unexpected selection rule")
    if report.get("external_test_used_for_selection") is not False:
        failures.append("seed-selection evidence does not prove external holdout isolation")
    candidates = report.get("candidates")
    minimum = int(policy.get("minimum_validation_seeds", 3))
    if not isinstance(candidates, list) or len(candidates) < minimum:
        failures.append(f"seed selection requires at least {minimum} validation-only candidates")
        return
    expected_selection_splits = {
        "train": list(SHOWCASE_SPLIT.train),
        "validation": list(SHOWCASE_SPLIT.validation),
    }
    seeds: set[int] = set()
    valid_rows: list[dict[str, Any]] = []
    dataset_signatures: set[str] = set()
    experiment_signatures: set[str] = set()
    for row in candidates:
        if not isinstance(row, dict):
            failures.append("seed-selection candidate row is invalid")
            continue
        seed = row.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed in seeds:
            failures.append("seed-selection candidates must have unique integer seeds")
            continue
        seeds.add(seed)
        row_valid = True
        if row.get("splits") != expected_selection_splits:
            failures.append(f"seed {seed} did not use validation-only split evidence")
            row_valid = False
        patch_counts = row.get("patch_counts")
        if not isinstance(patch_counts, dict):
            failures.append(f"seed {seed} patch-count evidence is invalid")
            row_valid = False
        elif "external_test" in patch_counts:
            failures.append(f"seed {seed} candidate touched external-test patch evaluation")
            row_valid = False
        signature = row.get("dataset_signature")
        if not isinstance(signature, str) or SHA256_PATTERN.fullmatch(signature) is None:
            failures.append(f"seed {seed} has invalid dataset signature")
            row_valid = False
        else:
            dataset_signatures.add(signature)
        experiment = row.get("experiment_signature")
        if not isinstance(experiment, str) or SHA256_PATTERN.fullmatch(experiment) is None:
            failures.append(f"seed {seed} has invalid experiment signature")
            row_valid = False
        elif experiment in experiment_signatures:
            failures.append("seed-selection candidates must have distinct experiment signatures")
            row_valid = False
        else:
            experiment_signatures.add(experiment)
        for metric in ("macro_iou", "weighted_iou"):
            value = _finite_number(row.get(metric))
            if value is None or not 0.0 <= value <= 1.0:
                failures.append(f"seed {seed} has invalid validation metric: {metric}")
                row_valid = False
        if row_valid:
            valid_rows.append(row)
    if len(dataset_signatures) != 1:
        failures.append("seed candidates were not evaluated against one immutable selection dataset")
    elif expected_dataset_signature is not None and dataset_signatures != {expected_dataset_signature}:
        failures.append("seed-selection dataset no longer matches current train/validation data")
    if not valid_rows:
        return
    mathematically_selected = max(
        valid_rows,
        key=lambda row: (float(row["macro_iou"]), float(row["weighted_iou"])),
    ).get("seed")
    if report.get("selected_seed") != mathematically_selected:
        failures.append("recorded selected seed is not the declared validation-metric winner")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, default=Path("reports/landcover/evaluation.json"))
    parser.add_argument("--bundle", type=Path, default=Path("models/landcover"))
    parser.add_argument("--data-root", type=Path, default=Path("data/curated"))
    parser.add_argument("--policy", type=Path, default=Path("config/showcase-policy.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/landcover/release-gate.json"))
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="return success for research-threshold misses while still failing on integrity errors",
    )
    args = parser.parse_args()

    evaluation = _json(args.evaluation)
    bundle = verify_bundle(args.bundle)
    policy = yaml.safe_load(args.policy.read_text())
    if not isinstance(policy, dict) or policy.get("schema_version") != "1.0":
        raise SystemExit("showcase policy is invalid")
    integrity_failures: list[str] = []
    research_failures: list[str] = []

    dataset_policy = policy.get("dataset_policy", {})
    if not isinstance(dataset_policy, dict):
        raise SystemExit("showcase dataset policy is invalid")
    _verify_curated_split(args.data_root, dataset_policy, integrity_failures)

    seed_selection_path = Path("reports/landcover/seed-selection.json")
    if not seed_selection_path.is_file():
        integrity_failures.append("validation-only seed-selection evidence is missing")
        seed_selection: dict[str, Any] = {}
    else:
        seed_selection = _json(seed_selection_path)
        selection_signature = dataset_signature(
            [
                args.data_root / key
                for key in (*SHOWCASE_SPLIT.train, *SHOWCASE_SPLIT.validation)
            ]
        )
        _verify_seed_selection(
            seed_selection,
            dataset_policy,
            integrity_failures,
            expected_dataset_signature=selection_signature,
        )
        if seed_selection.get("selected_seed") != evaluation.get("config", {}).get("seed"):
            integrity_failures.append("final model seed differs from the validation-selected seed")

    expected_splits = _expected_splits()
    if evaluation.get("splits") != expected_splits:
        integrity_failures.append("final spatial split differs from the frozen showcase split")
    all_keys = [*SHOWCASE_SPLIT.train, *SHOWCASE_SPLIT.validation, *SHOWCASE_SPLIT.external_test]
    current_signature = dataset_signature([args.data_root / key for key in all_keys])
    if evaluation.get("dataset_signature") != current_signature:
        integrity_failures.append("evaluation dataset signature does not match current curated data")
    if evaluation.get("dataset_signature") != bundle.get("dataset_signature"):
        integrity_failures.append("evaluation/bundle dataset signature mismatch")
    if evaluation.get("experiment_signature") != bundle.get("experiment_signature"):
        integrity_failures.append("evaluation/bundle experiment signature mismatch")

    ext = evaluation.get("external_test")
    if not isinstance(ext, dict):
        integrity_failures.append("external-test metrics are missing")
        ext = {}
    gates = policy.get("research_showcase_gates")
    if not isinstance(gates, dict):
        raise SystemExit("showcase research gates are invalid")
    for key, minimum in gates.get("minimum", {}).items():
        value = _finite_number(ext.get(key))
        required = _finite_number(minimum)
        if value is None or required is None:
            integrity_failures.append(f"external-test metric is invalid: {key}")
        elif value < required:
            research_failures.append(f"{key}={value} is below required {minimum}")
    for key, maximum in gates.get("maximum", {}).items():
        value = _finite_number(ext.get(key))
        allowed = _finite_number(maximum)
        if value is None or allowed is None:
            integrity_failures.append(f"external-test metric is invalid: {key}")
        elif value > allowed:
            research_failures.append(f"{key}={value} exceeds allowed {maximum}")
    iou = ext.get("iou")
    if not isinstance(iou, list) or len(iou) != len(CLASS_NAMES):
        integrity_failures.append("external-test class-IoU vector is invalid")
        per_class: dict[str, float | None] = {}
    else:
        per_class = dict(zip(CLASS_NAMES, iou, strict=True))
    for name, minimum in gates.get("class_iou_minimum", {}).items():
        value = _finite_number(per_class.get(name))
        required = _finite_number(minimum)
        if value is None or required is None:
            integrity_failures.append(f"external-test class IoU is invalid: {name}")
        elif value < required:
            research_failures.append(f"{name}_iou={value} is below required {minimum}")

    summary_path = Path("reports/landcover/showcase-summary.json")
    if not summary_path.is_file():
        integrity_failures.append("generated showcase summary is missing")
    else:
        summary = _json(summary_path)
        model_sha = bundle["files"][bundle["model_file"]]["sha256"]
        if summary.get("model_bundle_sha256") != model_sha:
            integrity_failures.append("showcase summary/model artifact hash mismatch")
        if summary.get("dataset_signature") != current_signature:
            integrity_failures.append("showcase summary/data signature mismatch")
        if summary.get("experiment_signature") != evaluation.get("experiment_signature"):
            integrity_failures.append("showcase summary/experiment signature mismatch")

    failures = [*integrity_failures, *research_failures]
    model_sha = bundle["files"][bundle["model_file"]]["sha256"]
    payload = {
        "schema_version": "1.1",
        "eligible_for_github_showcase": not failures,
        "runtime_integrity_ok": not integrity_failures,
        "integrity_failures": integrity_failures,
        "research_threshold_failures": research_failures,
        "failures": failures,
        "policy": policy,
        "dataset_signature": evaluation.get("dataset_signature"),
        "experiment_signature": evaluation.get("experiment_signature"),
        "model_sha256": model_sha,
        "important_boundary": (
            "Passing this research showcase gate is not production authorization; production "
            "deployment remains deny-by-default and requires independent review."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if integrity_failures:
        raise SystemExit(1)
    if research_failures and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
