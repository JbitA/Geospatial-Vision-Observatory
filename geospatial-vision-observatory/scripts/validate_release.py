#!/usr/bin/env python3
"""Repository publication checks that do not manufacture scientific evidence.

The validator distinguishes *published source* from transient workspace state. In a Git checkout,
only tracked files are inspected for release hygiene. In an exported source archive, known runtime
and tool-cache directories are excluded so tests can run before validation without causing the
validator to reject files that the tests themselves created.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geo_vision.ml.integrity import verify_bundle  # noqa: E402

# Presentation/history checks intentionally apply only to public narrative files. Keeping the marker
# vocabulary here is an implementation detail, not part of the published project narrative.
NARRATIVE_MARKERS = (
    "previous version",
    "prior version",
    "replaces the old",
    "formerly known as",
)
NARRATIVE_NAMES = {
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CITATION.cff",
}
NARRATIVE_SUFFIXES = {".md", ".cff"}
TRANSIENT_DIRS = {
    ".git",
    ".venv",
    ".showcase",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "htmlcov",
    "dist",
    "build",
    "data",
}
TRANSIENT_NAMES = {".coverage"}
TRANSIENT_SUFFIXES = {".pyc", ".pyo", ".db", ".sqlite", ".sqlite3"}
SENSITIVE_NAMES = {".env", "id_rsa", "id_ed25519"}
SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
PRIVATE_KEY_MARKERS = (
    "-----BEGIN " + "PRIVATE KEY-----",
    "-----BEGIN OPENSSH " + "PRIVATE KEY-----",
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True)
    output = (result.stdout + result.stderr)[-8000:]
    if result.returncode:
        raise SystemExit(f"command failed: {' '.join(command)}\n{output}")
    return {"returncode": result.returncode, "output_tail": output}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _git_tracked_files(root: Path) -> list[Path] | None:
    """Return tracked files when validation runs in a Git checkout, otherwise ``None``."""

    if not (root / ".git").exists() or shutil.which("git") is None:
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        return None
    return [root / raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]


def _archive_source_files(root: Path) -> list[Path]:
    """Best-effort published-source view for a Git-less extracted archive.

    Runtime/tool outputs are ignored because there is no repository index to distinguish files that
    shipped in the archive from files created by a preceding validation command. Clean archive
    hygiene is asserted by ``scripts/package_release.py`` before distribution.
    """

    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in TRANSIENT_DIRS for part in relative.parts):
            continue
        if path.name in TRANSIENT_NAMES or path.suffix in TRANSIENT_SUFFIXES:
            continue
        files.append(path)
    return files


def _published_files(root: Path) -> tuple[list[Path], str]:
    tracked = _git_tracked_files(root)
    if tracked is not None:
        return tracked, "git-tracked"
    return _archive_source_files(root), "archive-source-view"


def _is_narrative(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        path.name in NARRATIVE_NAMES
        or (relative.parts and relative.parts[0] == "docs" and path.suffix in NARRATIVE_SUFFIXES)
        or (
            len(relative.parts) >= 2
            and relative.parts[0] == ".github"
            and path.suffix in NARRATIVE_SUFFIXES
        )
    )


def _repository_hygiene(root: Path) -> tuple[list[str], str]:
    failures: list[str] = []
    files, source_mode = _published_files(root)
    for path in files:
        try:
            relative = path.relative_to(root)
        except ValueError:
            failures.append(f"published file escaped repository root: {path}")
            continue
        if path.is_symlink():
            failures.append(f"symlinks must not be published: {relative}")
        if any(part in TRANSIENT_DIRS for part in relative.parts):
            failures.append(f"transient directory must not be published: {relative}")
        if path.name in TRANSIENT_NAMES or path.suffix in TRANSIENT_SUFFIXES:
            failures.append(f"generated/cache file must not be published: {relative}")
        if path.name in SENSITIVE_NAMES or path.suffix.lower() in SENSITIVE_SUFFIXES:
            failures.append(f"sensitive credential/key file must not be published: {relative}")
        if path.suffix.lower() in {
            ".py",
            ".ps1",
            ".cmd",
            ".md",
            ".toml",
            ".yaml",
            ".yml",
            ".txt",
            ".cff",
        }:
            try:
                text = path.read_text()
            except (OSError, UnicodeDecodeError):
                text = ""
            if any(marker in text for marker in PRIVATE_KEY_MARKERS):
                failures.append(f"private-key material marker found in {relative}")
        if _is_narrative(path, root):
            try:
                text = path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            lower = text.lower()
            for marker in NARRATIVE_MARKERS:
                if marker.lower() in lower:
                    failures.append(f"historical narrative marker {marker!r} found in {relative}")
    return failures, source_mode


def _required_files() -> Iterable[Path]:
    return (
        Path("README.md"),
        Path("LICENSE"),
        Path("SECURITY.md"),
        Path("CONTRIBUTING.md"),
        Path("CODE_OF_CONDUCT.md"),
        Path("CITATION.cff"),
        Path("AGENTS.md"),
        Path(".github/workflows/ci.yml"),
        Path(".github/workflows/codeql.yml"),
        Path(".github/dependabot.yml"),
        Path(".github/ISSUE_TEMPLATE/bug.yml"),
        Path(".github/ISSUE_TEMPLATE/research.yml"),
        Path(".github/pull_request_template.md"),
        Path("docs/ARCHITECTURE.md"),
        Path("docs/DATA_CARD.md"),
        Path("docs/SCIENTIFIC_METHOD.md"),
        Path("docs/THREAT_MODEL.md"),
        Path("docs/REPRODUCIBILITY.md"),
        Path("scripts/windows_common.ps1"),
        Path("scripts/showcase.ps1"),
        Path("scripts/showcase.cmd"),
        Path("scripts/setup_windows.ps1"),
        Path("scripts/setup.cmd"),
        Path("scripts/bootstrap.ps1"),
        Path("scripts/bootstrap.cmd"),
        Path("scripts/run_windows.ps1"),
        Path("scripts/run_windows.cmd"),
        Path("scripts/quality.ps1"),
        Path("scripts/quality.cmd"),
        Path("scripts/start_api.ps1"),
        Path("scripts/start_api.cmd"),
        Path("scripts/select_local_port.py"),
        Path("scripts/worker_once.ps1"),
        Path("scripts/worker_once.cmd"),
        Path("scripts/package_windows.ps1"),
        Path("scripts/package_windows.cmd"),
        Path("scripts/capture_environment.py"),
        Path(".gitattributes"),
        Path("scripts/prepare_training_data.py"),
        Path("scripts/release_gate.py"),
        Path("scripts/package_release.py"),
        Path("scripts/check_distribution.py"),
        Path("scripts/run_showcase_pipeline.py"),
        Path("config/showcase-policy.yaml"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="validate a source checkout that does not contain a trained model bundle",
    )
    parser.add_argument(
        "--allow-nonpublishable",
        action="store_true",
        help=(
            "validate integrity/completeness of trained evidence even when the measured research "
            "showcase gate is explicitly non-eligible"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/release-validation.json"))
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="skip pytest when a preceding coverage/preflight command already executed the suite",
    )
    args = parser.parse_args()
    if args.allow_untrained and args.allow_nonpublishable:
        parser.error("--allow-untrained and --allow-nonpublishable are mutually exclusive")
    root = Path.cwd().resolve()

    missing = [str(path) for path in _required_files() if not path.is_file()]
    if missing:
        raise SystemExit(f"release files missing: {missing}")

    hygiene, source_mode = _repository_hygiene(root)
    if hygiene:
        raise SystemExit("repository hygiene failed:\n- " + "\n- ".join(sorted(set(hygiene))))

    checks: dict[str, Any] = {
        "compile": _run([sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"]),
    }
    if args.skip_tests:
        checks["tests"] = {"status": "skipped", "reason": "preceding preflight/coverage run"}
    else:
        checks["tests"] = _run([sys.executable, "-m", "pytest", "-q"])
    if _module_available("ruff"):
        checks["ruff"] = _run([sys.executable, "-m", "ruff", "check", "."])
        checks["ruff_format"] = _run(
            [sys.executable, "-m", "ruff", "format", "--check", "."]
        )
    if _module_available("mypy"):
        checks["mypy"] = _run([sys.executable, "-m", "mypy", "src"])
    if shutil.which("node"):
        checks["dashboard_javascript"] = _run(["node", "--check", "src/geo_vision/dashboard.js"])

    readme = Path("README.md").read_text()
    bundle_dir = Path("models/landcover")
    trained = (bundle_dir / "bundle.json").is_file()
    bundle: dict[str, Any] | None = None
    if trained:
        try:
            bundle = verify_bundle(bundle_dir)
        except ValueError as error:
            raise SystemExit(f"trained bundle integrity failed: {error}") from error

    # Environment capture is runtime provenance, not trained-model evidence. Quick mode intentionally
    # refreshes it before validating an otherwise untrained checkout, so it must not trip the stale
    # model-evidence guard. A trained bundle, however, always requires a complete and internally
    # consistent evidence set; --allow-nonpublishable relaxes only the scientific eligibility bit.
    trained_evidence_paths = [
        Path("reports/landcover/seed-selection.json"),
        Path("reports/landcover/evaluation.json"),
        Path("reports/landcover/showcase-summary.json"),
        Path("reports/landcover/release-gate.json"),
        Path("reports/sbom.cdx.json"),
        Path("docs/assets/prediction-triptych.png"),
        Path("docs/assets/class-iou.svg"),
        Path("docs/assets/training-history.svg"),
    ]
    provenance_paths = [Path("reports/environment-freeze.txt")]
    gate_eligible: bool | None = None

    if not trained:
        if args.allow_nonpublishable:
            raise SystemExit("--allow-nonpublishable requires a trained model bundle")
        if not args.allow_untrained:
            raise SystemExit(
                r"trained model bundle is required; run `.\scripts\showcase.cmd` from PowerShell"
            )
        stale = [str(path) for path in trained_evidence_paths if path.is_file()]
        if stale:
            raise SystemExit(f"untrained repository contains stale trained evidence: {stale}")
        if "Measured model results are intentionally not pre-filled" not in readme:
            raise SystemExit(
                "untrained repository README must retain the explicit no-results placeholder"
            )
    else:
        required_evidence_paths = [
            *trained_evidence_paths,
            *provenance_paths,
            Path("docs/RESULTS.md"),
        ]
        absent = [str(path) for path in required_evidence_paths if not path.is_file()]
        if absent:
            raise SystemExit(f"trained evidence is missing: {absent}")

        gate = _json(Path("reports/landcover/release-gate.json"))
        eligible_raw = gate.get("eligible_for_github_showcase")
        gate_failures = gate.get("failures")
        if not isinstance(eligible_raw, bool) or not isinstance(gate_failures, list):
            raise SystemExit("research showcase release gate has an invalid eligibility schema")
        gate_eligible = eligible_raw
        runtime_integrity_ok = gate.get("runtime_integrity_ok", True)
        integrity_failures = gate.get("integrity_failures", [])
        if runtime_integrity_ok is not True or not isinstance(integrity_failures, list):
            raise SystemExit("trained research gate reports invalid runtime integrity")
        if integrity_failures:
            raise SystemExit("trained research gate contains integrity failures")
        if gate_eligible and gate_failures:
            raise SystemExit("eligible research showcase gate must not contain failures")
        if not gate_eligible and not gate_failures:
            raise SystemExit("non-eligible research showcase gate must explain its failures")
        if not gate_eligible and not args.allow_nonpublishable:
            raise SystemExit("research showcase release gate is not eligible")

        summary = _json(Path("reports/landcover/showcase-summary.json"))
        evaluation = _json(Path("reports/landcover/evaluation.json"))
        assert bundle is not None
        model_file = str(bundle["model_file"])
        model_sha = bundle["files"][model_file]["sha256"]
        for source_name, source in {
            "summary": summary,
            "evaluation": evaluation,
            "gate": gate,
        }.items():
            if source.get("dataset_signature") != bundle.get("dataset_signature"):
                raise SystemExit(f"{source_name}/bundle dataset signature mismatch")
        if summary.get("experiment_signature") != bundle.get("experiment_signature"):
            raise SystemExit("summary/bundle experiment signature mismatch")
        if evaluation.get("experiment_signature") != bundle.get("experiment_signature"):
            raise SystemExit("evaluation/bundle experiment signature mismatch")
        if summary.get("model_bundle_sha256") != model_sha or gate.get("model_sha256") != model_sha:
            raise SystemExit("trained evidence/model SHA-256 mismatch")
        if "## Measured model result" not in readme:
            raise SystemExit("README has not been generated with measured model evidence")

    payload = {
        "schema_version": "1.2",
        "status": "passed",
        "trained_bundle_present": trained,
        "trained_evidence_required": trained,
        "research_showcase_eligible": gate_eligible,
        "nonpublishable_evidence_allowed": bool(trained and args.allow_nonpublishable),
        "bundle_dataset_signature": bundle.get("dataset_signature") if bundle else None,
        "primary_platform": "Windows 10/11 x64",
        "python_contract": ">=3.12,<3.13",
        "windows_launchers_present": True,
        "environment_freeze_present": Path("reports/environment-freeze.txt").is_file(),
        "checks": checks,
        "repository_hygiene": "passed",
        "repository_hygiene_source": source_mode,
        "claim_policy": (
            "No metric is accepted unless generated from the verified real-data training/evaluation "
            "pipeline."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
