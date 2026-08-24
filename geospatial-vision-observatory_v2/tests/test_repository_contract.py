from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_keeps_runtime_on_least_privilege_boundaries() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    services = compose["services"]
    assert services["api"]["ports"] == ["127.0.0.1:8080:8080"]
    assert services["worker"]["entrypoint"] == ["geospatial-vision-worker"]
    for name in ("api", "worker"):
        service = services[name]
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in service["security_opt"]
        assert any("noexec" in entry and "nosuid" in entry for entry in service["tmpfs"])
        assert service["environment"]["VISION_DATA_DIR"] == "/data"


def test_ci_contract_is_windows_first_with_optional_container_scan() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert "runs-on: windows-latest" in workflow
    assert "Windows quality" in workflow
    assert "Windows package" in workflow
    assert ".\\scripts\\showcase.ps1 -Device cpu" in workflow
    assert "coverage report --fail-under=80" in workflow
    assert "python scripts/validate_release.py --allow-untrained --skip-tests" in workflow
    assert "python -m pip check" in workflow
    assert "docker compose config --quiet" in workflow
    assert "trivy" in workflow.lower()
    assert "if: github.event_name == 'workflow_dispatch'" in workflow


def test_windows_launchers_are_native_and_do_not_require_bash() -> None:
    readme = (ROOT / "README.md").read_text()
    setup = (ROOT / "scripts/setup_windows.ps1").read_text()
    showcase = (ROOT / "scripts/showcase.ps1").read_text()
    assert "No WSL, Bash, or Git Bash is required" in readme
    assert ".\\scripts\\showcase.cmd" in readme
    assert "%USERPROFILE%\\.gvo\\venv-py314" in readme
    assert (ROOT / "scripts/windows_common.ps1").is_file()
    assert (ROOT / "scripts/run_windows.cmd").is_file()
    assert (ROOT / "scripts/run_windows.ps1").is_file()
    common = (ROOT / "scripts/windows_common.ps1").read_text()
    assert "venv-py314" in common
    assert '"-3.14"' in setup
    assert "$env:USERPROFILE" in common
    assert "Get-GvoUserStateRoot" in common
    assert "$env:GVO_STATE_ROOT" in common
    assert '"--data-root", $DataRoot' in showcase
    assert ".venv/bin" not in readme
    assert "bash scripts/" not in readme
    assert not (ROOT / "Makefile").exists()
    assert not (ROOT / "scripts/showcase.sh").exists()


def test_public_workflow_preserves_holdout_isolation() -> None:
    pipeline = (ROOT / "scripts/run_showcase_pipeline.py").read_text()
    assert '"--selection-only"' in pipeline
    assert "assert_selection_dataset_unchanged(data_root)" in pipeline
    assert "record_external_evaluation_exposure()" in pipeline
    assert pipeline.index("validation_seed_selection(") < pipeline.rindex(
        'run([python, "scripts/prepare_training_data.py", "--output", str(data_root)])'
    )
    assert ".showcase/candidates" in pipeline
    assert '"--data-root"' in pipeline
    assert ".codex/" not in pipeline


def test_container_repository_and_windows_eol_contracts() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "USER 10001:10001" in dockerfile
    assert 'ENTRYPOINT ["geospatial-vision-api"]' in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "python:3.14.7-slim-bookworm" in dockerfile
    assert "pip install --upgrade pip" not in dockerfile

    gitignore = (ROOT / ".gitignore").read_text()
    for required in (".env", ".venv/", "data/", "*.db", ".coverage", ".showcase/"):
        assert required in gitignore

    attributes = (ROOT / ".gitattributes").read_text()
    assert "*.ps1 text eol=crlf" in attributes
    assert "*.cmd text eol=crlf" in attributes
    assert "*.py text eol=lf" in attributes

def test_windows_powershell_scripts_are_fail_fast_and_avoid_drive_qualified_variable_trap() -> None:
    entrypoints = (
        "setup_windows.ps1",
        "showcase.ps1",
        "quality.ps1",
        "start_api.ps1",
        "worker_once.ps1",
        "package_windows.ps1",
    )
    scripts = ROOT / "scripts"
    common = (scripts / "windows_common.ps1").read_text()
    assert "$LASTEXITCODE:" not in common
    assert "${LASTEXITCODE}:" in common

    for name in entrypoints:
        text = (scripts / name).read_text()
        assert "Set-StrictMode -Version Latest" in text
        assert '$ErrorActionPreference = "Stop"' in text
        common_import = '. (Join-Path $PSScriptRoot "windows_common.ps1")'
        assert text.index('$ErrorActionPreference = "Stop"') < text.index(common_import)


def test_windows_package_script_does_not_shadow_automatic_args_variable() -> None:
    text = (ROOT / "scripts/package_windows.ps1").read_text()
    assert "$PackageArgs" in text
    assert "$Args =" not in text


def test_quick_showcase_uses_valid_early_stopping_configuration() -> None:
    pipeline = (ROOT / "scripts/run_showcase_pipeline.py").read_text()
    trainer = (ROOT / "src/geo_vision/ml/train.py").read_text()
    assert '"--epochs",\n                "3",' in pipeline
    assert '"--patience",\n                "2",' in pipeline
    assert 'parser.add_argument("--patience", type=int, default=4)' in trainer
    assert 'patience=args.patience' in trainer


def test_windows_runtime_path_does_not_block_on_engineering_quality_checks() -> None:
    pipeline = (ROOT / "scripts/run_showcase_pipeline.py").read_text()
    bootstrap = (ROOT / "scripts/bootstrap.ps1").read_text()
    quality = (ROOT / "scripts/quality.ps1").read_text()
    onego = (ROOT / "scripts/run_windows.ps1").read_text()

    assert "run_quality_preflight" not in pipeline
    assert '"-m", "ruff"' not in bootstrap
    assert '"-m", "mypy"' not in bootstrap
    assert "$Strict" in quality
    assert "local quality checks are advisory by default" in quality
    assert 'Join-Path $PSScriptRoot "bootstrap.ps1"' in onego
    assert 'Join-Path $PSScriptRoot "showcase.ps1"' in onego
    assert 'Join-Path $PSScriptRoot "start_api.ps1"' in onego



def test_untrained_release_validation_allows_environment_provenance() -> None:
    validator = (ROOT / "scripts/validate_release.py").read_text()
    assert 'provenance_paths = [Path("reports/environment-freeze.txt")]' in validator
    assert 'stale = [str(path) for path in trained_evidence_paths if path.is_file()]' in validator
    stale_line = next(line for line in validator.splitlines() if line.strip().startswith("stale ="))
    assert "provenance_paths" not in stale_line


def test_nonpublishable_training_is_reported_without_masking_integrity_failures() -> None:
    pipeline = (ROOT / "scripts/run_showcase_pipeline.py").read_text()
    gate = (ROOT / "scripts/release_gate.py").read_text()
    quality = (ROOT / "scripts/quality.ps1").read_text()

    assert '"--report-only"' in pipeline
    assert 'integrity_failures' in gate
    assert 'research_failures' in gate
    assert 'if integrity_failures:' in gate
    assert 'raise SystemExit(1)' in gate
    assert 'if research_failures and not args.report_only:' in gate
    assert 'raise SystemExit(2)' in gate
    assert "run_optional(" in pipeline
    assert 'trained research gate contains integrity failures' in (ROOT / "scripts/validate_release.py").read_text()
    assert '$RequirePublishable' in quality


def test_native_api_launcher_uses_loopback_port_probe_and_fallback() -> None:
    launcher = (ROOT / "scripts/start_api.ps1").read_text()
    helper = (ROOT / "scripts/select_local_port.py").read_text()
    validator = (ROOT / "scripts/validate_release.py").read_text()

    assert 'scripts\\select_local_port.py' in launcher
    assert '$env:VISION_BIND_HOST = "127.0.0.1"' in launcher
    assert '$env:VISION_BIND_PORT = "$SelectedPort"' in launcher
    assert 'API URL: http://127.0.0.1:$SelectedPort' in launcher
    assert 'DEFAULT_FALLBACKS = (8765, 8888, 8000, 8899)' in helper
    assert 'Path("scripts/select_local_port.py")' in validator


def test_release_metadata_scaled_entrypoint_and_lazy_ml_contract_are_consistent() -> None:
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    namespace: dict[str, object] = {}
    exec((ROOT / "src/geo_vision/__init__.py").read_text(), namespace)
    assert project["project"]["version"] == "2.0.0"
    assert citation["version"] == "2.0.0"
    assert namespace["__version__"] == "2.0.0"
    assert (ROOT / "scripts/prepare_processing_cells.py").is_file()
    assert (ROOT / "docs/SCALING_ROADMAP.md").is_file()
    assert (ROOT / "docs/LAZY_RASTER_IO.md").is_file()
    assert "pyproj==3.7.2" in project["project"]["dependencies"]
    training = (ROOT / "src/geo_vision/ml/train.py").read_text()
    assert "load_lazy_scene" in training


def test_v2_scaled_cli_and_source_manifest_are_directly_executable() -> None:
    help_result = subprocess.run(
        [sys.executable, "scripts/prepare_processing_cells.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "deterministic AOI v2 processing-cell artifacts" in help_result.stdout

    manifest_result = subprocess.run(
        [sys.executable, "scripts/source_manifest.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert manifest_result.returncode == 0, manifest_result.stderr
    assert "source manifest verified" in manifest_result.stdout
