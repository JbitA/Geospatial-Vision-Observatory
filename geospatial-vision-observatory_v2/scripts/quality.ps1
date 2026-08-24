param(
    [switch]$Strict,
    [switch]$AllowUntrained,
    [switch]$RequirePublishable
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "windows_common.ps1")
$Root = Get-GvoRoot
Set-Location $Root
$Python = Get-GvoVenvPython -Root $Root
if (-not (Test-Path $Python)) { throw "Virtual environment not found. Run .\scripts\bootstrap.cmd first." }
Ensure-GvoEditableInstall -Root $Root -Python $Python

$Failures = New-Object System.Collections.Generic.List[string]

function Invoke-Advisory {
    param([string]$Label, [string]$FilePath, [string[]]$Arguments)
    Write-Host "+ $FilePath $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $FilePath @Arguments
    $Code = $LASTEXITCODE
    if ($Code -ne 0) {
        $Failures.Add("$Label (exit $Code)")
        Write-Warning "$Label did not pass. Continuing because local quality checks are advisory by default."
    }
}

# Install development-only quality tools when possible, but do not make local runtime usability
# depend on that installation. Strict CI/package workflows install these extras separately.
Write-Host "Preparing optional development quality tools..." -ForegroundColor Cyan
& $Python -m pip install --prefer-binary -e ".[ml,dev]"
if ($LASTEXITCODE -ne 0) {
    $Failures.Add("development-tool installation")
    Write-Warning "Development tools could not be installed; available checks will still run."
}

$Checks = @(
    @{ Label = "Ruff lint"; Args = @("-m", "ruff", "check", ".") },
    @{ Label = "Ruff format"; Args = @("-m", "ruff", "format", "--check", ".") },
    @{ Label = "mypy"; Args = @("-m", "mypy", "src") },
    @{ Label = "tests"; Args = @("-m", "coverage", "run", "--branch", "-m", "pytest", "-q") },
    @{ Label = "coverage"; Args = @("-m", "coverage", "report", "--fail-under=80") },
    @{ Label = "compile"; Args = @("-m", "compileall", "-q", "src", "scripts", "tests") },
    @{ Label = "pip check"; Args = @("-m", "pip", "check") },
    @{ Label = "dependency audit"; Args = @("-m", "pip_audit") }
)

& $Python -m coverage erase *> $null
foreach ($Check in $Checks) {
    Invoke-Advisory -Label $Check.Label -FilePath $Python -Arguments $Check.Args
}

$Node = Get-Command node.exe -ErrorAction SilentlyContinue
if ($null -ne $Node) {
    Invoke-Advisory -Label "dashboard JavaScript" -FilePath $Node.Source -Arguments @("--check", "src\geo_vision\dashboard.js")
}
else {
    Write-Host "Node.js not found; dashboard JavaScript syntax check skipped locally." -ForegroundColor Yellow
}

$ValidationArgs = @("scripts\validate_release.py", "--skip-tests")
$BundlePath = Join-Path $Root "models\landcover\bundle.json"
if ($AllowUntrained -or -not (Test-Path $BundlePath)) {
    $ValidationArgs += "--allow-untrained"
}
elseif (-not $RequirePublishable) {
    $ValidationArgs += "--allow-nonpublishable"
}
Invoke-Advisory -Label "release validation" -FilePath $Python -Arguments $ValidationArgs

Write-Host ""
if ($Failures.Count -eq 0) {
    Write-Host "Quality report: all available checks passed." -ForegroundColor Green
    exit 0
}

Write-Warning ("Quality report completed with advisory findings: " + ($Failures -join "; "))
if ($Strict) {
    throw "Strict quality mode failed: $($Failures -join '; ')"
}
Write-Host "Runtime remains usable. Use .\scripts\quality.cmd -Strict for CI-style enforcement." -ForegroundColor Yellow
exit 0
