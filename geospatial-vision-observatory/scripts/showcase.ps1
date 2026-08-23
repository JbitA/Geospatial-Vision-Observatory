param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [string]$Seeds = "20260823,20260824,20260825",
    [ValidateSet("Default", "Cpu", "Cuda130")]
    [string]$TorchBackend = "Default",
    [switch]$Quick,
    [switch]$ForceData,
    [switch]$ForceTrain,
    [string]$DataRoot = "",
    [switch]$SkipSetup
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"


. (Join-Path $PSScriptRoot "windows_common.ps1")
$Root = Get-GvoRoot
Set-Location $Root
$VenvPython = Get-GvoVenvPython -Root $Root

if ((-not $SkipSetup) -and (-not (Test-Path $VenvPython))) {
    & (Join-Path $PSScriptRoot "setup_windows.ps1") -TorchBackend $TorchBackend
}
elseif (-not (Test-Path $VenvPython)) {
    throw "Virtual environment not found. Run .\scripts\setup.cmd first."
}

Ensure-GvoEditableInstall -Root $Root -Python $VenvPython

$env:PYTHONHASHSEED = "0"
$DataRoot = Get-GvoCuratedRoot -Root $Root -Requested $DataRoot
$StateRoot = Get-GvoUserStateRoot
Write-Host "Checking curated cache accessibility..." -ForegroundColor DarkGray
$ResolvedRootOutput = & $VenvPython "scripts\resolve_data_root.py" "--primary" $DataRoot "--state-root" $StateRoot
if ($LASTEXITCODE -ne 0) {
    throw "Could not select a readable/writable curated cache root."
}
$DataRoot = [string]($ResolvedRootOutput | Select-Object -Last 1)
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    throw "Curated cache resolver returned an empty path."
}
$DataRoot = $DataRoot.Trim()
Write-Host "Curated data root: $DataRoot"
Invoke-GvoNative $VenvPython @("scripts\capture_environment.py")

$PipelineArgs = @(
    "scripts\run_showcase_pipeline.py",
    "--device", $Device,
    "--seeds", $Seeds,
    "--data-root", $DataRoot
)
if ($Quick) { $PipelineArgs += "--quick" }
if ($ForceData) { $PipelineArgs += "--force-data" }
if ($ForceTrain) { $PipelineArgs += "--force-train" }

Write-Host "+ $VenvPython $($PipelineArgs -join ' ')" -ForegroundColor DarkGray
& $VenvPython @PipelineArgs
$PipelineExitCode = $LASTEXITCODE
if ($PipelineExitCode -eq 0) {
    exit 0
}
throw "Showcase runtime failed with exit code ${PipelineExitCode}. See the preceding error; research-threshold misses do not produce a runtime failure."
