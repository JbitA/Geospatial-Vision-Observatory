param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [ValidateSet("Default", "Cpu", "Cuda130")]
    [string]$TorchBackend = "Default",
    [switch]$Quick,
    [switch]$ForceData,
    [switch]$ForceTrain,
    [switch]$WithQuality,
    [switch]$NoApi,
    [ValidateRange(0, 65535)]
    [int]$Port = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "windows_common.ps1")
$Root = Get-GvoRoot
Set-Location $Root

Write-Host "Geospatial Vision Observatory - Windows one-command run" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "bootstrap.ps1") -TorchBackend $TorchBackend

$ShowcaseArgs = @{ Device = $Device; SkipSetup = $true }
if ($Quick) { $ShowcaseArgs.Quick = $true }
if ($ForceData) { $ShowcaseArgs.ForceData = $true }
if ($ForceTrain) { $ShowcaseArgs.ForceTrain = $true }
& (Join-Path $PSScriptRoot "showcase.ps1") @ShowcaseArgs

if ($WithQuality) {
    Write-Host "Running advisory engineering quality report..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "quality.ps1")
}

if ($NoApi) {
    Write-Host "Windows run completed. API start skipped by -NoApi." -ForegroundColor Green
    exit 0
}

Write-Host "Starting local API. Press Ctrl+C to stop it." -ForegroundColor Green
& (Join-Path $PSScriptRoot "start_api.ps1") -Port $Port
exit $LASTEXITCODE
