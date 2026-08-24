param(
    [ValidateSet("Default", "Cpu", "Cuda130")]
    [string]$TorchBackend = "Default",
    [switch]$ForceRecreate,
    [switch]$WithDevTools
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "windows_common.ps1")
$Root = Get-GvoRoot
Set-Location $Root

$SetupArgs = @{ TorchBackend = $TorchBackend }
if ($ForceRecreate) { $SetupArgs.ForceRecreate = $true }
if ($WithDevTools) { $SetupArgs.WithDevTools = $true }
& (Join-Path $PSScriptRoot "setup_windows.ps1") @SetupArgs

$Python = Get-GvoVenvPython -Root $Root
Ensure-GvoEditableInstall -Root $Root -Python $Python

# Runtime bootstrap intentionally avoids lint/type/coverage/audit gates. Those are engineering
# quality checks, not prerequisites for running the scientific workflow on Windows. Syntax and
# dependency consistency remain hard requirements because failures there make execution invalid.
Invoke-GvoNative $Python @("-m", "compileall", "-q", "src", "scripts")
Invoke-GvoNative $Python @("-m", "pip", "check")

Write-Host ""
Write-Host "Bootstrap complete. Runtime prerequisites are ready." -ForegroundColor Green
Write-Host "Optional engineering checks: .\scripts\quality.cmd" -ForegroundColor DarkGray
