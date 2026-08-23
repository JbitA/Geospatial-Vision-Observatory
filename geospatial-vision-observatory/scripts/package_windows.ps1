param([switch]$SkipPreflight)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "windows_common.ps1")
$Root = Get-GvoRoot
Set-Location $Root
$Python = Get-GvoVenvPython -Root $Root
if (-not (Test-Path $Python)) { throw "Environment not installed. Run .\scripts\setup.cmd first." }
Ensure-GvoEditableInstall -Root $Root -Python $Python
$PackageArgs = @("scripts\package_release.py")
if ($SkipPreflight) { $PackageArgs += "--skip-preflight" }
Invoke-GvoNative $Python $PackageArgs
