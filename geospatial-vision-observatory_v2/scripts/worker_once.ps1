Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "windows_common.ps1")
$Root = Get-GvoRoot
Set-Location $Root
$Python = Get-GvoVenvPython -Root $Root
if (-not (Test-Path $Python)) { throw "Environment not installed. Run .\scripts\setup.cmd first." }
Ensure-GvoEditableInstall -Root $Root -Python $Python
$Executable = Join-Path (Get-GvoVenvRoot -Root $Root) "Scripts\geospatial-vision-worker.exe"
if (-not (Test-Path $Executable)) { throw "Environment not installed. Run .\scripts\setup.cmd first." }
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
Set-GvoNativeRuntimeStorage -Root $Root
& $Executable --once
exit $LASTEXITCODE
