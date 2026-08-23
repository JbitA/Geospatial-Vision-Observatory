param(
    [ValidateRange(0, 65535)]
    [int]$Port = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "windows_common.ps1")
$Root = Get-GvoRoot
Set-Location $Root
$Python = Get-GvoVenvPython -Root $Root
if (-not (Test-Path $Python)) { throw "Environment not installed. Run .\scripts\setup.cmd first." }
Ensure-GvoEditableInstall -Root $Root -Python $Python
$Executable = Join-Path (Get-GvoVenvRoot -Root $Root) "Scripts\geospatial-vision-api.exe"
if (-not (Test-Path $Executable)) { throw "Environment not installed. Run .\scripts\setup.cmd first." }
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
Set-GvoNativeRuntimeStorage -Root $Root

$PreferredPort = if ($Port -gt 0) { $Port } else { 8080 }
$PortArgs = @("scripts\select_local_port.py", "--preferred", "$PreferredPort")
if ($Port -gt 0) { $PortArgs += "--no-fallback" }
Write-Host "+ $Python $($PortArgs -join ' ')" -ForegroundColor DarkGray
$SelectedOutput = & $Python @PortArgs
if ($LASTEXITCODE -ne 0) {
    throw "No usable local API port was found. Pass -Port <port> to choose another loopback port."
}
$SelectedPort = [int](($SelectedOutput | Select-Object -Last 1).Trim())
$env:VISION_BIND_HOST = "127.0.0.1"
$env:VISION_BIND_PORT = "$SelectedPort"
Write-Host "API URL: http://127.0.0.1:$SelectedPort" -ForegroundColor Green
& $Executable
exit $LASTEXITCODE
