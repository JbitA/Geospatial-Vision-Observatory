param(
    [ValidateSet("Default", "Cpu", "Cuda130")]
    [string]$TorchBackend = "Default",
    [switch]$ForceRecreate,
    [switch]$WithDevTools
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"


. (Join-Path $PSScriptRoot "windows_common.ps1")

function Find-Python314 {
    $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        & $PyLauncher.Source "-3.14" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,14,7) and sys.version_info < (3,15,0) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return [PSCustomObject]@{ Exe = $PyLauncher.Source; Args = @("-3.14") }
        }
    }

    $Python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $Python) {
        & $Python.Source -c "import sys; raise SystemExit(0 if sys.version_info >= (3,14,7) and sys.version_info < (3,15,0) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return [PSCustomObject]@{ Exe = $Python.Source; Args = @() }
        }
    }

    throw @"
Python 3.14.7 or newer in the 3.14 series is required for the reproducible Windows environment.
Install it from python.org or with:
  winget install -e --id Python.Python.3.14
Then reopen PowerShell and rerun this script.
"@
}

$Root = Get-GvoRoot
Set-Location $Root
$Venv = Get-GvoVenvRoot -Root $Root
$VenvPython = Get-GvoVenvPython -Root $Root

if ($ForceRecreate -and (Test-Path $Venv)) {
    Write-Host "Removing existing virtual environment: $Venv"
    Remove-Item -Recurse -Force $Venv
}

if (-not (Test-Path $VenvPython)) {
    $Launcher = Find-Python314
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Venv) | Out-Null
    Invoke-GvoNative $Launcher.Exe ($Launcher.Args + @("-m", "venv", $Venv))
}

if (-not (Test-Path $VenvPython)) {
    throw @"
Python created the virtual environment at a redirected location that PowerShell cannot use:
  expected: $VenvPython

This is commonly caused by Microsoft Store Python virtualizing LocalAppData.
The default GVO environment is deliberately under %USERPROFILE%\.gvo to avoid that redirect.
If this still occurs, install the latest Python 3.14 from python.org (or winget Python.Python.3.14),
then rerun setup with -ForceRecreate.
"@
}

Invoke-GvoNative $VenvPython @("-m", "pip", "install", "--upgrade", "pip==26.2.1")

if ($TorchBackend -eq "Cpu") {
    Invoke-GvoNative $VenvPython @(
        "-m", "pip", "install", "torch==2.13.0",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    )
}
elseif ($TorchBackend -eq "Cuda130") {
    Invoke-GvoNative $VenvPython @(
        "-m", "pip", "install", "torch==2.13.0",
        "--index-url", "https://download.pytorch.org/whl/cu130"
    )
}

$ProjectExtra = if ($WithDevTools) { ".[ml,dev]" } else { ".[ml]" }
Invoke-GvoNative $VenvPython @("-m", "pip", "install", "--prefer-binary", "-e", $ProjectExtra)
Invoke-GvoNative $VenvPython @("-m", "pip", "check")

Write-Host ""
Write-Host "Windows environment ready." -ForegroundColor Green
Invoke-GvoNative $VenvPython @(
    "-c",
    "import platform,sys,torch; print('Python:', sys.version.split()[0]); print('Windows:', platform.platform()); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
)
Write-Host "Virtual environment: $VenvPython"
