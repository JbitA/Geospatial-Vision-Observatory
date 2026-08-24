Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-GvoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-GvoUserStateRoot {
    if (-not [string]::IsNullOrWhiteSpace($env:GVO_STATE_ROOT)) {
        return [System.IO.Path]::GetFullPath($env:GVO_STATE_ROOT)
    }
    if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        return Join-Path $env:USERPROFILE ".gvo"
    }
    throw "USERPROFILE is unavailable. Set GVO_STATE_ROOT to a writable non-synced directory."
}

function Get-GvoVenvRoot {
    param([string]$Root)
    if (-not [string]::IsNullOrWhiteSpace($env:GVO_VENV_DIR)) {
        return [System.IO.Path]::GetFullPath($env:GVO_VENV_DIR)
    }
    return Join-Path (Get-GvoUserStateRoot) "venv-py314"
}

function Get-GvoVenvPython {
    param([string]$Root)
    return Join-Path (Get-GvoVenvRoot -Root $Root) "Scripts\python.exe"
}

function Get-GvoCuratedRoot {
    param([string]$Root, [string]$Requested = "")
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        return [System.IO.Path]::GetFullPath($Requested)
    }
    if (-not [string]::IsNullOrWhiteSpace($env:GVO_DATA_ROOT)) {
        return [System.IO.Path]::GetFullPath($env:GVO_DATA_ROOT)
    }
    return Join-Path (Get-GvoUserStateRoot) "curated"
}

function Set-GvoNativeRuntimeStorage {
    param([string]$Root)
    if ([string]::IsNullOrWhiteSpace($env:VISION_DATA_DIR)) {
        $RuntimeRoot = Join-Path (Get-GvoUserStateRoot) "runtime"
        New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
        $env:VISION_DATA_DIR = $RuntimeRoot
        $env:VISION_DB_PATH = Join-Path $RuntimeRoot "geospatial.db"
    }
}

function Invoke-GvoNative {
    param([string]$FilePath, [string[]]$Arguments)
    Write-Host "+ $FilePath $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Test-GvoEditableInstall {
    param([string]$Root, [string]$Python)
    $Previous = $env:GVO_EXPECTED_ROOT
    try {
        $env:GVO_EXPECTED_ROOT = $Root
        & $Python -c "import os,pathlib,sys; expected=(pathlib.Path(os.environ['GVO_EXPECTED_ROOT'])/'src'/'geo_vision').resolve(); import geo_vision; actual=pathlib.Path(geo_vision.__file__).resolve(); sys.exit(0 if expected in actual.parents else 1)" *> $null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        if ($null -eq $Previous) {
            Remove-Item Env:GVO_EXPECTED_ROOT -ErrorAction SilentlyContinue
        }
        else {
            $env:GVO_EXPECTED_ROOT = $Previous
        }
    }
}

function Ensure-GvoEditableInstall {
    param([string]$Root, [string]$Python)
    if (-not (Test-Path $Python)) {
        throw "Virtual environment not found. Run .\scripts\setup.cmd first."
    }
    if (Test-GvoEditableInstall -Root $Root -Python $Python) {
        return
    }

    Write-Host "Repairing editable install for current project folder..." -ForegroundColor Yellow
    Invoke-GvoNative $Python @(
        "-m", "pip", "install", "--no-deps", "--force-reinstall", "-e", $Root
    )
    if (-not (Test-GvoEditableInstall -Root $Root -Python $Python)) {
        throw "Editable install repair completed but geo_vision still does not resolve to the current project."
    }
}
