[CmdletBinding()]
param(
    [switch]$Rebuild,
    [switch]$SamOnly,
    [switch]$SkipSam,
    [switch]$RequireSam,
    [switch]$ForceSam
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'

function Invoke-Checked {
    param([string]$Executable, [string[]]$CommandArguments)
    & $Executable @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed (exit code $LASTEXITCODE)."
    }
}

function Get-InputHash {
    param([string]$Path)
    $stream = [System.IO.File]::OpenRead($Path)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return [System.BitConverter]::ToString($sha.ComputeHash($stream))
    } finally {
        $sha.Dispose()
        $stream.Dispose()
    }
}

if (!(Test-Path -LiteralPath $pythonPath)) {
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.venv')) {
        throw 'The existing .venv is not a Windows environment. Rename it and retry.'
    }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        Invoke-Checked $launcher.Source @('-3.12', '-m', 'venv', (Join-Path $PSScriptRoot '.venv'))
    } else {
        $pythonCommand = (Get-Command python.exe -ErrorAction Stop).Source
        Invoke-Checked $pythonCommand @('-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)')
        Invoke-Checked $pythonCommand @('-m', 'venv', (Join-Path $PSScriptRoot '.venv'))
    }
}
Invoke-Checked $pythonPath @('-c', 'import os, sys; sys.exit(0 if os.name == ''nt'' and sys.version_info >= (3, 12) else 1)')

$requirementsPath = Join-Path $PSScriptRoot 'requirements.txt'
$requirementsHash = Get-InputHash $requirementsPath
$requirementsStamp = Join-Path $PSScriptRoot '.venv/.labelimg-requirements.sha256'
$installedHash = if (Test-Path -LiteralPath $requirementsStamp) { (Get-Content -LiteralPath $requirementsStamp -Raw).Trim() } else { '' }
if ($installedHash -ne $requirementsHash -or $Rebuild) {
    Write-Host '[setup] Installing backend dependencies.'
    Invoke-Checked $pythonPath @('-m', 'pip', 'install', '-r', $requirementsPath)
    Set-Content -LiteralPath $requirementsStamp -Value $requirementsHash -Encoding ASCII
}

if (!$SamOnly) {
    $frontendDir = Join-Path $PSScriptRoot 'frontend'
    $buildInputs = @(
        Get-ChildItem -LiteralPath (Join-Path $frontendDir 'src') -Recurse -File
        Get-ChildItem -LiteralPath (Join-Path $frontendDir 'public') -Recurse -File
        Get-ChildItem -LiteralPath $frontendDir -File
        Get-Item -LiteralPath (Join-Path $PSScriptRoot 'build.ps1')
        Get-Item -LiteralPath (Join-Path $PSScriptRoot '.nvmrc')
    )
    $buildHash = ($buildInputs | Sort-Object FullName | ForEach-Object {
        $_.FullName + ':' + (Get-InputHash $_.FullName)
    }) -join "`n"
    $buildStamp = Join-Path $frontendDir 'dist/.windows-build.sha256'
    $previousBuildHash = if (Test-Path -LiteralPath $buildStamp) { (Get-Content -LiteralPath $buildStamp -Raw).TrimEnd() } else { '' }
    if ($Rebuild -or $previousBuildHash -ne $buildHash -or !(Test-Path -LiteralPath (Join-Path $frontendDir 'dist/index.html'))) {
        Write-Host '[setup] Building frontend.'
        # A fresh npm ci also replaces any Linux/WSL native Node modules.
        Invoke-Checked 'powershell.exe' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'build.ps1'), '-Install')
        Set-Content -LiteralPath $buildStamp -Value $buildHash -Encoding UTF8
    }
}

if (!$SkipSam) {
    $samArguments = @((Join-Path $PSScriptRoot 'install_sam.py'))
    if (!$RequireSam) { $samArguments += '--if-available' }
    if ($ForceSam) { $samArguments += '--force' }
    Invoke-Checked $pythonPath $samArguments
}
Write-Host '[setup] Ready.'
