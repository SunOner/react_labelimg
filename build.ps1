[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Install,
    [switch]$NoInstall,
    [switch]$Lint,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'

if ($Help) {
    Write-Host @'
Usage: .\build.ps1 [-Clean] [-Install] [-NoInstall] [-Lint]

  -Clean       Remove frontend/node_modules and frontend/dist, then run npm ci.
  -Install     Run npm ci even if node_modules already exists.
  -NoInstall   Skip dependency installation.
  -Lint        Run npm run lint before building.
  -Help        Show this help.
'@
    exit 0
}

if ($NoInstall -and ($Clean -or $Install)) {
    throw '-NoInstall cannot be combined with -Clean or -Install.'
}

$frontendDir = Join-Path $PSScriptRoot 'frontend'
$nodeCommand = (Get-Command node.exe -ErrorAction Stop).Source
# Use npm.cmd so that npm.ps1 does not require an execution policy change.
$npmCommand = (Get-Command npm.cmd -ErrorAction Stop).Source

function Invoke-Npm {
    param([string[]]$NpmArguments)
    & $npmCommand @NpmArguments --prefix $frontendDir
    if ($LASTEXITCODE -ne 0) {
        throw "npm $($NpmArguments -join ' ') failed (exit code $LASTEXITCODE)."
    }
}

& $nodeCommand -e "const [major, minor] = process.versions.node.split('.').map(Number); process.exit(major > 22 || (major === 22 && minor >= 12) || (major === 20 && minor >= 19) ? 0 : 1)"
if ($LASTEXITCODE -ne 0) {
    throw 'Use Node 22 from .nvmrc, Node 20.19+, or Node 22.12+.'
}
Write-Host "[build] Node: $(& $nodeCommand --version) ($nodeCommand)"

if ($Clean) {
    foreach ($directoryName in @('node_modules', 'dist')) {
        $target = [System.IO.Path]::GetFullPath((Join-Path $frontendDir $directoryName))
        $expected = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "frontend/$directoryName"))
        if ($target -ne $expected) {
            throw "Unexpected cleanup path: $target"
        }
        if (Test-Path -LiteralPath $target) {
            Write-Host "[build] Removing $target"
            Remove-Item -LiteralPath $target -Recurse -Force
        }
    }
}

if (!$NoInstall -and ($Clean -or $Install -or !(Test-Path -LiteralPath (Join-Path $frontendDir 'node_modules')))) {
    Invoke-Npm -NpmArguments @('ci')
}
if ($Lint) {
    Invoke-Npm -NpmArguments @('run', 'lint')
}
Invoke-Npm -NpmArguments @('run', 'build')
Write-Host '[build] Done. Frontend build is in frontend/dist.'
