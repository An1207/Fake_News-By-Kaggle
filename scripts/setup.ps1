$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. See README.md for pip or uv setup instructions.'
}
uv --cache-dir .uv-cache sync --frozen --extra web --extra dev
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
