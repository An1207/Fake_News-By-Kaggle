param([string]$DataDir = '')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not $DataDir) {
    $DataDir = if (Test-Path -LiteralPath 'Fake.csv') { '.' } else { 'data/raw' }
}
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
uv sync --extra web --extra dev --extra transformer --cache-dir .uv-cache
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
.venv/Scripts/python.exe -m news_ai train-transformer --data-dir $DataDir --revision 6720a49834876b4ec5afcf6ef0d149a2c6501cce --max-length 128 --batch-size 8 --threads 2 --backend openvino --device AUTO
if ($LASTEXITCODE -ne 0) { throw 'Transformer training failed. Cached features are retained for resuming.' }
