$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Run scripts/setup.ps1 first.' }
& $projectPython -m news_ai train --data-dir $projectRoot --output-dir (Join-Path $projectRoot 'artifacts')
if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }
