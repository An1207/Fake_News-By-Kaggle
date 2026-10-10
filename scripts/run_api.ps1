$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    # Single worker: local inference queue and restart recovery are process-local.
    $apiPort = & .\.venv\Scripts\python.exe -c 'from news_api.config import Settings; print(Settings().api_port)'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read API port configuration.' }
    & .\.venv\Scripts\python.exe -m uvicorn news_api.main:app --host 127.0.0.1 --port $apiPort
} finally { Pop-Location }
