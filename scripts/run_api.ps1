$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    # Single worker: local inference queue and restart recovery are process-local.
    & .\.venv\Scripts\python.exe -m uvicorn news_api.main:app --host 127.0.0.1 --port 8001
} finally { Pop-Location }
