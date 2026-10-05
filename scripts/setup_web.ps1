$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    uv --cache-dir .uv-cache sync --frozen --extra web --extra dev
    if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
    & .\.venv\Scripts\python.exe scripts\init_web_env.py
    if ($LASTEXITCODE -ne 0) { throw "Environment initialization failed." }
    docker compose -p sai-news up -d --wait --wait-timeout 120 mysql
    if ($LASTEXITCODE -ne 0) { throw "MySQL startup failed. Check Docker Desktop and the configured port." }
    & .\.venv\Scripts\python.exe -m alembic -c backend\alembic.ini upgrade head
    if ($LASTEXITCODE -ne 0) { throw "Database migration failed." }
    Push-Location frontend
    try {
        npm.cmd ci --cache ..\.npm-cache
        if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
        npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    } finally { Pop-Location }
    Write-Host "Setup complete. Run scripts/run_api.ps1 and scripts/run_frontend.ps1 in separate terminals."
} finally { Pop-Location }
