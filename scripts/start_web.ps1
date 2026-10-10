param([ValidateRange(5, 180)][int]$TimeoutSeconds = 60)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $projectRoot '.tools/server-logs'

function Test-Listening([int]$Port) {
    $connection = [System.Net.Sockets.TcpClient]::new()
    try {
        return $connection.ConnectAsync('127.0.0.1', $Port).Wait(1000) -and $connection.Connected
    } catch { return $false } finally { $connection.Dispose() }
}

function Test-Ready([string]$Url, [string]$Kind) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
        if ($response.StatusCode -ne 200) { return $false }
        if ($Kind -eq 'api') {
            $health = $response.Content | ConvertFrom-Json
            return $health.status -eq 'ok' -and $health.database -eq 'connected'
        }
        return $response.Content.Contains('<title>SAI')
    } catch { return $false }
}

function Wait-Ready([string]$Url, [string]$Kind) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        if (Test-Ready $Url $Kind) { return }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    throw "Service did not become ready: $Url. Check private logs in $logRoot."
}

Push-Location $projectRoot
try {
    foreach ($required in @('.env', '.venv/Scripts/python.exe', 'frontend/node_modules/vite/bin/vite.js')) {
        if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $required))) {
            throw "Missing $required. Complete scripts/setup_web.ps1 first."
        }
    }
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    docker desktop start
    if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop, then retry this script.' }
    docker compose -p sai-news up -d --wait --wait-timeout $TimeoutSeconds mysql
    if ($LASTEXITCODE -ne 0) { throw 'Project MySQL is not healthy. Check Docker Desktop.' }

    & (Join-Path $projectRoot '.venv/Scripts/python.exe') -m alembic -c backend/alembic.ini upgrade head
    if ($LASTEXITCODE -ne 0) { throw 'Database migration failed. Check MySQL settings.' }

    $apiPort = & (Join-Path $projectRoot '.venv/Scripts/python.exe') -c 'from news_api.config import Settings; print(Settings().api_port)'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read API port configuration.' }
    $apiUrl = "http://127.0.0.1:$apiPort/api/health"
    if (-not (Test-Listening ([int]$apiPort))) {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        $apiProcess = Start-Process -FilePath (Join-Path $projectRoot '.venv/Scripts/python.exe') `
            -ArgumentList @('-m', 'uvicorn', 'news_api.main:app', '--host', '127.0.0.1', '--port', $apiPort) `
            -WorkingDirectory $projectRoot -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logRoot "api-$stamp.stdout.log") `
            -RedirectStandardError (Join-Path $logRoot "api-$stamp.stderr.log") -PassThru
        Write-Host "Started API in background (launcher PID $($apiProcess.Id))."
    }
    Wait-Ready $apiUrl 'api'

    $frontendUrl = 'http://127.0.0.1:5173/'
    if (-not (Test-Listening 5173)) {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        $frontendProcess = Start-Process -FilePath (Get-Command node.exe -ErrorAction Stop).Source `
            -ArgumentList @('node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5173', '--strictPort') `
            -WorkingDirectory (Join-Path $projectRoot 'frontend') -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logRoot "frontend-$stamp.stdout.log") `
            -RedirectStandardError (Join-Path $logRoot "frontend-$stamp.stderr.log") -PassThru
        Write-Host "Started frontend in background (PID $($frontendProcess.Id))."
    }
    Wait-Ready $frontendUrl 'frontend'
    # Test the same proxy route used by React, not just the direct API port.
    Wait-Ready ($frontendUrl + 'api/health') 'api'
    Write-Host 'Ready: MySQL + FastAPI + React proxy.'
    Write-Host "Open $frontendUrl (refresh an already-open page)."
    Write-Host "Private logs: $logRoot"
} finally { Pop-Location }
