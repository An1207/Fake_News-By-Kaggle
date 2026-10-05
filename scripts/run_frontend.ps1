$ErrorActionPreference = "Stop"
Push-Location (Join-Path (Split-Path -Parent $PSScriptRoot) "frontend")
try { npm.cmd run dev } finally { Pop-Location }
