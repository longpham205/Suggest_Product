# Script to run frontend web server
# Uses Python's built-in HTTP server

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendPath = Join-Path $scriptPath "frontend"

Write-Host "Starting Frontend on http://127.0.0.1:5500"
Write-Host "Make sure Backend is running on http://127.0.0.1:8080"
Write-Host ""

Set-Location $frontendPath
python -m http.server 5500 --bind 127.0.0.1
