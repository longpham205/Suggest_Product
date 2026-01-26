# Script to run backend with correct PYTHONPATH
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Get-Item $scriptPath).Parent.Parent.FullName

$env:PYTHONPATH = $projectRoot
Write-Host "PYTHONPATH set to: $projectRoot"
Write-Host "Starting backend on http://127.0.0.1:8080"
Write-Host "Swagger docs at http://127.0.0.1:8080/docs"
Write-Host ""

Set-Location $scriptPath
uvicorn main:app --host 127.0.0.1 --port 8080 --reload
