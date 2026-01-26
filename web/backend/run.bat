

@echo off
REM Script to run backend with correct PYTHONPATH
cd /d "%~dp0"
set PYTHONPATH=%~dp0..\..
echo Starting backend on http://127.0.0.1:8080
echo Swagger docs at http://127.0.0.1:8080/docs
echo.
uvicorn main:app --host 127.0.0.1 --port 8080 --reload
