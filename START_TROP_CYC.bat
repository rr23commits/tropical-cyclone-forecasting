@echo off
setlocal
cd /d "%~dp0"
title TROP-CYC Project Launcher

echo ==========================================
echo       TROP-CYC - Project Setup
echo ==========================================
echo.

if not exist "requirements.txt" (
  echo ERROR: requirements.txt not found.
  echo Open the extracted project folder in VS Code and run this file there.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  where py >nul 2>nul
  if %errorlevel%==0 (
    py -3.11 -m venv .venv
    if errorlevel 1 python -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 (
    echo ERROR: Could not create .venv. Install Python 3.11 and try again.
    pause
    exit /b 1
  )
)

set "PY=%CD%\.venv\Scripts\python.exe"

echo.
echo Installing/checking Python requirements...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto failed
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo.
echo Preparing the spatial replay data...
"%PY%" -m src.export_replay
if errorlevel 1 goto failed

echo.
echo Preparing monthly model data (including ARCH/GARCH, VAR and PatchTST)...
"%PY%" -m src.export_monthly_frontend
if errorlevel 1 goto failed

if not exist "frontend\data\genesis.json" (
  echo ERROR: frontend\data\genesis.json is missing.
  echo The Genesis page cannot load without this project data file.
  pause
  exit /b 1
)

echo.
echo Checking that the three pages and data files exist...
for %%F in ("frontend\index.html" "frontend\monthly.html" "frontend\genesis.html" "frontend\data\replay.json" "frontend\data\monthly.json" "frontend\data\genesis.json") do (
  if not exist "%%~F" (
    echo ERROR: Missing %%~F
    pause
    exit /b 1
  )
)

echo.
echo Starting local web server on port 8000...
start "TROP-CYC Web Server" cmd /k ""%PY%" -m http.server 8000 --directory frontend"
timeout /t 3 /nobreak >nul

echo Opening the three project pages...
start "" "http://127.0.0.1:8000/index.html"
start "" "http://127.0.0.1:8000/monthly.html"
start "" "http://127.0.0.1:8000/genesis.html"

echo.
echo TROP-CYC is running.
echo Keep the TROP-CYC Web Server window open.
echo To stop the server, press Ctrl+C in that window.
echo.
pause
exit /b 0

:failed
echo.
echo ERROR: A setup or data-preparation command failed.
echo Read the error above, fix it, then run START_TROP_CYC.bat again.
pause
exit /b 1
