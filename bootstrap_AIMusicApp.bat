@echo off
setlocal

cd /d "%~dp0"

set "BOOTSTRAP_PYTHON="
if exist "%~dp0.venv\Scripts\python.exe" set "BOOTSTRAP_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined BOOTSTRAP_PYTHON if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "BOOTSTRAP_PYTHON=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined BOOTSTRAP_PYTHON set "BOOTSTRAP_PYTHON=python"

echo Bootstrapping AIMusicApp with %BOOTSTRAP_PYTHON%
"%BOOTSTRAP_PYTHON%" tools\common\bootstrap_aimusicapp.py %*

if errorlevel 1 (
  echo.
  echo AIMusicApp bootstrap failed.
  pause
)

endlocal