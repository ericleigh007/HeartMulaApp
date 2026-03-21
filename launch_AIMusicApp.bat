@echo off
setlocal

set "APP_DIR=%~dp0"
for %%I in ("%APP_DIR%.") do set "APP_ROOT=%%~fI"
cd /d "%APP_ROOT%"

set "PYTHON_EXE="
if exist "%APP_ROOT%\.venv-1\Scripts\python.exe" set "PYTHON_EXE=%APP_ROOT%\.venv-1\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%APP_ROOT%\.venv\Scripts\python.exe" set "PYTHON_EXE=%APP_ROOT%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

echo Launching AIMusicApp with %PYTHON_EXE%
"%PYTHON_EXE%" tools\ai\music_compare_gui.py

if errorlevel 1 (
  echo.
  echo AIMusicApp exited with an error.
  pause
)

endlocal
