@echo off
setlocal

set "APP_DIR=%~dp0"
for %%I in ("%APP_DIR%.") do set "APP_ROOT=%%~fI"
cd /d "%APP_ROOT%"

if not defined ACESTEP15_ROOT if exist "%APP_ROOT%third_party\ACE-Step-1.5" set "ACESTEP15_ROOT=%APP_ROOT%third_party\ACE-Step-1.5"
if not defined ACESTEP15_CKPT_DIR if exist "%APP_ROOT%models\comparison\ace-step-1.5\checkpoints" set "ACESTEP15_CKPT_DIR=%APP_ROOT%models\comparison\ace-step-1.5\checkpoints"
if not defined ACESTEP15_PYTHON if exist "%APP_ROOT%third_party\ACE-Step-1.5\.venv\Scripts\python.exe" set "ACESTEP15_PYTHON=%APP_ROOT%third_party\ACE-Step-1.5\.venv\Scripts\python.exe"
if not defined ACESTEP15_TURBO_CONFIG_PATH set "ACESTEP15_TURBO_CONFIG_PATH=acestep-v15-turbo"
if not defined ACESTEP15_SFT_CONFIG_PATH set "ACESTEP15_SFT_CONFIG_PATH=acestep-v15-sft"
if not defined ACESTEP15_TASK_TYPE set "ACESTEP15_TASK_TYPE=auto"

set "PYTHON_EXE="
if defined AIMUSICAPP_PYTHON call :TryPython "%AIMUSICAPP_PYTHON%"
if not defined PYTHON_EXE if exist "%APP_ROOT%\.venv-1\Scripts\python.exe" call :TryPython "%APP_ROOT%\.venv-1\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%APP_ROOT%\.venv\Scripts\python.exe" call :TryPython "%APP_ROOT%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" call :TryPython "%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE call :TryPython python

if not defined PYTHON_EXE (
  echo.
  echo Could not find a Python interpreter with the required AIMusicApp dependencies.
  echo Set AIMUSICAPP_PYTHON to a valid python.exe or run setup_app_env.bat.
  pause
  exit /b 1
)

echo Launching AIMusicApp with %PYTHON_EXE%
"%PYTHON_EXE%" tools\ai\music_compare_gui.py

if errorlevel 1 (
  echo.
  echo AIMusicApp exited with an error.
  pause
)

endlocal
exit /b 0

:TryPython
set "CANDIDATE=%~1"
if not defined CANDIDATE goto :eof
"%CANDIDATE%" -c "import numpy, soundfile, PIL, scipy, imageio_ffmpeg" >nul 2>&1
if errorlevel 1 goto :eof
set "PYTHON_EXE=%CANDIDATE%"
goto :eof
