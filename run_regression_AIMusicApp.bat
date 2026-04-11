@echo off
setlocal

set "APP_DIR=%~dp0"
for %%I in ("%APP_DIR%.") do set "APP_ROOT=%%~fI"
cd /d "%APP_ROOT%"

set "PYTHON_EXE="
if defined AIMUSICAPP_TEST_PYTHON call :TryPython "%AIMUSICAPP_TEST_PYTHON%"
if not defined PYTHON_EXE if defined AIMUSICAPP_PYTHON call :TryPython "%AIMUSICAPP_PYTHON%"
if not defined PYTHON_EXE if exist "%APP_ROOT%\.venv-1\Scripts\python.exe" call :TryPython "%APP_ROOT%\.venv-1\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%APP_ROOT%\.venv\Scripts\python.exe" call :TryPython "%APP_ROOT%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" call :TryPython "%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE call :TryPython python

if not defined PYTHON_EXE (
  echo.
  echo Could not find a Python interpreter with the required AIMusicApp regression test dependencies.
  echo Set AIMUSICAPP_TEST_PYTHON to a valid python.exe or run setup_app_env.bat.
  pause
  exit /b 1
)

set "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"

echo Running full AIMusicApp regression suite with %PYTHON_EXE%
"%PYTHON_EXE%" -m pytest ^
  tests\test_setup_backend_env.py ^
  tests\test_music_backend_checks.py ^
  tests\test_music_model_backends.py ^
  tests\test_compare_music_models_integration.py ^
  tests\test_bootstrap_and_model_downloads.py ^
  tests\test_music_compare_gui_settings.py ^
  tests\test_music_compare_gui_desktop.py ^
  %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Full regression suite failed.
  pause
)

endlocal & exit /b %EXIT_CODE%

:TryPython
set "CANDIDATE=%~1"
if not defined CANDIDATE goto :eof
"%CANDIDATE%" -c "import tkinter, pytest, numpy, soundfile, PIL, scipy, imageio_ffmpeg" >nul 2>&1
if errorlevel 1 goto :eof
set "PYTHON_EXE=%CANDIDATE%"
goto :eof