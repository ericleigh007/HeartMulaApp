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
  echo Could not find a Python interpreter with the required AIMusicApp desktop test dependencies.
  echo Set AIMUSICAPP_TEST_PYTHON to a valid python.exe or install the app dependencies into .venv.
  pause
  exit /b 1
)

set "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"

echo Running desktop regression suite with %PYTHON_EXE%
"%PYTHON_EXE%" -m pytest tests\test_music_compare_gui_desktop.py tests\test_music_compare_gui_settings.py %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Desktop regression suite failed.
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