@echo off
setlocal
cd /d "%~dp0"
python tools\common\setup_backend_env.py melodyflow %*
endlocal