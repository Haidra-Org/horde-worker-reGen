@echo off
cd /d %~dp0
call runtime python -s download_models.py
if %ERRORLEVEL% NEQ 0 GOTO ABORT
echo "Model Download OK. Starting worker..."
call runtime python -s run_worker.py %*

GOTO END

:ABORT
echo "download_models.py exited with error code. Aborting"

:END
call micromamba deactivate >nul
call deactivate >nul
pause