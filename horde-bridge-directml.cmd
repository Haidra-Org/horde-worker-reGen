@echo off
cd /d %~dp0

: Ensure the environment is set up
call runtime python -s -c "import torch"
if %ERRORLEVEL% NEQ 0 (
    echo "Please run update-runtime-directml.cmd."
    GOTO END
)

:DOWNLOAD
call runtime python -s download_models.py --directml=0
if %ERRORLEVEL% NEQ 0 GOTO ABORT
echo "Model Download OK. Starting worker..."
call runtime python -s run_worker.py --directml=0 %*

GOTO END

:ABORT
echo "download_models.py exited with error code. Aborting"

:END
pause
