@echo off
cd /d %~dp0
title AI Horde Worker

echo ============================================
echo   AI Horde Worker
echo ============================================
echo.

: Ensure the environment is set up
call runtime python -s -c "import torch"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Environment not set up. Please run update-runtime.cmd first.
    echo        Or use horde-worker.cmd which handles setup automatically.
    GOTO END
)

:DOWNLOAD
echo Downloading / verifying models...
call runtime python -s download_models.py
if %ERRORLEVEL% NEQ 0 GOTO ABORT
echo.
echo Models ready. Starting worker...
echo (Press Ctrl+C to stop the worker gracefully)
echo.
call runtime python -s run_worker.py %*

GOTO END

:ABORT
echo.
echo ERROR: Model download failed. Check the output above and try again.
echo        Common fix: check your internet connection and disk space.

:END
pause
