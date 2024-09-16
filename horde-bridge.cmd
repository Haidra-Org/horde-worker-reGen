@echo off
cd /d %~dp0

: This first call to runtime activates the environment for the rest of the script
call runtime python -s -m pip -V

call python -s -m pip uninstall hordelib
call python -s -m pip install horde_sdk~=0.14.3 horde_model_reference~=0.9.0 horde_engine~=2.15.0 horde_safety~=0.2.3 -U

if %ERRORLEVEL% NEQ 0 (
    echo "Please run update-runtime.cmd."
    GOTO END
)

call python -s -m pip check
if %ERRORLEVEL% NEQ 0 (
    echo "Please run update-runtime.cmd."
    GOTO END
)

:DOWNLOAD
call python -s download_models.py
if %ERRORLEVEL% NEQ 0 GOTO ABORT
echo "Model Download OK. Starting worker..."
call python -s run_worker.py %*

GOTO END

:ABORT
echo "download_models.py exited with error code. Aborting"

:END
call micromamba deactivate >nul
call deactivate >nul
pause
