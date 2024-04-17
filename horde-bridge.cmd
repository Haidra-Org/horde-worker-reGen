@echo off
cd /d %~dp0

:: Check if AI_HORDE_URL is set and if it is not, set it
if "%AI_HORDE_URL%"=="" (
    echo Setting AI_HORDE_URL environment variable
    set AI_HORDE_URL=https://api.aipowergrid.io/api/
) else (
    echo AI_HORDE_URL is already set to %AI_HORDE_URL%
)

:: Ask for GPU selection
echo Select which GPU to use by index (e.g., 0, 1, 2...):
set /p GPU_INDEX="Enter GPU index: "
set CUDA_VISIBLE_DEVICES=%GPU_INDEX%
echo Using GPU %CUDA_VISIBLE_DEVICES%

: This first call to runtime activates the environment for the rest of the script
call runtime python -s -m pip -V

call python -s -m pip install horde_sdk~=0.9.3 horde_model_reference~=0.6.3 hordelib~=2.8.1 -U
if %ERRORLEVEL% NEQ 0 (
    echo "Please run update-runtime.cmd."
    GOTO END
)

:: Change constants in path_consts.py
echo Modifying path_consts.py to update repository details...
call python update_path_consts.py


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
