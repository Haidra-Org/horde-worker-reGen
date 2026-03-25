@echo off
cd /d "%~dp0"
title AI Horde Worker - Update Runtime

echo ============================================
echo   AI Horde Worker - Install / Update
echo ============================================
echo.

:Isolation
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=
SET CONDA_SHLVL=

Reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d "1" /f 2>nul

REM Parse arguments for GPU backend selection
SET GPU_EXTRA=cu128
for %%a in (%*) do (
    if /I "%%a"=="--rocm" SET GPU_EXTRA=rocm
    if /I "%%a"=="--directml" SET GPU_EXTRA=directml
    if /I "%%a"=="--cpu" SET GPU_EXTRA=cpu
)

REM Install uv if not present
if not exist "%~dp0bin\uv.exe" (
    echo Downloading uv package manager...
    powershell -ExecutionPolicy ByPass -NoProfile -c "$env:UV_INSTALL_DIR='%~dp0bin'; irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to download uv. Check your internet connection.
        pause
        exit /b 1
    )
    echo Done.
    echo.
)

echo Installing dependencies for GPU backend: %GPU_EXTRA%
echo (This may take a few minutes on first run...)
echo.
"%~dp0bin\uv.exe" sync --locked --extra %GPU_EXTRA%
if errorlevel 1 (
    echo.
    echo ERROR: Installation failed.
    echo   - Try deleting the .venv folder and running this script again.
    echo   - If the problem persists, ask for help in #local-workers on Discord.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Installation complete!
echo ============================================
echo.
echo Next steps:
echo   1. Edit bridgeData.yaml with your API key and worker name
echo   2. Run horde-bridge.cmd to start the worker
echo      (or horde-worker.cmd for the interactive launcher)
echo.
pause
