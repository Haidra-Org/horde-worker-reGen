@echo off
cd /d "%~dp0"

:Isolation
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=
SET CONDA_SHLVL=

Reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d "1" /f 2>nul

IF EXIST ".venv" GOTO APP

:INSTALL
call update-runtime

:APP
"%~dp0bin\uv.exe" run --no-sync %*
