@echo off
cd /d %~dp0
call runtime python -s download_models.py
call "%MAMBA_ROOT_PREFIX%\condabin\micromamba.bat" deactivate >nul
call deactivate >nul
pause
