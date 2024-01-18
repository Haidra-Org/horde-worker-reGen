@echo off
cd /d %~dp0
call runtime python -s download_models.py
if ERRORLEVEL 1 GOTO ABORT
echo "Model Download OK. Starting worker..."
call runtime python -s run_worker.py %*

:ABORT
echo "download_models.py exited with error code. Aborting"
