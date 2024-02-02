@echo off
cd /d %~dp0
call runtime python -s download_models.py
@REM if ERRORLEVEL 1 GOTO ABORT
echo "Model Download OK. Starting worker..."
call runtime python -s run_worker.py --amd %*

@REM :ABORT
@REM echo "download_models.py exited with error code. Aborting"
