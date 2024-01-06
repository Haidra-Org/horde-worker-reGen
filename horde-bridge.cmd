@echo off
cd /d %~dp0
call runtime python -s download_models.py
call runtime python -s run_worker.py %*
