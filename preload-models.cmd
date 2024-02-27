@echo off
cd /d %~dp0
call runtime python -s download_models.py
pause
