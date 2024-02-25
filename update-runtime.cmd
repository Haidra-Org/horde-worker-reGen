@echo off
:Check if micromamba is already installed
if exist micromamba.exe goto Isolation
  curl.exe -L -o micromamba.exe https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64

cd /d "%~dp0"

:Isolation
SET CONDA_SHLVL=
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=
SET MAMBA_ROOT_PREFIX=%~dp0conda
echo %MAMBA_ROOT_PREFIX%



setlocal EnableDelayedExpansion
for %%a in (%*) do (
    if /I "%%a"=="--hordelib" (
        set hordelib=true
    ) else (
        set hordelib=
    )
    if /I "%%a"=="--scribe" (
        set scribe=true
    ) else (
        set scribe=
    )
)
endlocal

if defined scribe (
  SET CONDA_ENVIRONMENT_FILE=environment_scribe.yaml

) else (
  SET CONDA_ENVIRONMENT_FILE=environment.yaml
)

Reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d "1" /f 2>nul
:We do this twice the first time to workaround a conda bug where pip is not installed correctly the first time - Henk
IF EXIST CONDA GOTO WORKAROUND_END
micromamba create --no-shortcuts -r conda -n windows -f %CONDA_ENVIRONMENT_FILE% -y
:WORKAROUND_END
micromamba create --no-shortcuts -r conda -n windows -f %CONDA_ENVIRONMENT_FILE% -y

REM Check if hordelib argument is defined

micromamba.exe shell hook -s cmd.exe -p %MAMBA_ROOT_PREFIX% -v
call "%MAMBA_ROOT_PREFIX%\condabin\mamba_hook.bat"
call "%MAMBA_ROOT_PREFIX%\condabin\micromamba.bat" activate windows

if defined hordelib (
  python -s -m pip uninstall -y hordelib horde_model_reference
  python -s -m pip install hordelib horde_model_reference --extra-index-url https://download.pytorch.org/whl/cu121
) else (
  if defined scribe (
    python -s -m pip install -r requirements-scribe.txt
  ) else (
    python -s -m pip install -r requirements.txt -U --extra-index-url https://download.pytorch.org/whl/cu121
  )
)
call micromamba deactivate

echo If there are no errors above everything should be correctly installed (If not, try deleting the folder /conda/envs/ and try again).

pause
