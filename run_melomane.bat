@echo off
setlocal enabledelayedexpansion
set UV_SYSTEM_CERTS=1

cd /d "%~dp0"

echo ======================================================================
echo                      MELOMANE WINDOWS LAUNCHER
echo   Music-domain replication of Epicure (arXiv:2604.22776, 2605.22391)
echo ======================================================================

set PYTHON_CMD=
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV_BIN=%USERPROFILE%\.local\bin\uv.exe"
) else (
    where uv >nul 2>&1
    if !errorlevel! equ 0 (
        set "UV_BIN=uv"
    )
)

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else if defined UV_BIN (
    echo Using uv to bootstrap Python 3.12 environment...
    "%UV_BIN%" run --python 3.12 scripts\bootstrap.py setup
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON_CMD=.venv\Scripts\python.exe"
    )
)

if not defined PYTHON_CMD (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=python"
    ) else (
        echo [ERROR] Neither uv nor python.exe found in PATH.
        echo Please ensure Python 3.12 or uv is installed.
        pause
        exit /b 1
    )
)

if "%~1"=="" (
    echo Launching full pipeline: Smoke Test -^> Status -^> Overnight Training...
    "%PYTHON_CMD%" scripts\bootstrap.py auto
) else (
    "%PYTHON_CMD%" scripts\bootstrap.py %*
)

if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Process exited with code !errorlevel!.
    echo Running diagnostic doctor...
    "%PYTHON_CMD%" scripts\bootstrap.py doctor
)

echo.
echo Process complete.
if "%~1"=="" pause
