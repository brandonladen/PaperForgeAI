@echo off
title PaperForge AI - Research Paper to MVP
echo.
echo ====================================================
echo   PaperForge AI - One Click Launcher (Windows)
echo ====================================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check for API key
if "%OPENAI_API_KEY%"=="" (
    echo [WARNING] OPENAI_API_KEY not set
    echo.
    set /p OPENAI_API_KEY="Enter your OpenAI API key: "
)

REM Run the launcher
python run.py

pause
