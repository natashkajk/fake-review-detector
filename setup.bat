@echo off
chcp 65001 >nul

REM Fake Review Detector - Setup Script for Windows
REM Automates the setup process for the project

echo ==========================================
echo Fake Review Detector - Setup Script
echo ==========================================
echo.

REM Check Python version
echo [INFO] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    exit /b 1
)

for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
echo [INFO] Found Python %PYTHON_VERSION%

REM Setup backend
echo [INFO] Setting up backend...

cd server

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo [INFO] Installing Python dependencies...
pip install -r requirements.txt

echo [INFO] Backend setup complete!
cd ..

REM Create .env file
echo [INFO] Creating environment file...
if not exist "server\.env" (
    copy server\.env.example server\.env
    echo [INFO] Created server\.env from example
) else (
    echo [WARN] server\.env already exists, skipping
)

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo.
echo 1. Start the backend server:
echo    cd server
echo    venv\Scripts\activate.bat
echo    python main.py
echo.
echo 2. Install the Chrome extension:
echo    - Open Chrome and go to chrome://extensions/
echo    - Enable 'Developer mode' (toggle in top right)
echo    - Click 'Load unpacked'
echo    - Select the 'client' folder
echo.
echo 3. Test the API:
echo    cd server
echo    venv\Scripts\activate.bat
echo    python test_api.py
echo.
echo API Documentation: http://localhost:8000/docs
echo.

pause