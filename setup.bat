@echo off

echo HomeValue-Analytics Installer
echo ===========================

REM Check Python
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.12 or newer.
    exit /b 1
)

REM Setup virtual environment
echo Setting up virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Build executable
echo Building application...
pyinstaller --onefile --windowed --clean --add-binary ".venv/Scripts/streamlit.exe;." --add-data "src;src" src/app.py

REM Create shortcut if requested
set /p create_shortcut="Create shortcut on Desktop? (Y/N): "
if /i "%create_shortcut%"=="Y" (
    echo Creating desktop shortcut...
    powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut([System.IO.Path]::Combine($env:USERPROFILE, 'Desktop', 'HomeValue-Analytics.lnk')); $s.TargetPath = '%cd%\dist\HomeValue-Analytics.exe'; $s.WorkingDirectory = '%cd%\dist'; $s.Save()"
    if errorlevel 1 (
        echo Failed to create shortcut!
    ) else (
        echo Shortcut created successfully!
    )
)

echo.
echo Installation complete! You can find the application in the dist folder.
pause