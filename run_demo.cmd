```bat
@echo off
setlocal

rem ============================================================
rem AutonomousML - Offline Demo Launcher
rem ============================================================

set "PROJECT_NAME=AutonomousML"
set "PROJECT=G:\Other computers\My Laptop\Study\AI\AutonomousML"
set "VENV=C:\venvs\AutonomousML\.venv"

rem Validate virtual environment
if not exist "%VENV%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found:
    echo %VENV%\Scripts\activate.bat
    pause
    exit /b 1
)

rem Validate project directory
if not exist "%PROJECT%\auto_ml_agent\server.py" (
    echo [ERROR] Backend file not found:
    echo %PROJECT%\auto_ml_agent\server.py
    pause
    exit /b 1
)

if not exist "%PROJECT%\frontend\index.html" (
    echo [ERROR] Frontend file not found:
    echo %PROJECT%\frontend\index.html
    pause
    exit /b 1
)

echo.
echo Starting %PROJECT_NAME%...
echo.

rem Start backend
start "%PROJECT_NAME% - Backend" cmd /k "call ""%VENV%\Scripts\activate.bat"" && cd /d ""%PROJECT%"" && python -m uvicorn auto_ml_agent.server:app --host 127.0.0.1 --port 8000 --reload"

rem Wait for backend startup
timeout /t 5 /nobreak >nul

rem Start frontend
@REM  start "%PROJECT_NAME% - Frontend" cmd /k "call ""%VENV%\Scripts\activate.bat"" && cd /d ""%PROJECT%\frontend"" && python -m http.server 5500"

@REM  rem Wait before opening browser
@REM  timeout /t 3 /nobreak >nul

start "" "http://127.0.0.1:8000"

echo.
echo Backend:  http://127.0.0.1:8000
echo API Docs: http://127.0.0.1:8000/docs
@REM  echo Frontend: http://127.0.0.1:5500
echo.
echo Close the two server windows to stop the demo.
echo.

endlocal
```