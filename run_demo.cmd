@echo off
setlocal

rem ============================================================
rem  Offline Demo Launcher
rem Starts Backend and Frontend in separate CMD windows.
rem ============================================================

set "PROJECT_NAME=AutonomousML"
set "PROJECT=G:\Other computers\My Laptop\Study\AI\AutonomousML"
set "VENV=c:\venvs\AutonomousML\.venv"

if not exist "%VENV%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found:
    echo %VENV%\Scripts\activate.bat
    pause
    exit /b 1
)

@REM  if not exist "%PROJECT%\server.py" (
@REM      echo [ERROR] server.py not found:
@REM      echo %PROJECT%\server.py
@REM      pause
@REM      exit /b 1
@REM  )

@REM  if not exist "%PROJECT%\frontend\index.html" (
@REM      echo [ERROR] frontend\index.html not found:
@REM      echo %PROJECT%\frontend\index.html
@REM      pause
@REM      exit /b 1
@REM  )

echo.
echo Starting %PROJECT_NAME%...
echo.

start "%PROJECT_NAME% - Backend" cmd /k "call ""%VENV%\Scripts\activate.bat"" && cd /d ""%PROJECT%\auto_ml_agent"" && echo [%PROJECT_NAME% - Backend] http://127.0.0.1:8081 && uvicorn server.py:app --host 127.0.0.1 --port 8081 --reload"
rem Wait 5 seconds
timeout /t 5 /nobreak >nul

start "%PROJECT_NAME% - Frontend" cmd /k "call ""%VENV%\Scripts\activate.bat"" && cd /d ""%PROJECT%\frontend"" && echo [%PROJECT_NAME% - Frontend] http://127.0.0.1:5500 && python -m http.server 5500"

rem ============================================================
rem Position Backend and Frontend windows 50/50
rem ============================================================

timeout /t 5 /nobreak >nul

powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Windows.Forms; $code = 'using System; using System.Runtime.InteropServices; public class Win32 { [DllImport(\"user32.dll\")] public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags); [DllImport(\"user32.dll\")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow); }'; Add-Type -TypeDefinition $code; $screen=[System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea; $width=[int]($screen.Width/2); $height=$screen.Height; $backend=(Get-Process cmd | Where-Object {$_.MainWindowTitle -eq '%PROJECT_NAME% - Backend'} | Select-Object -First 1); $frontend=(Get-Process cmd | Where-Object {$_.MainWindowTitle -eq '%PROJECT_NAME% - Frontend'} | Select-Object -First 1); if($backend){[Win32]::ShowWindow($backend.MainWindowHandle,3); [Win32]::SetWindowPos($backend.MainWindowHandle,[IntPtr]::Zero,$width,0,$width,$height,0x0040)}; if($frontend){[Win32]::ShowWindow($frontend.MainWindowHandle,3); [Win32]::SetWindowPos($frontend.MainWindowHandle,[IntPtr]::Zero,0,0,$width,$height,0x0040)}"

timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:5500"

echo.
echo Backend:  http://127.0.0.1:8081
echo Frontend: http://127.0.0.1:5500
echo.
echo Two CMD windows were opened.
echo Close those windows to stop the demo.
echo.

endlocal

pause