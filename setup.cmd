@echo off
rem JaNai Upscaler - one-shot setup. Any options are passed to setup.ps1,
rem e.g.  setup.cmd -Torch cpu -Models manga
setlocal
set "HERE=%~dp0"
where powershell.exe >nul 2>nul
if errorlevel 1 (
    echo Windows PowerShell is required but was not found.
    pause
    exit /b 1
)
powershell.exe -NoProfile -NoLogo -ExecutionPolicy Bypass -File "%HERE%setup.ps1" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo Setup exited with code %RC%.
echo.
pause
exit /b %RC%
