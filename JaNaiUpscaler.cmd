@echo off
rem JaNai Upscaler - launcher. Uses the environment in .\backend\python.
setlocal
set "HERE=%~dp0"
set "MAIN=%HERE%src\janai\app\main.py"
set "PY="

rem 1. whatever setup recorded
if exist "%HERE%janai.runtime.txt" set /p PY=<"%HERE%janai.runtime.txt"
if defined PY if not exist "%PY%" set "PY="

rem 2. the venv uv creates, or a standalone CPython in the same folder
if not defined PY if exist "%HERE%backend\python\Scripts\pythonw.exe" set "PY=%HERE%backend\python\Scripts\pythonw.exe"
if not defined PY if exist "%HERE%backend\python\Scripts\python.exe"  set "PY=%HERE%backend\python\Scripts\python.exe"
if not defined PY if exist "%HERE%backend\python\pythonw.exe"         set "PY=%HERE%backend\python\pythonw.exe"
if not defined PY if exist "%HERE%backend\python\python.exe"          set "PY=%HERE%backend\python\python.exe"

if defined PY (
    start "JaNai Upscaler" "%PY%" "%MAIN%" %*
    goto :eof
)

echo No environment in backend\python - run setup.cmd once to create it.
echo Trying the system Python: the interface will open, but upscaling needs setup.
where pythonw.exe >nul 2>nul
if not errorlevel 1 (
    start "JaNai Upscaler" pythonw.exe "%MAIN%" %*
    goto :eof
)
where python.exe >nul 2>nul
if not errorlevel 1 (
    python.exe "%MAIN%" %*
    goto :eof
)

echo No Python was found at all. Run setup.cmd first.
pause
