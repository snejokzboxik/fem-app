@echo off
REM Windows launcher for the Streamlit app.
REM This file is intentionally ASCII-only for cmd.exe compatibility.

cd /d "%~dp0"

echo Starting charged_particle_trap Streamlit app
echo Project folder: %CD%
echo.

if not exist ".venv\Scripts\python.exe" (
    echo .venv was not found. Creating local virtual environment...
    py -m venv ".venv"
    if errorlevel 1 goto error
)

echo Installing or updating requirements...
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 goto error

echo.
echo Starting Streamlit...
echo Local URL: http://localhost:8501
echo If the browser does not open, copy the URL manually.
echo.

".venv\Scripts\python.exe" -m streamlit run "app.py" --server.headless=false
if errorlevel 1 goto error

exit /b 0

:error
echo.
echo Launch failed. Check the messages above.
pause
exit /b 1
