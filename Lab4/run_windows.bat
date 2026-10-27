@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    set "PY=python"
) else (
    set "PY=py -3"
)
%PY% -m pip install --upgrade PyOpenGL >nul 2>&1
if errorlevel 1 goto :error
%PY% main.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Failed to start. Make sure you use 64-bit Python and that the Microsoft
echo Visual C++ 2015-2022 Redistributable is installed.
pause
exit /b 1
