@echo off
setlocal DisableDelayedExpansion
rem Keep server output on handle 3; cmd's Ctrl+C confirmation gets no input/output.
call :run %* 3>&1 >nul <nul
exit /b %errorlevel%

:run
set "PYTHONUTF8=1"
pushd "%~dp0" || exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows.ps1" 1>&3 2>&3
if errorlevel 1 goto finish
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py" %* 1>&3 2>&3
:finish
set "LABELIMG_EXIT_CODE=%errorlevel%"
if "%LABELIMG_EXIT_CODE%"=="130" goto stopped
if "%LABELIMG_EXIT_CODE%"=="-1073741510" goto stopped
if not "%LABELIMG_EXIT_CODE%"=="0" if not defined LABELIMG_NO_PAUSE pause <con >con
:stopped
popd
exit /b %LABELIMG_EXIT_CODE%
