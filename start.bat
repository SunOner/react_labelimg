@echo off
setlocal DisableDelayedExpansion
set "PYTHONUTF8=1"
pushd "%~dp0" || exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows.ps1"
if errorlevel 1 goto finish
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py" %*
:finish
set "LABELIMG_EXIT_CODE=%errorlevel%"
if not "%LABELIMG_EXIT_CODE%"=="0" if not defined LABELIMG_NO_PAUSE pause
popd
exit /b %LABELIMG_EXIT_CODE%
