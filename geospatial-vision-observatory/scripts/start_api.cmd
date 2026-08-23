@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_api.ps1" %*
exit /b %ERRORLEVEL%
