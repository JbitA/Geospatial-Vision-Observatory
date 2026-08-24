@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0showcase.ps1" %*
exit /b %ERRORLEVEL%
