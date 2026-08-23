@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0worker_once.ps1" %*
exit /b %ERRORLEVEL%
