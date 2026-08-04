@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0一键启动.ps1" %*
exit /b %ERRORLEVEL%
