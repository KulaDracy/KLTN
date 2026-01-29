@echo off
cd /d %~dp0
call venv\Scripts\activate.bat
echo Virtual environment activated!
cmd /k