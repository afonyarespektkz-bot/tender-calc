@echo off
chcp 65001 >nul
cd /d %~dp0
call .venv\Scripts\activate.bat
start "" http://localhost:8800
python server.py
pause
