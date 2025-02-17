@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python src\uci.py

