@echo off
setlocal
if "%CONDA_DEFAULT_ENV%" NEQ "zgc3-assistant" (
    call conda activate zgc3-assistant
)
python app.py
pause
