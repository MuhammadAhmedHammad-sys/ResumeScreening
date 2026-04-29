@echo off
cd /d "%~dp0"

echo Starting FastAPI...
start cmd /k uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

echo Starting Streamlit...
start cmd /k streamlit run frontend\streamlit_app.py

pause