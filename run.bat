@echo off
REM FinSight — start FastAPI (port 8000) and Streamlit (port 8501) simultaneously

echo Starting FastAPI backend on http://localhost:8000 ...
start "FinSight API" cmd /k "cd /d %~dp0 && uvicorn src.api.main:app --port 8000 --reload"

timeout /t 2 /nobreak >nul

echo Starting Streamlit dashboard on http://localhost:8501 ...
start "FinSight UI" cmd /k "cd /d %~dp0 && streamlit run app/streamlit_app.py --server.port 8501"

echo.
echo Both services are starting in separate windows.
echo   API:       http://localhost:8000/docs
echo   Dashboard: http://localhost:8501
