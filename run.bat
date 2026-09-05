@echo off
REM One-click setup and start for the Student Performance Prediction app.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv --system-site-packages venv || goto :fail
)

echo Installing dependencies...
venv\Scripts\python.exe -m pip install -q -r requirements.txt || goto :fail

if not exist "model\model.pkl" (
    echo Training the model...
    venv\Scripts\python.exe train_model.py || goto :fail
)

echo.
echo Starting the app on http://127.0.0.1:5000  (press Ctrl+C to stop)
echo.
start http://127.0.0.1:5000
venv\Scripts\python.exe app.py
goto :eof

:fail
echo.
echo Setup failed. Check that Python is installed and on your PATH.
pause
