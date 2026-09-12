@echo off
REM Arayuzu baslatir ve tarayicida acar.
REM Kapatmak icin bu siyah pencereyi kapat ya da Ctrl+C.
cd /d "%~dp0"
echo Arayuz baslatiliyor... tarayici birkac saniye icinde acilacak.
python -m streamlit run app.py --server.port 8501 --server.headless false
pause
