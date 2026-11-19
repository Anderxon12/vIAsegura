@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo === RAG de normas viales de Bogota ===
python -m pip install -q -r requirements_rag.txt
if not exist ".env" copy "plantilla_env.txt" ".env" >nul && echo Se creo .env a partir de plantilla_env.txt: ponga ahi su ANTHROPIC_API_KEY.
if /I "%~1"=="indexar" goto indexar
if not exist "out\rag\indice.faiss" goto indexar
goto chat
:indexar
echo.
echo Indexando documentos de data\resoluciones (la primera vez descarga el modelo de embeddings, ~470 MB)...
python src\rag\cli.py indexar
if errorlevel 1 goto fin
:chat
echo.
python src\rag\cli.py chat
:fin
pause
