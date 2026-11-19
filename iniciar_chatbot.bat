@echo off
REM Abre VíaSegura IA (mapa + chat + normas + modelos ML) en el navegador: http://127.0.0.1:5000
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PY=python
where python >nul 2>&1 || set PY=py
%PY% -c "import flask" 2>nul || (echo Instalando Flask... & %PY% -m pip install --quiet flask)
if not exist "out\panel\panel.csv" if exist "despliegue\datos_app.zip" (
  echo Preparando los datos de la app desde despliegue\datos_app.zip (solo la primera vez^)...
  %PY% src\preparar_render.py descomprimir
)
if not exist "out\panel\panel.csv" if exist "..\vIAsegura_ML\input\raw\siniestros\accidentes.csv" (
  echo Construyendo el panel desde la base de siniestros de vIAsegura_ML (solo la primera vez^)...
  %PY% src\panel_sdm.py
)
if not exist "out\panel\panel.csv" (echo Falta el panel. Ejecute primero ejecutar_todo.bat & pause & exit /b 1)
REM Proyecto fusionado: normas de vIAsegura_ML para el RAG del chat (no modifica nada del ML)
if exist "..\vIAsegura_ML\input\resoluciones\catalogo_resoluciones.csv" if not exist "data\resoluciones\vIAsegura_ML" %PY% src\sincronizar_normas_ml.py
if not exist "data\ml\tramos_2026.parquet" if exist "..\vIAsegura_ML\output\intermedios\siniestros_enriquecidos.parquet" (
  echo Preparando los tramos de toda la malla vial para Modelos ML (solo la primera vez, 1-2 minutos^)...
  %PY% src\preparar_tramos_ml.py
)
%PY% -c "import sklearn, joblib" 2>nul || (
  echo Instalando scikit-learn y joblib para la pestana Modelos ML (solo la primera vez^)...
  %PY% -m pip install --quiet "scikit-learn==1.7.2" "lightgbm==4.6.0" joblib
)
findstr /m "GRAVEDAD" "out\panel\siniestros.csv" >nul 2>&1 || (
  echo Actualizando el panel con la base nueva de Movilidad (solo la primera vez^)...
  %PY% src\panel_sdm.py
)
if not exist "data\raw\limites" (
  echo Descargando limites de localidades y barrios (solo la primera vez^)...
  %PY% src\ingesta_sdm.py --solo limites
)
if not exist ".env" if exist "plantilla_env.txt" copy "plantilla_env.txt" ".env" >nul
%PY% -c "import faiss, sentence_transformers, fastembed, anthropic, dotenv" 2>nul || (
  echo Instalando las librerias del RAG de normas (solo la primera vez^)...
  %PY% -m pip install --quiet -r requirements_rag.txt
)
%PY% -c "import pydantic,sys; v=tuple(int(x) for x in pydantic.VERSION.split('.')[:2]); sys.exit(0 if v>=(2,12) or sys.version_info<(3,14) else 1)" 2>nul || (
  echo Actualizando anthropic y pydantic para Python 3.14 (solo una vez^)...
  %PY% -m pip install --quiet -U anthropic pydantic httpx
)
if not exist "out\rag\indice.faiss" (
  echo Indexando las normas de data\resoluciones (solo la primera vez; descarga el modelo de embeddings^)...
  %PY% src\rag\cli.py indexar
)
echo Abriendo ViaSegura IA... (cierre esta ventana para apagarla)
%PY% src\app_web.py
pause
