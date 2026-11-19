@echo off
REM ============================================================
REM  Actualiza datos, panel, modelo y efectos del proyecto accidentes_Bogota
REM  Doble clic para correr. El resultado queda en out\log_ejecucion.txt
REM  La descarga salta lo que ya existe, salvo siniestros, camaras y resaltos
REM  (que cambian cada mes). Para bajar todo de nuevo: python src\ingesta_sdm.py --forzar
REM ============================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist out mkdir out
set LOG=out\log_ejecucion.txt
echo Inicio: %date% %time% > %LOG%

set PY=python
where python >nul 2>&1 || set PY=py
%PY% --version >> %LOG% 2>&1 || (echo No se encontro Python. Instale Python 3 y vuelva a intentar. & pause & exit /b 1)

echo [1/9] Dependencias...
echo. >> %LOG% & echo ===== [1/9] DEPENDENCIAS ===== >> %LOG%
%PY% -m pip install --quiet requests pandas numpy scipy openpyxl beautifulsoup4 statsmodels lightgbm pdfplumber huggingface_hub flask >> %LOG% 2>&1

echo [2/9] Descargando datos de Movilidad (siniestros, camaras, resaltos y lo que falte)...
echo. >> %LOG% & echo ===== [2/9] INGESTA SDM ===== >> %LOG%
%PY% src\ingesta_sdm.py >> %LOG% 2>&1

echo [3/9] Panel 2015-hoy desde la base de Movilidad (quita duplicados AA)...
echo. >> %LOG% & echo ===== [3/9] PANEL SDM ===== >> %LOG%
%PY% src\panel_sdm.py >> %LOG% 2>&1

echo [4/9] Modelo de conteo...
echo. >> %LOG% & echo ===== [4/9] MODELO CONTEO ===== >> %LOG%
%PY% src\modelo_conteo.py >> %LOG% 2>&1

echo [5/9] Corredores desde la malla vial...
echo. >> %LOG% & echo ===== [5/9] CORREDORES ===== >> %LOG%
%PY% src\corredores_desde_malla.py >> %LOG% 2>&1

echo [6/9] Catalogo de camaras y resaltos...
echo. >> %LOG% & echo ===== [6/9] CATALOGO DESDE DATOS ===== >> %LOG%
%PY% src\catalogo_desde_datos.py >> %LOG% 2>&1

echo [7/9] Resoluciones de SISJUR...
echo. >> %LOG% & echo ===== [7/9] CATALOGO RESOLUCIONES ===== >> %LOG%
%PY% src\catalogo_resoluciones.py >> %LOG% 2>&1

echo [8/9] Efecto de las medidas (antes y despues con controles; tarda unos minutos)...
echo. >> %LOG% & echo ===== [8/9] EFECTOS CATALOGO ===== >> %LOG%
%PY% src\efecto_intervencion.py --catalogo >> %LOG% 2>&1

echo [9/9] Probando el motor de evaluacion...
echo. >> %LOG% & echo ===== [9/9] EVALUAR MEDIDA ===== >> %LOG%
%PY% src\evaluar_medida.py data\ejemplos\medida_boyaca.json >> %LOG% 2>&1

echo. >> %LOG% & echo Fin: %date% %time% >> %LOG%
echo.
echo Listo. Avisele a Claude.
pause
