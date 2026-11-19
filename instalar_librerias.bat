@echo off
REM Instala todas las librerias de VíaSegura IA (app + datos + RAG) en el MISMO Python que usa iniciar_chatbot.bat
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=python
where python >nul 2>&1 || set PY=py
echo === Python que se va a usar ===
%PY% -c "import sys; print(sys.version.split()[0], sys.executable)"
echo.
echo === Actualizando pip ===
%PY% -m pip install --upgrade pip
echo.
echo === Librerias de la app y del modelo (requirements_app.txt) ===
%PY% -m pip install -U -r requirements_app.txt
if errorlevel 1 goto error
echo.
echo === Librerias del RAG (requirements_rag.txt) ===
%PY% -m pip install -U -r requirements_rag.txt
if errorlevel 1 goto error
echo.
echo === Verificando claves de Claude y Hugging Face ===
%PY% src\entorno.py
echo.
echo Listo. Ahora abra iniciar_chatbot.bat
goto fin
:error
echo.
echo *** Hubo un error instalando. Copie el texto de arriba y envielo. ***
:fin
pause
