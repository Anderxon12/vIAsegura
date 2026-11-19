"""Punto de entrada para servidores (Render): gunicorn --chdir src wsgi:app

Carga datos y normas al arrancar. Debe correr con UN solo proceso (workers=1): los
trabajos de indexación de normas y el estado del RAG viven en memoria.
"""

import app_web

app = app_web.iniciar()
