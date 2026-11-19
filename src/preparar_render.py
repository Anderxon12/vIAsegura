"""
=============================================================
  VÍASEGURA IA — Datos para el servidor (Render)
=============================================================

La app web necesita archivos que no van a GitHub porque son grandes o se
regeneran (out/panel, data/raw/limites, el índice del RAG...). Este script
los junta, ya recortados, en un solo zip que sí se sube al repositorio, y en
Render los vuelve a poner en su lugar.

En su PC (cada vez que actualice datos, panel o normas):
    python src/preparar_render.py empaquetar
    → despliegue/datos_app.zip   (haga commit y push de ese archivo)

En Render (lo corre el build, ver render.yaml):
    python src/preparar_render.py instalar
    → descomprime el zip y descarga el modelo de embeddings del RAG
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.path.join(RAIZ, "despliegue", "datos_app.zip")
MESES_SINIESTROS = 36            # la app usa los últimos 24 meses completos; se guarda un margen

# Archivos que la app lee al correr (rutas relativas a la raíz del proyecto).
ARCHIVOS = [
    "out/panel/panel.csv",
    "out/panel/panel_vias.csv",
    "data/raw/colegios/0_Colegios.csv",
    "out/rag/indice.faiss",
    "out/rag/fragmentos.jsonl",
    "out/rag/info.json",
]
CARPETAS = ["data/raw/limites"]
SINIESTROS = "out/panel/siniestros.csv"
COLUMNAS_SINIESTROS = ["fecha", "GRAVEDAD", "lat", "lon", "n_heridos", "n_muertos", "via"]


def _siniestros_recortados() -> bytes:
    """Solo las columnas y los meses que usa app_web.py (de ~60 MB a unos pocos)."""
    import pandas as pd
    s = pd.read_csv(os.path.join(RAIZ, SINIESTROS), low_memory=False, encoding="utf-8-sig",
                    usecols=lambda c: c in COLUMNAS_SINIESTROS)
    f = pd.to_datetime(s["fecha"], errors="coerce")
    s = s[f >= f.max() - pd.DateOffset(months=MESES_SINIESTROS)]
    return s.to_csv(index=False).encode("utf-8")


def empaquetar() -> None:
    faltan = [a for a in ARCHIVOS + CARPETAS + [SINIESTROS] if not os.path.exists(os.path.join(RAIZ, a))]
    if faltan:
        sys.exit("Faltan archivos (ejecute antes ejecutar_todo.bat / rag.bat indexar):\n  " + "\n  ".join(faltan))
    os.makedirs(os.path.dirname(ZIP), exist_ok=True)
    tmp = ZIP + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for a in ARCHIVOS:
            z.write(os.path.join(RAIZ, a), a)
        for c in CARPETAS:
            for nombre in sorted(os.listdir(os.path.join(RAIZ, c))):
                z.write(os.path.join(RAIZ, c, nombre), f"{c}/{nombre}")
        z.writestr(SINIESTROS, _siniestros_recortados())
    os.replace(tmp, ZIP)
    with zipfile.ZipFile(ZIP) as z:
        for i in z.infolist():
            print(f"  {i.filename:<45} {i.file_size / 1e6:7.1f} MB → {i.compress_size / 1e6:6.1f} MB")
    print(f"\nListo: {os.path.relpath(ZIP, RAIZ)} ({os.path.getsize(ZIP) / 1e6:.1f} MB). Haga commit y push.")


def instalar() -> None:
    if not os.path.exists(ZIP):
        sys.exit(f"No está {os.path.relpath(ZIP, RAIZ)}. En su PC ejecute: python src/preparar_render.py empaquetar")
    with zipfile.ZipFile(ZIP) as z:
        for i in z.infolist():
            destino = os.path.normpath(os.path.join(RAIZ, i.filename))
            if not destino.startswith(RAIZ + os.sep):           # nada fuera del proyecto
                continue
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            with z.open(i) as origen, open(destino, "wb") as f:
                f.write(origen.read())
            print(f"  {i.filename}")
    print("Descargando el modelo de embeddings del RAG…")
    sys.path.insert(0, os.path.join(RAIZ, "src"))
    from rag.embeddings import _MotorLigero
    onnx, sp = _MotorLigero.descargar()
    print(f"  {os.path.relpath(onnx, RAIZ)}\n  {os.path.relpath(sp, RAIZ)}")
    _MotorLigero()                              # prueba que cargue
    print("Listo.")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["empaquetar", "instalar"])
    {"empaquetar": empaquetar, "instalar": instalar}[ap.parse_args().accion]()
