"""
=============================================================
  VÍASEGURA — Normas de vIAsegura_ML → RAG del asistente
=============================================================

En el proyecto fusionado, la fuente de verdad de las normas es la carpeta de
resoluciones de vIAsegura_ML:

    vIAsegura_ML/input/resoluciones/
        catalogo_resoluciones.csv   catálogo editable (26 normas, fechas, estado, temas)
        textos/<id_norma>.txt       texto plano extraído (insumo del NLP del ML)

Este script copia esos textos a la carpeta que indexa el RAG del chat
(data/resoluciones/vIAsegura_ML/), con una cabecera que da al RAG el título,
la fuente y el estado de cada norma. Así el chat responde con las mismas normas
que usa el motor de asignación del ML. No modifica nada en vIAsegura_ML.

Omite las normas que el RAG ya tenía como texto SISJUR propio (mismo número de
SISJUR o misma norma en data/resoluciones/semillas.csv), para no duplicar
fragmentos.

Uso
    python src/sincronizar_normas_ml.py            copia / actualiza
    python src/sincronizar_normas_ml.py --limpiar  borra antes la carpeta destino
Después: pestaña Normas → «Indexar pendientes» (o: python src/rag/cli.py indexar).
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                 # vIAsegura_IA
RESOL_ML = os.environ.get("VIASEGURA_RESOLUCIONES_ML") or os.path.join(
    os.path.dirname(RAIZ), "vIAsegura_ML", "input", "resoluciones")
DOCS_RAG = os.path.join(RAIZ, "data", "resoluciones")
DESTINO = os.path.join(DOCS_RAG, "vIAsegura_ML")
TIPOS = {"LEY": "Ley", "DEC": "Decreto", "RES": "Resolución", "ACU": "Acuerdo", "CIR": "Circular"}


def _sisjur(url: str) -> str | None:
    m = re.search(r"sisjur/normas/Norma1\.jsp\?i=(\d+)", url or "")
    return m.group(1) if m else None


def normas_del_rag() -> tuple[set[str], set[str], dict[str, str]]:
    """Números SISJUR que el RAG ya tiene CON TEXTO (SISJUR a veces entrega el archivo vacío: esos no
    cuentan, y se reemplazan con el texto de vIAsegura_ML) y {número SISJUR: nota de semillas.csv}."""
    sys.path.insert(0, os.path.join(RAIZ, "src"))
    from rag.documentos import cargar_documento
    con_texto, existentes = set(), set()
    for raiz, _, archivos in os.walk(DOCS_RAG):
        if os.path.abspath(raiz).startswith(os.path.abspath(DESTINO)):
            continue
        for a in archivos:
            m = re.match(r"sisjur_(\d+)\.", a)
            if m:
                existentes.add(m.group(1))
                rel = os.path.relpath(os.path.join(raiz, a), DOCS_RAG)
                if cargar_documento(DOCS_RAG, rel, {})[0] is not None:
                    con_texto.add(m.group(1))
    notas = {}
    ruta = os.path.join(DOCS_RAG, "semillas.csv")
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8-sig") as f:
            for fila in csv.DictReader(f):
                notas[str(fila.get("id_sisjur", "")).strip()] = (fila.get("nota") or "").lower()
    return con_texto, existentes, notas


def sincronizar(limpiar: bool = False) -> None:
    catalogo = os.path.join(RESOL_ML, "catalogo_resoluciones.csv")
    if not os.path.exists(catalogo):
        sys.exit(f"No encuentro {catalogo}. ¿Está la carpeta vIAsegura_ML junto a vIAsegura_IA?")
    if limpiar and os.path.isdir(DESTINO):
        shutil.rmtree(DESTINO)
    os.makedirs(DESTINO, exist_ok=True)
    con_texto, existentes, notas = normas_del_rag()
    copiadas, omitidas, sin_texto, reemplazan = [], [], [], []
    with open(catalogo, encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))
    for n in filas:
        idn = n["id_norma"]
        tipo = TIPOS.get(idn.split("_")[0], n.get("tipo") or "Norma")
        titulo = f"{tipo} {n.get('numero', '').lstrip('0') or n.get('numero', '')} de {n.get('anio', '')}"
        numero = (n.get("numero") or "").lstrip("0")
        ids = {_sisjur(n.get("url_texto", ""))} | {i for i, nota in notas.items()
                                                    if numero and re.search(rf"\b0*{numero} de {n.get('anio')}\b", nota)}
        ids.discard(None)
        if ids & con_texto:
            omitidas.append(f"{idn} (ya está en el RAG como texto SISJUR: sisjur_{min(ids & con_texto)})")
            continue
        if ids & existentes:
            reemplazan.append(f"{idn} (el texto SISJUR del RAG, sisjur_{min(ids & existentes)}, está vacío)")
        origen = os.path.join(RESOL_ML, "textos", f"{idn}.txt")
        if not os.path.exists(origen):
            sin_texto.append(idn)
            continue
        with open(origen, encoding="utf-8", errors="replace") as fo:
            texto = fo.read()
        url = n.get("url_texto") or ""
        resumen = (n.get("resumen") or "").strip()
        if "RÉGIMEN LEGAL DE BOGOTÁ" in texto:
            # formato SISJUR: el RAG toma el título de la 2.ª línea y el enlace de «FUENTE:»; la cabecera
            # del portal (hasta «permanente actualización.») no entra al índice
            cabecera = f"FUENTE: {url or '(catálogo de vIAsegura_ML)'}\n{titulo} — {resumen}\n"
        else:
            estado = (n.get("estado") or "").strip()
            cabecera = (f"{titulo}. {n.get('resumen', '').strip()}.\n"
                        + (f"Derogada según el catálogo de vIAsegura_ML.\n" if estado == "derogada" else "")
                        + (f"Estado según el catálogo de vIAsegura_ML: {estado}.\n" if estado and estado != "derogada" else "")
                        + (f"FUENTE: {url}\n" if url else "") + "\n")
        destino = os.path.join(DESTINO, f"{idn}.txt")
        nuevo = cabecera + texto
        viejo = open(destino, encoding="utf-8").read() if os.path.exists(destino) else None
        if viejo != nuevo:
            with open(destino, "w", encoding="utf-8") as fd:
                fd.write(nuevo)
            copiadas.append(idn)
    print(f"Normas del catálogo de vIAsegura_ML: {len(filas)}")
    print(f"  copiadas o actualizadas en {os.path.relpath(DESTINO, RAIZ)}: {len(copiadas)}"
          + (f" ({', '.join(copiadas)})" if copiadas else " (ya estaban al día)"))
    for o in omitidas:
        print(f"  omitida: {o}")
    for r in reemplazan:
        print(f"  completa el RAG: {r}")
    for s in sin_texto:
        print(f"  sin texto en vIAsegura_ML (corra allí: python -m src.ingesta.descargar_resoluciones): {s}")
    print("Ahora indéxelas: pestaña Normas → «Indexar pendientes», o python src/rag/cli.py indexar")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limpiar", action="store_true", help="borra la carpeta destino antes de copiar")
    sincronizar(ap.parse_args().limpiar)
