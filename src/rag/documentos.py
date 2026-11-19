"""
1. CARGA DE DOCUMENTOS
Lee las normas de la carpeta de documentos (.txt, .md, .pdf, .docx), quita el
encabezado que agrega el portal SISJUR y extrae metadatos (título, fechas,
si está derogada). Los archivos que SISJUR descargó vacíos («Cargando el
Contenido del Documento…») se omiten y se reportan.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field

EXTENSIONES = (".txt", ".md", ".pdf", ".docx")
MINIMO_CARACTERES = 400        # menos que esto, después de limpiar, se considera vacío


@dataclass
class Documento:
    id: str
    titulo: str
    texto: str
    metadatos: dict = field(default_factory=dict)


def _leer_archivo(ruta: str) -> str:
    ext = os.path.splitext(ruta)[1].lower()
    if ext in (".txt", ".md"):
        with open(ruta, encoding="utf-8", errors="replace") as f:
            return f.read()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(p.extract_text() or "" for p in PdfReader(ruta).pages)
    if ext == ".docx":
        from docx import Document          # pip install python-docx
        return "\n\n".join(p.text for p in Document(ruta).paragraphs)
    return ""


def _campo(texto: str, etiqueta: str) -> str:
    m = re.search(etiqueta + r":\s*\n\s*\n?\s*([^\n]+)", texto)
    valor = m.group(1).strip() if m else ""
    return "" if valor.endswith(":") or valor in ("Temas", "Anexos") else valor


def limpiar_sisjur(bruto: str) -> tuple[str, dict]:
    """Quita el encabezado del portal y une las líneas cortadas a mitad de frase."""
    meta = {}
    m = re.search(r"FUENTE:\s*(\S+)", bruto)
    if m:
        meta["url"] = m.group(1)
    lineas = [l.strip() for l in bruto.splitlines() if l.strip()]
    if len(lineas) > 1 and not lineas[1].startswith("RÉGIMEN"):
        meta["titulo"] = lineas[1]
    meta["fecha_expedicion"] = _campo(bruto, "Fecha de Expedición")
    meta["fecha_vigencia"] = _campo(bruto, "Fecha de Entrada en Vigencia")
    meta["publicacion"] = _campo(bruto, "Medio de Publicación")

    corte = bruto.find("permanente actualización.")
    cuerpo = bruto[corte + len("permanente actualización."):] if corte >= 0 else bruto
    cuerpo = re.sub(r"^FUENTE:.*$", "", cuerpo, flags=re.M)
    d = re.search(r"(Derogad[oa][^\n]*|Compilad[oa][^\n]*)", cuerpo[:600])
    if d:
        meta["estado"] = d.group(1).strip()

    # el portal corta las frases en varias líneas: se une todo y se normalizan espacios
    texto = re.sub(r"\s+", " ", cuerpo)
    texto = re.sub(r"\s+([.,;:)])", r"\1", texto)
    texto = re.sub(r"([(])\s+", r"\1", texto)
    return texto.strip(), meta


def _semillas(carpeta: str) -> dict:
    """Notas de semillas.csv (id_sisjur → descripción), útiles para títulos y para reportar vacíos."""
    ruta = os.path.join(carpeta, "semillas.csv")
    if not os.path.exists(ruta):
        return {}
    with open(ruta, encoding="utf-8-sig") as f:
        return {r["id_sisjur"]: r.get("nota", "") for r in csv.DictReader(f)}


RE_TITULO = re.compile(r"\b(RESOLUCI[ÓO]N|DECRETO|ACUERDO|LEY|CIRCULAR|DIRECTIVA)\s+(?:N[°ºo.]*\s*)?(\d[\d.]*)\s+(?:DE|DEL)\s+((?:19|20)\d{2})",
                       re.I)


def limpiar_pdf(bruto: str) -> tuple[str, dict]:
    """Texto de un PDF o Word: une palabras cortadas con guion y líneas partidas; busca el título de la norma."""
    meta = {}
    texto = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", bruto)          # «seguri-\ndad» → «seguridad»
    texto = re.sub(r"\s+", " ", texto).strip()
    m = RE_TITULO.search(texto[:3000])
    if m:
        meta["titulo"] = f"{m.group(1).capitalize()} {m.group(2)} de {m.group(3)}"
    d = re.search(r"(Derogad[oa][^.]{0,150}\.)", texto[:3000])
    if d:
        meta["estado"] = d.group(1).strip()
    return texto, meta


def listar_archivos(carpeta: str) -> list[str]:
    """Rutas relativas (con «/») de todos los documentos de la carpeta."""
    salida = []
    for raiz, _, archivos in os.walk(carpeta):
        for nombre in sorted(archivos):
            if nombre.lower().endswith(EXTENSIONES):
                salida.append(os.path.relpath(os.path.join(raiz, nombre), carpeta).replace(os.sep, "/"))
    return sorted(salida)


def cargar_documento(carpeta: str, archivo: str, notas: dict | None = None) -> tuple[Documento | None, str | None]:
    """Un documento (ruta relativa a la carpeta) → (Documento, None) o (None, motivo por el que se omite)."""
    notas = _semillas(carpeta) if notas is None else notas
    ruta = os.path.join(carpeta, archivo)
    nombre = os.path.basename(archivo)
    doc_id = os.path.splitext(archivo)[0]
    num = os.path.splitext(nombre)[0].replace("sisjur_", "")
    try:
        bruto = _leer_archivo(ruta)
    except Exception as e:          # PDF dañado, falta python-docx, etc.
        return None, f"no se pudo leer ({type(e).__name__}: {e})"
    if "RÉGIMEN LEGAL DE BOGOTÁ" in bruto or nombre.startswith("sisjur"):
        texto, meta = limpiar_sisjur(bruto)
        vacio = "SISJUR no entregó el texto"
    else:
        texto, meta = limpiar_pdf(bruto)
        vacio = ("el PDF no tiene texto seleccionable (parece escaneado: necesita OCR)"
                 if nombre.lower().endswith(".pdf") else "el archivo casi no tiene texto")
    if len(texto) < MINIMO_CARACTERES:
        return None, vacio + (f" — {notas[num]}" if num in notas else "")
    meta.setdefault("titulo", notas.get(num, "").split(" - ")[0] or os.path.splitext(nombre)[0].replace("_", " "))
    meta["archivo"] = archivo
    meta["mtime"] = os.path.getmtime(ruta)
    return Documento(id=doc_id, titulo=meta["titulo"], texto=texto, metadatos=meta), None


def cargar_documentos(carpeta: str) -> tuple[list[Documento], list[str]]:
    """Devuelve (documentos con contenido, avisos de los archivos omitidos)."""
    notas = _semillas(carpeta)
    docs, omitidos = [], []
    for archivo in listar_archivos(carpeta):
        doc, motivo = cargar_documento(carpeta, archivo, notas)
        if doc:
            docs.append(doc)
        else:
            omitidos.append(f"{archivo}: {motivo}")
    return docs, omitidos
