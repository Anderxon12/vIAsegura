"""
=============================================================
  CATÁLOGO DE INTERVENCIONES — Resoluciones y medidas viales
  Bogotá D.C. · SISJUR (Alcaldía) + documentos locales
=============================================================

Qué hace
    Convierte resoluciones (texto libre) en una tabla estructurada de
    intervenciones viales: qué medida, dónde, desde cuándo y con qué valor.
    Esa tabla es la que permite comparar la siniestralidad antes y después.

    1. Lee la lista de normas de data/resoluciones/semillas.csv
       (columnas: id_sisjur, url, nota). Se llena a mano con las normas de
       interés; la URL de SISJUR tiene la forma
       https://www.alcaldiabogota.gov.co/sisjur/normas/Norma1.jsp?i=<id>
    2. Descarga el texto de cada norma y lo guarda en data/resoluciones/textos/.
       También toma cualquier .txt o .pdf que se copie en esa carpeta.
    3. Extrae los campos con reglas (expresiones regulares) y, si se pide,
       con un modelo de lenguaje de Hugging Face (--llm).
    4. Escribe data/intervenciones/catalogo_intervenciones.csv, una fila por
       norma y corredor, con la columna `revisado` en False.

Uso
    pip install requests pandas beautifulsoup4
    python src/catalogo_resoluciones.py                 # reglas
    python src/catalogo_resoluciones.py --llm           # reglas + LLM
        (requiere: pip install huggingface_hub  y la variable HF_TOKEN)
    python src/catalogo_resoluciones.py --sin-descarga  # solo textos locales

IMPORTANTE
    La extracción es automática y puede equivocarse. Cada fila debe
    revisarse contra el texto original antes de usarla en el modelo; al
    revisarla, cambie `revisado` a True. Las filas revisadas se conservan
    entre ejecuciones.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata

import pandas as pd
import requests

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RES = os.path.join(RAIZ, "data", "resoluciones")
DIR_TXT = os.path.join(DIR_RES, "textos")
DIR_INT = os.path.join(RAIZ, "data", "intervenciones")
SEMILLAS = os.path.join(DIR_RES, "semillas.csv")
SALIDA = os.path.join(DIR_INT, "catalogo_intervenciones.csv")
URL_SISJUR = "https://www.alcaldiabogota.gov.co/sisjur/normas/Norma1.jsp?i={id}"
MODELO_LLM = os.environ.get("MODELO_LLM", "Qwen/Qwen2.5-7B-Instruct")

COLUMNAS = ["id_norma", "fuente", "tipo_norma", "entidad", "numero", "anio", "objeto", "relevante",
            "fecha_expedicion", "tipo_medida",
            "direccion", "valor_anterior", "valor_nuevo", "unidad", "corredor", "localidad",
            "ubicacion_texto", "fecha_inicio", "extracto", "metodo", "confianza", "revisado"]

MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
         "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}

# tipo de medida → palabras que la delatan (sobre texto sin tildes y en minúscula)
TIPOS = {
    "limite_velocidad":   [r"limite[s]? de velocidad", r"velocidad maxima", r"km/h", r"kilometros por hora"],
    "fotodeteccion":      [r"fotodeteccion", r"deteccion electronica", r"camaras? salvavidas", r"\bsast\b", r"fotomulta"],
    "zona_escolar":       [r"zona[s]? escolar", r"entorno[s]? escolar", r"ciempies", r"camino[s]? seguro"],
    "semaforizacion":     [r"semafor", r"interseccion(?:es)? semaforizada"],
    "reductor_velocidad": [r"resalto", r"reductor(?:es)? de velocidad", r"pompeyano", r"banda[s]? sonora"],
    "pico_y_placa":       [r"pico y placa", r"restriccion vehicular"],
    "restriccion_carga":  [r"vehiculos? de carga", r"transporte de carga"],
    "motocicletas":       [r"motociclet", r"parrillero", r"acompanante"],
    "cicloinfraestructura": [r"ciclorruta", r"cicloruta", r"bicicarril", r"ciclovia", r"cicloinfraestructura"],
    "sentido_vial":       [r"sentido[s]? vial", r"cambio de sentido", r"contraflujo", r"reversible"],
    "estacionamiento":    [r"estacionamiento", r"prohibido parquear", r"zona[s]? de parqueo"],
    "control_operativo":  [r"control(?:es)? de embriaguez", r"alcoholemia", r"operativo[s]? de control", r"puesto[s]? de control"],
    "infraestructura":    [r"obra[s]? de infraestructura", r"ampliacion de la via", r"glorieta", r"puente peatonal", r"paso peatonal", r"andenes"],
    "senalizacion":       [r"senalizacion", r"demarcacion"],
}

# corredores principales: nombre canónico → variantes (sin tildes, minúscula)
CORREDORES = {
    # incluye la nomenclatura IDECA (AK = avenida carrera, AC = avenida calle) — verificar con la malla vial
    "Av. Boyacá": [r"boyaca", r"\bak ?72\b(?! ?(?:bis|[a-h]\b))"],
    "Av. Carrera 68": [r"(?:carrera|kr|cra\.?|ak) ?68\b(?! ?(?:bis|[a-h]\b))", r"avenida 68"],
    "Av. Ciudad de Cali": [r"ciudad de cali", r"\bak ?86\b(?! ?(?:bis|[a-h]\b))"], "Av. Primero de Mayo": [r"primero de mayo", r"1 ?de mayo"],
    "Av. Las Américas": [r"las americas", r"avenida americas"], "Av. Suba": [r"avenida suba", r"av\.? suba"],
    "Av. Caracas": [r"caracas", r"\bak ?14\b(?! ?(?:bis|[a-h]\b))"], "Autopista Norte": [r"autopista norte", r"autonorte", r"\bak ?45\b(?! ?(?:bis|[a-h]\b))"],
    "Autopista Sur": [r"autopista sur", r"autosur"], "NQS (Av. Carrera 30)": [r"\bnqs\b", r"(?:carrera|kr|cra\.?|ak) ?30\b(?! ?(?:bis|[a-h]\b))", r"norte quito sur"],
    "Calle 80": [r"calle 80\b(?! ?(?:bis|sur|[a-h]\b))", r"\b(?:cl|ac)\.? ?80\b(?! ?(?:bis|sur|[a-h]\b))", r"autopista medellin"],
    "Calle 26 (Av. El Dorado)": [r"calle 26\b(?! ?(?:bis|sur|[a-h]\b))", r"\bac ?26\b(?! ?(?:bis|sur|[a-h]\b))", r"el dorado"],
    "Calle 13": [r"calle 13\b(?! ?(?:bis|sur|[a-h]\b))", r"\bac ?13\b(?! ?(?:bis|sur|[a-h]\b))", r"avenida centenario", r"av\.? colon"], "Av. Villavicencio": [r"villavicencio"],
    "Av. Circunvalar": [r"circunvalar"], "Carrera Séptima": [r"carrera (?:septima|7)\b", r"(?:kr|cra\.?|ak) ?7\b"],
    "Av. Guayacanes": [r"guayacanes"], "Av. Esperanza": [r"avenida (?:la )?esperanza"],
    "Calle 170": [r"calle 170\b"], "Calle 127": [r"calle 127\b"], "Calle 100": [r"calle 100\b"],
    "Calle 63": [r"calle 63\b", r"jose celestino mutis"], "Av. Bosa": [r"avenida bosa"],
    "Vía Usme": [r"via (?:a )?usme", r"avenida usme"], "Av. Tintal": [r"tintal"], "Av. Cota-Suba": [r"suba ?- ?cota", r"via cota"],
}

LOCALIDADES = ["Antonio Nariño", "Barrios Unidos", "Bosa", "Candelaria", "Chapinero", "Ciudad Bolívar",
               "Engativá", "Fontibón", "Kennedy", "Los Mártires", "Puente Aranda", "Rafael Uribe Uribe",
               "San Cristóbal", "Santa Fe", "Suba", "Sumapaz", "Teusaquillo", "Tunjuelito", "Usaquén", "Usme"]


# ════════════════════════════════════════════════════════════
#  TEXTO
# ════════════════════════════════════════════════════════════
def plano(s: str) -> str:
    """Minúsculas y sin tildes, para comparar."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def html_a_texto(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        sopa = BeautifulSoup(html, "html.parser")
        for t in sopa(["script", "style", "nav", "header", "footer", "form"]):
            t.decompose()
        texto = sopa.get_text("\n")
    except ImportError:
        texto = re.sub(r"<[^>]+>", "\n", html)
    return re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t\xa0]+", " ", texto)).strip()


def es_relleno(texto: str) -> bool:
    """La página de SISJUR sin el contenido de la norma (carga diferida)."""
    return "Cargando el Contenido del Documento" in texto and len(texto) < 3000


def descargar_semillas() -> None:
    os.makedirs(DIR_TXT, exist_ok=True)
    if not os.path.exists(SEMILLAS):
        pd.DataFrame(columns=["id_sisjur", "url", "nota"]).to_csv(SEMILLAS, index=False, encoding="utf-8-sig")
        print(f"Se creó {SEMILLAS}. Agregue las normas de interés y vuelva a ejecutar.")
        return
    sem = pd.read_csv(SEMILLAS, dtype=str).fillna("")
    for _, f in sem.iterrows():
        ident = f["id_sisjur"].strip() or re.sub(r"\W+", "_", f["url"])[-40:]
        url = f["url"].strip() or URL_SISJUR.format(id=ident)
        ruta = os.path.join(DIR_TXT, f"sisjur_{ident}.txt")
        if os.path.exists(ruta) and not es_relleno(open(ruta, encoding="utf-8", errors="replace").read()):
            continue
        texto = None
        for intento in range(4):
            try:
                r = requests.get(url, timeout=60, headers={"User-Agent": "accidentes-bogota-catalogo/1.0 (proyecto academico)"})
                r.raise_for_status()
                r.encoding = r.apparent_encoding or r.encoding
                t = html_a_texto(r.text)
                if not es_relleno(t):
                    texto = t
                    break
            except requests.RequestException as e:
                print(f"   ⚠ {ident}: {e}")
            time.sleep(5 * (intento + 1))   # SISJUR responde "Cargando…" si se le pide muy seguido
        if texto:
            with open(ruta, "w", encoding="utf-8") as fh:
                fh.write(f"FUENTE: {url}\n\n{texto}")
            print(f"   ↓ {ident}")
        else:
            if os.path.exists(ruta):
                os.remove(ruta)
            print(f"   ⚠ {ident}: SISJUR solo devolvió la página de 'Cargando…'. Ábrala en el navegador, "
                  f"copie el texto a data/resoluciones/textos/sisjur_{ident}.txt y vuelva a ejecutar con --sin-descarga.")
        time.sleep(3)


def leer_textos() -> dict[str, str]:
    textos = {}
    for nombre in sorted(os.listdir(DIR_TXT)) if os.path.isdir(DIR_TXT) else []:
        ruta = os.path.join(DIR_TXT, nombre)
        if nombre.lower().endswith(".txt"):
            t = open(ruta, encoding="utf-8", errors="replace").read()
            if not es_relleno(t):
                textos[os.path.splitext(nombre)[0]] = t
        elif nombre.lower().endswith(".pdf"):
            try:
                import pdfplumber
                with pdfplumber.open(ruta) as pdf:
                    textos[os.path.splitext(nombre)[0]] = "\n".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                print(f"   ⚠ {nombre}: instale pdfplumber para leer PDF")
    return textos


# ════════════════════════════════════════════════════════════
#  EXTRACCIÓN CON REGLAS
# ════════════════════════════════════════════════════════════
def _fecha(m) -> str | None:
    try:
        return f"{int(m.group(3)):04d}-{MESES[m.group(2)]:02d}-{int(m.group(1)):02d}"
    except (KeyError, ValueError):
        return None


RE_FECHA = re.compile(r"(\d{1,2})\s*(?:\([^)]*\)\s*)?(?:dias? del mes )?de\s+(" + "|".join(MESES) + r")\s+de(?:l)?\s+(\d{4})")
RE_NORMA = re.compile(r"(resolucion|decreto|acuerdo|circular|ley)\s+(?:n(?:o|umero)?\.?\s*[°º]?\s*)?(\d{1,15})\s+de(?:l)?\s+(?:\d{1,2}\s+de\s+\w+\s+de\s+)?(\d{4})")
# encabezado de SISJUR: "Resolución 1390645 de 2025 Secretaría Distrital de Movilidad"
RE_TITULO = re.compile(r"^(Resoluci[oó]n|Decreto|Acuerdo|Circular|Ley|Directiva)\s+(\d+)\s+de\s+(\d{4})\s+(.+)$", re.M | re.I)
RE_OBJETO = re.compile(r"(?:por (?:medio )?(?:de )?(?:la|el) cual|por la que)\s+([^.;]{10,400}?)(?=[.;]|\n\s*\n|\s(?:EL|LA) (?:SECRETARI|ALCALD|MINISTR)|\sCONSIDERANDO)", re.I)
TERMINOS_VIALES = [p for pats in TIPOS.values() for p in pats] + [r"transito", r"siniestr", r"seguridad vial", r"velocidad", r"\bvial(?:es)?\b"]
RE_VEL = re.compile(r"(\d{2,3})\s*(?:\([^)]*\)\s*)?(?:km\s*/\s*h|kilometros por hora|k\.?p\.?h\.?)")
RE_INICIO = re.compile(r"(?:a partir del?|desde el|rige a partir del?|entrara en vigencia el)\s+(?:dia\s+)?" + RE_FECHA.pattern)


def extraer_reglas(ident: str, texto: str) -> list[dict]:
    t = plano(texto)
    fuente = (re.search(r"FUENTE:\s*(\S+)", texto) or [None, ""])[1]
    base = {c: None for c in COLUMNAS}
    base.update(id_norma=ident, fuente=fuente, metodo="reglas", revisado=False)

    mt = RE_TITULO.search(texto[:1500])
    if mt:      # el encabezado de SISJUR es la fuente más confiable
        base.update(tipo_norma=mt.group(1).capitalize(), numero=mt.group(2), anio=int(mt.group(3)),
                    entidad=mt.group(4).strip())
    else:
        m = RE_NORMA.search(t)
        if m:
            base.update(tipo_norma=m.group(1).capitalize(), numero=m.group(2), anio=int(m.group(3)))
        base["entidad"] = ("Secretaría Distrital de Movilidad" if re.search(r"secretari[ao] distrital de movilidad", t)
                           else "Alcaldía Mayor de Bogotá" if "alcalde" in t or "alcaldia mayor" in t else None)
    mo = RE_OBJETO.search(texto)
    base["objeto"] = re.sub(r"\s+", " ", mo.group(0)).strip()[:400] if mo else None
    obj = plano(base["objeto"] or "")
    fechas = [f for f in (_fecha(x) for x in RE_FECHA.finditer(t)) if f]
    if fechas:                             # la fecha de expedición suele ser del año de la norma
        del_anio = [f for f in fechas if base["anio"] and f.startswith(str(base["anio"]))]
        base["fecha_expedicion"] = (del_anio or fechas)[0]
        md = re.search(r"dad[ao] en bogota[\s\S]{0,40}?" + RE_FECHA.pattern, t)   # "Dada en Bogotá, a los 5 días…"
        if md and _fecha(md):
            base["fecha_expedicion"] = _fecha(md)
    mi = RE_INICIO.search(t)
    if mi:
        base["fecha_inicio"] = f"{int(mi.group(3)):04d}-{MESES[mi.group(2)]:02d}-{int(mi.group(1)):02d}"

    # tipo de medida: el que más coincidencias tiene
    # primero en el objeto de la norma (lo que dice que regula); si no, en todo el texto
    puntajes = {tipo: sum(len(re.findall(p, obj)) for p in pats) for tipo, pats in TIPOS.items()}
    tipo, pts = max(puntajes.items(), key=lambda kv: kv[1])
    if not pts:
        puntajes = {tipo: sum(len(re.findall(p, t)) for p in pats) for tipo, pats in TIPOS.items()}
        tipo, pts = max(puntajes.items(), key=lambda kv: kv[1])
    base["tipo_medida"] = tipo if pts else "otro"
    # relevante: el objeto habla de tránsito / seguridad vial (si no hay objeto, se decide por el texto)
    base["relevante"] = ("si" if any(re.search(p, obj or t[:3000]) for p in TERMINOS_VIALES) else "no")

    # velocidades
    vels = [int(v) for v in RE_VEL.findall(t) if 10 <= int(v) <= 120]
    if vels and base["tipo_medida"] == "limite_velocidad":
        base["unidad"] = "km/h"
        mv = re.search(r"de\s+(\d{2,3})\s*(?:km/h)?\s+a\s+(\d{2,3})\s*km\s*/\s*h", t)
        if mv:
            base.update(valor_anterior=int(mv.group(1)), valor_nuevo=int(mv.group(2)))
        else:
            base["valor_nuevo"] = pd.Series(vels).mode().iloc[0]
        if base["valor_anterior"] and base["valor_nuevo"]:
            base["direccion"] = "reducir" if base["valor_nuevo"] < base["valor_anterior"] else "aumentar"
    if not base["direccion"]:
        for palabra, d in [("reduc", "reducir"), ("disminu", "reducir"), ("aument", "aumentar"), ("increment", "aumentar"),
                           ("prohib", "restringir"), ("restring", "restringir"), ("implement", "implementar"),
                           ("instal", "implementar"), ("derog", "derogar")]:
            if palabra in t:
                base["direccion"] = d
                break

    locs = [l for l in LOCALIDADES if re.search(r"\b" + re.escape(plano(l)) + r"\b", t)]
    base["localidad"] = "; ".join(locs) or None
    corredores = [c for c, pats in CORREDORES.items() if any(re.search(p, t) for p in pats)]

    # extracto: la frase resolutiva más informativa
    mr = re.search(r"(articulo\s+(?:1|primero)[^\n]{0,20}[\s\S]{0,600})", t)
    pos = mr.start() if mr else 0
    base["extracto"] = re.sub(r"\s+", " ", texto[pos:pos + 500]).strip()
    base["ubicacion_texto"] = "; ".join(corredores + locs) or None

    completos = sum(base[c] is not None for c in ["numero", "fecha_expedicion", "tipo_medida", "valor_nuevo", "fecha_inicio"]) + bool(corredores or locs)
    base["confianza"] = "alta" if completos >= 5 else "media" if completos >= 3 else "baja"
    return [dict(base, corredor=c) for c in corredores] or [base]


# ════════════════════════════════════════════════════════════
#  EXTRACCIÓN CON LLM (opcional)
# ════════════════════════════════════════════════════════════
PROMPT_LLM = """Eres un analista jurídico de movilidad en Bogotá. Lee la norma y devuelve SOLO un JSON
con una lista "intervenciones". Cada elemento tiene exactamente estas claves:
"tipo_medida" (uno de: {tipos}, otro), "direccion" (reducir, aumentar, implementar, restringir, derogar),
"valor_anterior" (número o null), "valor_nuevo" (número o null), "unidad" (texto o null),
"corredor" (nombre de la vía o null), "localidad" (texto o null), "ubicacion_texto" (tramo exacto citado o null),
"fecha_inicio" (AAAA-MM-DD o null), "extracto" (cita literal de máximo 300 caracteres que respalda la fila).
Reglas: usa solo lo que dice el texto; si un dato no aparece, escribe null; no inventes vías, fechas ni cifras;
una fila por cada vía o zona afectada. Si la norma no contiene ninguna medida vial, devuelve {{"intervenciones": []}}.

NORMA:
{texto}"""


def extraer_llm(ident: str, texto: str, base: dict) -> list[dict]:
    from huggingface_hub import InferenceClient
    cliente = InferenceClient(model=MODELO_LLM, token=os.environ.get("HF_TOKEN"))
    r = cliente.chat_completion(
        messages=[{"role": "user", "content": PROMPT_LLM.format(tipos=", ".join(TIPOS), texto=texto[:12000])}],
        max_tokens=1500, temperature=0)
    crudo = r.choices[0].message.content
    m = re.search(r"\{[\s\S]*\}", crudo)
    datos = json.loads(m.group(0)) if m else {}
    filas = []
    for it in datos.get("intervenciones", []):
        if not isinstance(it, dict):
            continue
        fila = {c: base.get(c) for c in ["id_norma", "fuente", "entidad", "numero", "anio", "fecha_expedicion"]}
        fila.update({c: it.get(c) for c in ["tipo_medida", "direccion", "valor_anterior", "valor_nuevo", "unidad",
                                            "corredor", "localidad", "ubicacion_texto", "fecha_inicio", "extracto"]})
        # control anti-alucinación: el extracto debe existir literalmente en la norma
        cita = plano(str(fila.get("extracto") or ""))[:80]
        fila["confianza"] = "media" if cita and cita in plano(texto) else "baja"
        fila.update(metodo=f"llm:{MODELO_LLM}", revisado=False)
        filas.append(fila)
    return filas


# ════════════════════════════════════════════════════════════
#  PRINCIPAL
# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Catálogo estructurado de intervenciones viales a partir de resoluciones.")
    ap.add_argument("--llm", action="store_true", help="agrega la extracción con un modelo de Hugging Face")
    ap.add_argument("--sin-descarga", action="store_true", help="no consulta SISJUR; usa solo textos locales")
    args = ap.parse_args()
    os.makedirs(DIR_TXT, exist_ok=True)
    os.makedirs(DIR_INT, exist_ok=True)

    if not args.sin_descarga:
        descargar_semillas()
    textos = leer_textos()
    if not textos:
        print(f"No hay textos en {DIR_TXT}. Llene {SEMILLAS} o copie allí archivos .txt / .pdf.")
        return

    filas = []
    for ident, texto in textos.items():
        reglas = extraer_reglas(ident, texto)
        filas += reglas
        if args.llm:
            try:
                filas += extraer_llm(ident, texto, reglas[0])
            except Exception as e:
                print(f"   ⚠ LLM falló en {ident}: {e}")
    nuevo = pd.DataFrame(filas, columns=COLUMNAS)

    # conserva las filas que una persona ya revisó
    if os.path.exists(SALIDA):
        previo = pd.read_csv(SALIDA, dtype=str)
        revisadas = previo[previo["revisado"].astype(str).str.lower().isin(["true", "1", "si", "sí"])]
        if len(revisadas):
            nuevo = pd.concat([revisadas, nuevo[~nuevo["id_norma"].isin(revisadas["id_norma"])]], ignore_index=True)
    nuevo.to_csv(SALIDA, index=False, encoding="utf-8-sig")

    print(f"\n✅ {len(textos)} normas → {len(nuevo)} filas en {SALIDA}")
    print(nuevo.groupby(["relevante", "tipo_medida"]).size().to_string())
    print("\nNormas y su objeto:")
    for _, f in nuevo.drop_duplicates("id_norma").iterrows():
        anio = int(f["anio"]) if pd.notna(f["anio"]) else "?"
        print(f"  [{f['relevante']}] {f['tipo_norma']} {f['numero']} de {anio} · {str(f['objeto'])[:110]}")
    print("\nRevise cada fila contra el texto original y marque `revisado` = True.")


if __name__ == "__main__":
    main()
