"""
=============================================================
  VÍASEGURA IA — Chatbot de medidas viales para Bogotá
=============================================================

Flujo de cada mensaje
    1. FILTRO DE TEMA   ¿El mensaje trata de medidas viales o siniestralidad
                        en Bogotá? Si no, responde que no tiene información
                        para eso y recuerda de qué tema responde.
    2. EXTRACCIÓN       Texto libre → medida estructurada (tipo, dirección,
                        valores, ubicación). Con reglas; si hay HF_TOKEN,
                        un LLM de Hugging Face completa lo que falte.
    3. ACLARACIÓN       Si falta la ubicación o un valor clave, pregunta.
    4. CÁLCULO          evaluar_medida.evaluar(): línea base + evidencia
                        medida en Bogotá + evidencia de referencia.
    5. RESPUESTA        Veredicto, cifras, por qué, recomendaciones y
                        limitaciones. Las cifras salen SOLO del cálculo; si el
                        LLM redacta, se verifica que no cambie ningún número.

Uso
    pip install gradio
    python src/chatbot.py              # abre la interfaz web (http://127.0.0.1:7860)
    python src/chatbot.py --consola    # chat en la terminal
    python src/chatbot.py --probar     # corre los casos de prueba

LLM opcional (mejora la comprensión y la redacción):
    set HF_TOKEN=hf_xxx   y   pip install huggingface_hub
    set MODELO_LLM=Qwen/Qwen2.5-7B-Instruct     (opcional)
"""

from __future__ import annotations

import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catalogo_resoluciones import CORREDORES, LOCALIDADES, TIPOS, plano  # noqa: E402
from efecto_intervencion import cargar_panel, celdas_de_localidad, celdas_de_puntos  # noqa: E402
from evaluar_medida import UMBRAL, _signo, evaluar, nombre_canonico, tipos_con_evidencia  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLEGIOS = os.path.join(RAIZ, "data", "raw", "colegios", "0_Colegios.csv")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entorno import cargar_entorno  # noqa: E402
cargar_entorno()                    # HF_TOKEN y MODELO_LLM desde .env (raíz o src/)
MODELO_LLM = os.environ.get("MODELO_LLM", "Qwen/Qwen2.5-7B-Instruct")

TEMA = "medidas viales y siniestralidad vial en Bogotá"
FUERA_DE_TEMA = (
    f"No tengo información para eso. Solo respondo sobre **{TEMA}**.\n\n"
    "Puedes preguntarme, por ejemplo:\n"
    "- ¿Funcionaría bajar la velocidad a 30 km/h en las zonas escolares de Kennedy?\n"
    "- ¿Qué pasa si suben el límite a 60 km/h en la Autopista Norte?\n"
    "- ¿Sirve poner resaltos en la Avenida Ciudad de Cali?\n"
    "- ¿Cuántos muertos hubo en siniestros viales en Suba en 2024?\n\n"
    "Escribe **ayuda** para ver cómo usar la herramienta.")
FUERA_DE_BOGOTA = ("Solo tengo datos de **Bogotá**, así que no puedo evaluar medidas en otras ciudades. "
                   "Si quieres, dime una vía o localidad de Bogotá y la evalúo.")
MARCA_ACLARACION = "Para evaluarla necesito"

NOMBRES_TIPO = {
    "limite_velocidad": "cambio del límite de velocidad", "fotodeteccion": "cámaras de fotodetección",
    "zona_escolar": "zona escolar", "semaforizacion": "semaforización", "reductor_velocidad": "resaltos o reductores de velocidad",
    "pico_y_placa": "restricción vehicular (pico y placa)", "restriccion_carga": "restricción a vehículos de carga",
    "motocicletas": "medida para motociclistas", "cicloinfraestructura": "cicloinfraestructura",
    "sentido_vial": "cambio de sentido vial", "estacionamiento": "regulación de estacionamiento",
    "control_operativo": "controles de alcoholemia y en vía", "glorieta": "glorieta",
    "paso_peatonal": "cruce peatonal seguro (cebra, isla de refugio, iluminación, fase peatonal)",
    "carril_preferencial": "carril preferencial para buses (SITP)",
    "infraestructura": "otra obra de infraestructura", "senalizacion": "señalización y demarcación",
}
# Dónde actúa cada medida. Una medida puntual (una glorieta) o de tramo (una ciclorruta)
# no cambia los siniestros de toda una localidad: su porcentaje aplica a los siniestros
# del punto o del tramo intervenido.
ALCANCE = {"glorieta": "puntual", "semaforizacion": "puntual", "paso_peatonal": "puntual", "fotodeteccion": "puntual",
           "reductor_velocidad": "tramo", "cicloinfraestructura": "tramo", "carril_preferencial": "tramo",
           "senalizacion": "tramo", "sentido_vial": "tramo", "infraestructura": "tramo", "estacionamiento": "tramo"}
ZONAS_VALIDAS = {"puntual": {"punto"}, "tramo": {"punto", "via", "corredor"}}
ALCANCE_TEXTO = {"puntual": ("las intersecciones o puntos intervenidos", "los puntos (con su radio)"),
                 "tramo": ("los tramos de vía intervenidos", "los tramos de vía")}

# palabras extra para reconocer medidas en lenguaje cotidiano (sin tildes, minúscula)
TIPOS_CHAT = {
    "limite_velocidad": [r"velocidad", r"\b\d{2,3} ?(?:km|kph|kilometros)"],
    "fotodeteccion": [r"camaras?\b", r"fotomultas?", r"radar(?:es)?"],
    "zona_escolar": [r"colegios?", r"escuelas?", r"escolar(?:es)?"],
    "semaforizacion": [r"semaforos?"],
    "reductor_velocidad": [r"policias? acostados?", r"reductores?", r"resaltos?"],
    "control_operativo": [r"polici", r"agentes de transito", r"retenes?", r"controles?", r"alcoholimetr", r"operativos?"],
    "motocicletas": [r"\bmotos?\b", r"motociclistas?"],
    "cicloinfraestructura": [r"ciclorrutas?", r"ciclovias?", r"bicicarril", r"bicicletas?"],
    "glorieta": [r"glorietas?", r"rotondas?"],
    "paso_peatonal": [r"cruces? peatonal(?:es)?", r"pasos? peatonal(?:es)?", r"cebras?", r"islas? de refugio",
                      r"fase peatonal", r"senderos? peatonal"],
    "carril_preferencial": [r"carril(?:es)? (?:preferencial|exclusivo|solo bus|para buses)", r"carril(?:es)? del? sitp",
                            r"carril(?:es)? de buses"],
    "infraestructura": [r"puentes? peatonal", r"andenes", r"ampliar la via"],
    "senalizacion": [r"senales", r"senalizar", r"demarcar"],
    "pico_y_placa": [r"pico y placa"],
}
TERMINOS_TEMA = [r"siniestr", r"accident", r"choques?\b", r"atropell", r"seguridad vial", r"transito", r"trafico",
                 r"muert[oe]s? en (?:la )?via", r"\bvias?\b", r"avenidas?", r"\bvial(?:es)?\b", r"conductor", r"peaton",
                 r"movilidad", r"heridos?", r"lesionados?", r"fallecidos?", r"muertos?", r"muertes?"]
OTRAS_CIUDADES = [r"medellin\b(?! ?\()", r"barranquilla", r"cartagena", r"bucaramanga", r"pereira", r"manizales",
                  r"cucuta", r"ibague", r"santa marta", r"soacha", r"chia\b", r"en cali\b", r"de cali\b(?<!ciudad de cali)"]


# ════════════════════════════════════════════════════════════
#  1–2. FILTRO DE TEMA Y EXTRACCIÓN CON REGLAS
# ════════════════════════════════════════════════════════════
def _tipo(t: str) -> str | None:
    puntajes = {}
    for tipo in NOMBRES_TIPO:
        pats = TIPOS.get(tipo, []) + TIPOS_CHAT.get(tipo, [])
        puntajes[tipo] = sum(len(re.findall(p, t)) for p in pats)
    # prioridades: zona escolar + velocidad → se evalúa como límite en entorno escolar
    tipo, pts = max(puntajes.items(), key=lambda kv: kv[1])
    if not pts:
        return None
    # un cambio explícito de velocidad ("de 50 a 60", "a 30 km/h") siempre es un cambio de límite;
    # las cámaras o la policía que lo acompañen quedan como acompañamiento
    if re.search(r"de \d{2,3}(?: ?km/?h)? a \d{2,3}\b", t) or \
            (re.search(r"velocidad|limite", t) and re.search(r"\b\d{2,3} ?(?:km|kph|kilometros)", t)):
        return "limite_velocidad"
    for preferido in ("reductor_velocidad", "fotodeteccion", "semaforizacion", "control_operativo"):
        if puntajes[preferido] and preferido != tipo and puntajes[preferido] >= puntajes[tipo] - 1 \
                and tipo in ("limite_velocidad", "senalizacion"):
            # "poner cámaras de velocidad" es fotodetección, no un cambio de límite
            if not re.search(r"\b\d{2,3} ?(?:km|kph|kilometros)", t):
                return preferido
    if puntajes["zona_escolar"] and puntajes["limite_velocidad"]:
        return "limite_velocidad"
    return tipo


def _corredor(t: str) -> str | None:
    t = re.sub(r"\bavenida ciudad de cali\b", "ciudad de cali", t)
    for canon, pats in CORREDORES.items():
        if any(re.search(p, t) for p in pats):
            return canon
    # nombres comunes que no están en los patrones del catálogo
    extra = {"Av. Carrera 68": r"\bla 68\b", "NQS (Av. Carrera 30)": r"\bla 30\b", "Calle 80": r"\bla 80\b",
             "Calle 26 (Av. El Dorado)": r"\bla 26\b", "Carrera Séptima": r"\bla septima\b|\bla 7(?:ma)?\b",
             "Av. Suba": r"\bavenida suba\b|\bav\.? suba\b", "Av. Las Américas": r"\bamericas\b",
             "Av. Boyacá": r"\bla boyaca\b", "Av. Caracas": r"\bla caracas\b", "Calle 13": r"\bla 13\b"}
    for canon, pat in extra.items():
        if re.search(pat, t):
            return canon
    return None


def _localidad(t: str, corredor: str | None) -> str | None:
    for loc in LOCALIDADES:
        p = plano(loc)
        if re.search(r"\b" + re.escape(p) + r"\b", t):
            # "avenida suba" / "via usme" son corredores, no la localidad
            if corredor and p in plano(corredor):
                if not re.search(r"(?:localidad|barrio|en|de) " + re.escape(p) + r"\b", t.replace("avenida " + p, "")):
                    continue
            return loc
    return None


def extraer_reglas(texto: str) -> dict:
    t = plano(texto)
    m = {"tipo_medida": _tipo(t), "direccion": None, "valor_actual": None, "valor_propuesto": None,
         "corredor": None, "localidad": None, "entorno_escolar": bool(re.search(r"colegio|escuela|escolar", t)),
         "acompanamiento": [], "anio": None}
    m["corredor"] = _corredor(t)
    m["localidad"] = _localidad(t, m["corredor"])
    m["toda_la_ciudad"] = bool(re.search(r"toda (?:la ciudad|bogota)|todo bogota|en bogota\b|de bogota\b|la ciudad", t)) \
        and not m["corredor"] and not m["localidad"]

    # velocidades: "de 60 a 50", "a 30 km/h", "50 km/h"
    par = re.search(r"de (\d{2,3})(?: ?km/?h)? a (\d{2,3})", t)
    vels = [int(v) for v in re.findall(r"(\d{2,3}) ?(?:km ?/? ?h|kph|kilometros)", t) if 10 <= int(v) <= 120]
    if par:
        m["valor_actual"], m["valor_propuesto"] = int(par.group(1)), int(par.group(2))
    elif vels:
        m["valor_propuesto"] = vels[-1]
        if len(vels) > 1:
            m["valor_actual"] = vels[0]
    if not m["valor_propuesto"] and re.search(r"velocidad|limite|\bbaj|\bsub|\breduc|\baument", t):
        suelto = re.search(r"\ba (?:los |las )?(\d{2,3})\b(?! ?(?:de |%|anos|metros|cuadras|colegios))", t)   # "bajar a 30"
        if suelto and 10 <= int(suelto.group(1)) <= 120:
            m["valor_propuesto"] = int(suelto.group(1))
            if m["tipo_medida"] in (None, "zona_escolar", "senalizacion"):
                m["tipo_medida"] = "limite_velocidad"
    for pal, d in [("reduc", "reducir"), ("baj", "reducir"), ("disminu", "reducir"), ("sub", "aumentar"),
                   ("aument", "aumentar"), ("increment", "aumentar"), ("elev", "aumentar"),
                   ("quit", "retirar"), ("retir", "retirar"), ("elimin", "retirar"), ("desmont", "retirar"),
                   ("pon", "implementar"), ("instal", "implementar"), ("coloc", "implementar"),
                   ("implement", "implementar"), ("constru", "implementar"), ("prohib", "restringir")]:
        if re.search(r"\b" + pal, t):
            m["direccion"] = d
            break
    if m["valor_actual"] and m["valor_propuesto"]:
        m["direccion"] = "reducir" if m["valor_propuesto"] < m["valor_actual"] else "aumentar"
    if m["tipo_medida"] != "fotodeteccion" and re.search(r"camara|fotomulta|fotodeteccion|radar", t):
        m["acompanamiento"].append("fotodeteccion")
    if m["tipo_medida"] != "control_operativo" and re.search(r"polici|agentes", t):
        m["acompanamiento"].append("control_policial")
    a = re.search(r"\b(20(?:1[5-9]|2\d))\b", t)
    m["anio"] = int(a.group(1)) if a else None

    m["en_tema"] = bool(m["tipo_medida"] or m["corredor"] or any(re.search(p, t) for p in TERMINOS_TEMA))
    m["otra_ciudad"] = any(re.search(p, t) for p in OTRAS_CIUDADES) and "bogota" not in t
    m["consulta_datos"] = bool(re.search(r"cuant[oa]s|cifras?|estadistic|numero de|total de|datos de", t)
                               and re.search(r"muert|fallecid|herid|lesionad|siniestr|accident|choque", t))
    return m


# ════════════════════════════════════════════════════════════
#  2b. EXTRACCIÓN CON LLM (opcional)
# ════════════════════════════════════════════════════════════
PROMPT_EXTRACTOR = """Eres el extractor de un asistente de seguridad vial para Bogotá. Devuelve SOLO un JSON con estas claves:
"en_tema": true si el mensaje trata de medidas viales, tránsito o siniestralidad vial; false si es cualquier otro tema.
"tipo_medida": uno de {tipos} o null.
"direccion": "reducir", "aumentar", "implementar", "retirar", "restringir" o null.
"valor_actual": número o null. "valor_propuesto": número o null (por ejemplo km/h).
"corredor": nombre de la vía principal mencionada o null. "localidad": localidad de Bogotá o null.
"entorno_escolar": true/false.
No inventes datos que el mensaje no dice. No respondas la pregunta; solo extrae.

Mensaje: {mensaje}"""


def cliente_llm():
    if not os.environ.get("HF_TOKEN"):
        return None
    try:
        from huggingface_hub import InferenceClient
        return InferenceClient(model=MODELO_LLM, token=os.environ["HF_TOKEN"])
    except Exception:
        return None


def extraer_llm(texto: str, cli) -> dict:
    try:
        r = cli.chat_completion(messages=[{"role": "user", "content": PROMPT_EXTRACTOR.format(
            tipos=", ".join(NOMBRES_TIPO), mensaje=texto[:1500])}], max_tokens=300, temperature=0)
        mjson = re.search(r"\{[\s\S]*\}", r.choices[0].message.content)
        d = json.loads(mjson.group(0)) if mjson else {}
    except Exception:
        return {}
    limpio = {}
    if isinstance(d.get("en_tema"), bool):
        limpio["en_tema"] = d["en_tema"]
    if d.get("tipo_medida") in NOMBRES_TIPO:
        limpio["tipo_medida"] = d["tipo_medida"]
    if d.get("direccion") in ("reducir", "aumentar", "implementar", "retirar", "restringir"):
        limpio["direccion"] = d["direccion"]
    for k in ("valor_actual", "valor_propuesto"):
        if isinstance(d.get(k), (int, float)) and 0 < d[k] <= 200:
            limpio[k] = d[k]
    if isinstance(d.get("corredor"), str):
        c = nombre_canonico(d["corredor"])
        if c in CORREDORES:                       # solo vías que el sistema conoce
            limpio["corredor"] = c
    if isinstance(d.get("localidad"), str):
        loc = next((x for x in LOCALIDADES if plano(x) == plano(d["localidad"])), None)
        if loc:
            limpio["localidad"] = loc
    return limpio


def extraer(texto: str, cli=None) -> dict:
    m = extraer_reglas(texto)
    if cli is not None:
        ll = extraer_llm(texto, cli)
        for k, v in ll.items():                   # el LLM completa lo que las reglas no vieron
            if k == "en_tema":
                m["en_tema"] = m["en_tema"] or v  # nunca saca del tema algo que las reglas reconocen
            elif m.get(k) in (None, "", []):
                m[k] = v
    return m


# ════════════════════════════════════════════════════════════
#  CONSULTAS DE DATOS ("¿cuántos muertos hubo en Suba en 2024?")
# ════════════════════════════════════════════════════════════
def _miles(x) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


def responder_datos(m: dict, panel: pd.DataFrame) -> str:
    d, zona = panel, "Bogotá"
    if m.get("localidad"):
        d = d[d["localidad"].astype(str).map(plano) == plano(m["localidad"])]
        zona = f"la localidad de {m['localidad']}"
    por_anio = d.groupby("anio")[["n_sin_victimas", "n_heridos", "n_muertos"]].sum()
    ultimo_mes = panel["mes"].max()
    if m.get("anio") and m["anio"] in por_anio.index:
        f = por_anio.loc[m["anio"]]
        parcial = f" (datos hasta {ultimo_mes.strftime('%m/%Y')})" if m["anio"] == ultimo_mes.year and ultimo_mes.month < 12 else ""
        return (f"En {zona}, en **{m['anio']}**{parcial} se registraron **{_miles(f.n_sin_victimas)} siniestros con víctimas**, "
                f"con **{_miles(f.n_heridos)} heridos** y **{_miles(f.n_muertos)} muertos**.\n\n"
                "*Fuente: base de siniestralidad de la Secretaría Distrital de Movilidad, depurada de duplicados; "
                "solo siniestros con coordenadas válidas.*")
    filas = "\n".join(f"| {a} | {_miles(f.n_sin_victimas)} | {_miles(f.n_heridos)} | {_miles(f.n_muertos)} |"
                      for a, f in por_anio.tail(6).iterrows())
    return (f"Siniestralidad vial en {zona} (último dato: {ultimo_mes.strftime('%m/%Y')}):\n\n"
            "| Año | Siniestros con víctimas | Heridos | Muertos |\n|---|---|---|---|\n" + filas +
            "\n\n*Fuente: base de siniestralidad de la Secretaría Distrital de Movilidad, depurada de duplicados.*")


# ════════════════════════════════════════════════════════════
#  3–4. ACLARACIÓN Y CÁLCULO
# ════════════════════════════════════════════════════════════
def faltantes(m: dict) -> list[str]:
    f = []
    if not (m.get("corredor") or m.get("localidad") or m.get("toda_la_ciudad")):
        f.append("**dónde** se aplicaría: una vía principal (por ejemplo, Av. Boyacá) o una localidad (por ejemplo, Kennedy)")
    if m["tipo_medida"] == "limite_velocidad" and not m.get("valor_propuesto"):
        f.append("**a qué velocidad** quedaría el límite (por ejemplo, 30 km/h)")
    return f


def a_medida(m: dict, panel: pd.DataFrame) -> tuple[dict, list[str]]:
    """Medida del chat → formato de evaluar_medida, con los supuestos que se hicieron."""
    notas = []
    med = {"tipo_medida": m["tipo_medida"], "direccion": m.get("direccion"), "acompanamiento": m.get("acompanamiento") or [],
           "valor_actual": m.get("valor_actual"), "valor_propuesto": m.get("valor_propuesto"), "velocidad_media_actual": None}
    med["entorno_escolar"] = bool(m.get("entorno_escolar")) or med["tipo_medida"] == "zona_escolar"
    if med["tipo_medida"] == "zona_escolar":
        med["tipo_medida"] = "limite_velocidad"
        if not med["valor_propuesto"]:
            med["valor_propuesto"] = 30
            notas.append("Tomé la zona escolar como un límite de 30 km/h, como lo define el Decreto 073 de 2021.")
    if med["tipo_medida"] == "limite_velocidad" and not med["valor_actual"]:
        med["valor_actual"] = 30 if (m.get("entorno_escolar") and (med["valor_propuesto"] or 0) > 30) else 50
        notas.append(f"Supuse que el límite actual es {med['valor_actual']} km/h (Decreto 073 de 2021), porque no lo indicaste.")
    if med["tipo_medida"] == "limite_velocidad" and med["valor_actual"] and med["valor_propuesto"]:
        med["direccion"] = "reducir" if med["valor_propuesto"] < med["valor_actual"] else "aumentar"
    if not med["direccion"]:
        med["direccion"] = "implementar"

    if m.get("corredor"):
        med["ubicacion"] = {"nivel": "corredor", "nombre": m["corredor"]}
        med["_tipos_zona"] = {"corredor"}
    elif m.get("localidad"):
        med["ubicacion"] = {"nivel": "localidad", "nombre": m["localidad"]}
        med["_tipos_zona"] = {"localidad"}
    else:
        med["_tipos_zona"] = {"ciudad"}
        med["ubicacion"] = {"nivel": "celdas", "ids": panel["celda"].unique().tolist()}
        med["_zona"] = "toda Bogotá"
        notas.append("Evalué la medida para toda la ciudad. Los efectos se midieron en avenidas y tramos concretos; "
                     "aplicarlos a toda la ciudad es una extrapolación.")

    # entorno escolar: solo las celdas cercanas a colegios de la localidad
    if m.get("entorno_escolar") and m.get("localidad") and not m.get("corredor") and os.path.exists(COLEGIOS):
        col = pd.read_csv(COLEGIOS, usecols=["lon", "lat"]).dropna()
        cerca = set(celdas_de_puntos(panel, col[["lon", "lat"]].values.tolist(), 150))
        ids = [c for c in celdas_de_localidad(panel, m["localidad"]) if c in cerca]
        if ids:
            med["ubicacion"] = {"nivel": "celdas", "ids": ids}
            med["_zona"] = f"entornos escolares de {m['localidad']} ({len(ids)} zonas de 500 m con colegios)"
            notas.append("Tomé como zona escolar las áreas a menos de unos 400 m de un colegio de la localidad.")
    return med, notas


# ════════════════════════════════════════════════════════════
#  5. RESPUESTA
# ════════════════════════════════════════════════════════════
def _pct(x) -> str:
    return f"{x:+.1f} %".replace(".", ",")


def _rango(r) -> str:
    return f"entre {_pct(r[0])} y {_pct(r[1])}"


def recomendaciones(med: dict, res: dict) -> list[str]:
    t, ver, rec = med["tipo_medida"], res.get("veredicto"), []
    if t == "limite_velocidad" and med.get("direccion") == "aumentar":
        rec.append("Si el objetivo es reducir tiempos de viaje, es más seguro mejorar la coordinación de semáforos y mantener el límite actual.")
    if t == "limite_velocidad" and med.get("direccion") == "reducir":
        rec.append("Un límite más bajo sin control cambia poco la velocidad real. En Bogotá, los resaltos parabólicos "
                   "son la medida con el efecto más claro (cerca de −20 % en siniestros con víctimas).")
    if t == "fotodeteccion":
        rec.append("En los datos de Bogotá, las cámaras por sí solas no muestran un cambio medible en los siniestros "
                   "de su entorno inmediato. Conviene combinarlas con medidas físicas, como resaltos, en los tramos críticos.")
    if t == "reductor_velocidad" and ver == "Positiva":
        rec.append("Priorizar los tramos con más siniestros graves y medir el resultado a los 12 meses, como hizo la ciudad en la Av. Boyacá y la Av. Guayacanes.")
    if t == "semaforizacion":
        rec.append("Un semáforo cambia el tipo de choque: reduce los choques en ángulo, pero aumenta los alcances. Conviene "
                   "revisar qué tipo de siniestro domina en la intersección; si son atropellos, un cruce peatonal seguro "
                   "o una fase peatonal adelantada atacan el problema más directo.")
    if t == "paso_peatonal":
        rec.append("El efecto se concentra en los atropellos: priorizar los cruces con más peatones heridos y combinar "
                   "demarcación visible con isla de refugio e iluminación.")
    if t == "glorieta":
        rec.append("La evidencia viene de intersecciones convertidas en glorieta moderna (de un carril y baja velocidad de "
                   "entrada); en intersecciones grandes de varios carriles el efecto suele ser menor.")
    if t == "control_operativo":
        rec.append("Los controles de alcoholemia solo actúan sobre los siniestros con alcohol, que son una parte pequeña del "
                   "total registrado. Su efecto crece cuando los controles son frecuentes, aleatorios y bien publicitados.")
    if t == "cicloinfraestructura":
        rec.append("La mayor reducción se da con carriles separados físicamente (con delineadores o bordillo), no solo pintados.")
    if t == "carril_preferencial":
        rec.append("En Bogotá, los carriles preferenciales SITP de 2014–2017 no cambiaron de forma medible los siniestros "
                   "con víctimas; su beneficio es de tiempos de viaje, no de seguridad vial.")
    if ver in ("Neutra", "Negativa") and t not in ("reductor_velocidad",) and not rec:
        rec.append("Considerar medidas con efecto comprobado en Bogotá, como los resaltos parabólicos en tramos críticos.")
    return rec


def _rango_corto(r) -> str:
    return f"{_pct(r[0])} a {_pct(r[1])}"


def _lado(r) -> str:
    inf, sup = r
    if sup < 0:
        return "todo el rango probable está por debajo de cero"
    if inf > 0:
        return "todo el rango probable está por encima de cero"
    return "el rango probable cruza el cero (incluye tanto reducciones como aumentos)"


def explicar_veredicto(res: dict) -> str:
    """Por qué salió Positiva / Negativa / Neutra, en una o dos frases."""
    c, r = res.get("cambio_sin_victimas_pct"), res.get("rango_pct")
    cm, rm = res.get("cambio_muertes_pct"), res.get("rango_muertes_pct")
    s1 = _signo(c / 100, [x / 100 for x in r]) if c is not None else 0
    s2 = _signo(cm / 100, [x / 100 for x in rm]) if cm is not None else 0
    partes = []
    for nombre, cc, rr, ss in (("siniestros con víctimas", c, r, s1), ("muertes", cm, rm, s2)):
        if cc is None:
            continue
        if ss == 1:
            partes.append(f"en {nombre} baja claramente: {_pct(cc)}, y {_lado(rr)}")
        elif ss == -1:
            partes.append(f"en {nombre} sube claramente: {_pct(cc)}, y {_lado(rr)}")
        elif abs(cc) < 100 * UMBRAL:
            partes.append(f"en {nombre} el cambio esperado ({_pct(cc)}) es menor al 5 %, demasiado pequeño para contarlo")
        else:
            partes.append(f"en {nombre} el cambio esperado es {_pct(cc)}, pero el rango probable ({_rango_corto(rr)}) "
                          "incluye tanto reducciones como aumentos, así que no es claro")
    regla = (" Cuento un efecto como claro cuando el cambio es de al menos 5 % y el rango probable no cruza el cero.")
    return ("Porque " + "; ".join(partes) + "." + regla) if partes else ""


def explicar_confianza(res: dict) -> str:
    """Por qué la confianza es alta, media o baja."""
    evs = res.get("evidencia", [])
    n_local = sum(e["tipo"] == "local" for e in evs)
    n_ref = sum(e["tipo"] == "referencia" for e in evs)
    r = res.get("rango_pct") or [0, 0]
    ancho = r[1] - r[0]
    sup_vel = any("velocidad media" in str(x) for x in res.get("supuestos", []))
    nivel = res.get("confianza", "baja")
    if nivel == "alta":
        return f"hay {n_local} estimaciones medidas en Bogotá y el rango probable es estrecho ({ancho:.0f} puntos)."
    razones = []
    if n_local == 0:
        razones.append("no hay casos medidos en Bogotá: se basa solo en estudios de otros países" if n_ref else
                       "no hay casos medidos en Bogotá")
    elif n_local == 1:
        razones.append("hay una sola estimación medida en Bogotá")
    if ancho >= 40:
        razones.append(f"el rango probable es muy amplio ({ancho:.0f} puntos)")
    elif ancho >= 20:
        razones.append(f"el rango probable es amplio ({ancho:.0f} puntos)")
    loc = [e["cambio"] for e in evs if e["tipo"] == "local"]
    ref = [e["cambio"] for e in evs if e["tipo"] == "referencia"]
    if loc and ref and abs(sum(loc) / len(loc) - sum(ref) / len(ref)) >= 0.10:
        razones.append(f"lo medido en Bogotá ({_pct(100 * sum(loc) / len(loc))}) y los estudios internacionales "
                       f"({_pct(100 * sum(ref) / len(ref))}) no coinciden, y esa diferencia amplía el rango")
    if sup_vel:
        razones.append("depende de supuestos no medidos, como la velocidad real que hoy tienen los vehículos")
    if not razones:
        razones.append(f"hay evidencia, pero el rango probable es de {ancho:.0f} puntos")
    return "; ".join(razones) + "."


def _corto(e: dict) -> str:
    f = e["fuente"]
    if e["tipo"] == "local":
        partes = f.replace("Bogotá · ", "").split(" · ")
        txt = partes[0]
        if "reescalado" in f:
            txt += ", ajustado al cambio de límite que planteas"
        return f"Medido en Bogotá — {txt}"
    if ":" in f:
        desc, resto = f.split(":", 1)
        autor = re.search(r"([A-ZÁÉÍÓÚ][\w .,&-]*?\(\d{4}\))", resto)
        return f"Estudio internacional — {desc.strip()}" + (f" ({autor.group(1)})" if autor else "")
    return f"Estudio internacional — {f}"


def redactar(med: dict, res: dict, notas: list[str]) -> str:
    nombre = NOMBRES_TIPO.get(med["tipo_medida"], med["tipo_medida"])
    zona = med.get("_zona") or res.get("zona", "")
    if med["tipo_medida"] == "limite_velocidad" and med.get("valor_propuesto"):
        nombre = f"límite de {med['valor_actual']:g} → {med['valor_propuesto']:g} km/h"
        if med.get("acompanamiento"):
            nombre += " con " + " y ".join({"fotodeteccion": "cámaras", "control_policial": "control policial"}.get(a, a)
                                            for a in med["acompanamiento"])
    if res.get("veredicto") == "Sin evaluar":
        return ("No pude ubicar esa zona en mis datos. Dime una vía principal (por ejemplo, Av. Boyacá, Calle 80, NQS) "
                "o una localidad de Bogotá y vuelvo a intentarlo.")
    base = res.get("linea_base", {})
    if res.get("cambio_sin_victimas_pct") is None:
        L = [f"**Sin evidencia suficiente** · {nombre} · {zona}",
             f"Todavía no tengo evidencia, ni de Bogotá ni de otros países, sobre el efecto de **{nombre}**, "
             "así que prefiero no darte un porcentaje inventado."]
        if base:
            L.append(f"**En la zona:** {_miles(base['sin_victimas_anual'])} siniestros con víctimas y "
                     f"{_miles(base['muertos_anual'])} muertes al año.")
        L.append("Escribe **ayuda** para ver qué medidas sí puedo evaluar.")
        return "\n\n".join(L)

    icono = {"Positiva": "🟢", "Negativa": "🔴", "Neutra": "🟡"}.get(res["veredicto"], "")
    L = [f"{icono} **{res['veredicto']}** · {nombre} · {zona}",
         f"**¿Por qué {res['veredicto'].lower()}?** {explicar_veredicto(res)}"]
    tabla = ["| | Cambio esperado | Rango probable |", "|---|---|---|",
             f"| Siniestros con víctimas | **{_pct(res['cambio_sin_victimas_pct'])}** | {_rango_corto(res['rango_pct'])} |"]
    if res.get("cambio_muertes_pct") is not None:
        tabla.append(f"| Siniestros fatales | **{_pct(res['cambio_muertes_pct'])}** | {_rango_corto(res['rango_muertes_pct'])} |")
    L.append("\n".join(tabla))
    alcance = ALCANCE.get(med["tipo_medida"])
    zona_ok = not alcance or not med.get("_tipos_zona") or med["_tipos_zona"] <= ZONAS_VALIDAS[alcance]
    if base and not zona_ok:
        donde, marcar = ALCANCE_TEXTO[alcance]
        L.append(f"**Alcance:** actúa solo en {donde}; el porcentaje aplica a sus siniestros, no a toda la zona. "
                 f"Para calcular cuántos se evitarían, marca en el mapa {marcar}.")
    elif base:
        ev = res.get("siniestros_victimas_evitados_anual") or 0
        signo = "menos" if ev >= 0 else "más"
        L.append(f"**En la zona:** {_miles(base['sin_victimas_anual'])} siniestros con víctimas y "
                 f"{_miles(base['muertos_anual'])} muertes al año → unos **{_miles(abs(ev))} siniestros con víctimas {signo}** al año.")
    L.append(f"**Confianza {res['confianza']}:** {explicar_confianza(res)[0].upper() + explicar_confianza(res)[1:]}")
    rec = recomendaciones(med, res)
    if rec:
        L.append(f"**Recomendación:** {rec[0]}")

    det = ["[[detalles]]", "**En qué me baso** (siniestros con víctimas)"]
    for e in res.get("evidencia", []):
        det.append(f"- {_corto(e)}: {_pct(100 * e['cambio'])} ({_rango_corto([100 * x for x in e['rango']])})")
    lim = notas + [a.replace("Supuesto: ", "") for a in res.get("advertencias", [])]
    lim = [re.sub(r"\{'sin_control'.*?\}", "25 % sin control y 50 % con control", x) for x in lim]
    lim = list(dict.fromkeys(lim)) + rec[1:] + ["Es una estimación de apoyo; no reemplaza un estudio técnico de la Secretaría de Movilidad."]
    det.append("**Supuestos y límites**")
    det += [f"- {x[0].upper() + x[1:]}" for x in lim]
    det.append("**Fuentes**")
    det += [f"- {e['fuente'].replace('Bogotá · ', 'Bogotá: ')}" for e in res.get("evidencia", [])]
    det.append("[[/detalles]]")
    L.append("\n".join(det))
    return "\n\n".join(L)


def ayuda() -> str:
    nombres = sorted(NOMBRES_TIPO[t].split(" (")[0] for t in tipos_con_evidencia() if t in NOMBRES_TIPO)
    return ("**Así puedes usar VíaSegura IA**\n\n"
            "**1. Evaluar una medida:** dime qué harías y dónde.\n"
            "- «¿Funcionaría bajar la velocidad a 30 km/h en la Av. Boyacá?»\n"
            "- «Poner resaltos en la Calle 80» · «Zona escolar en los colegios de Bosa»\n\n"
            "**2. Pedir cifras:** «¿Cuántos muertos hubo en Suba en 2024?» · «¿Cuántos accidentes hubo en Bogotá?»\n\n"
            "**3. Usar el mapa:** en la pestaña *Mapa* dibuja la vía, el área o el punto exacto, elige la medida y pulsa "
            "*Evaluar*. Es más preciso, porque cuenta solo los siniestros dentro de lo que marcas. Desde un resultado "
            "puedes pulsar *Conversar sobre esta propuesta* para preguntarme «¿y a 40 km/h?» o «¿por qué confianza baja?».\n\n"
            "**4. Preguntar por normas:** «¿Qué programas tiene el Plan Distrital de Seguridad Vial?» · "
            "«¿Qué deben incluir los planes de gestión de la velocidad?». Respondo con las normas cargadas y cito "
            "la norma y el artículo.\n\n"
            f"**Medidas que puedo evaluar:** {', '.join(nombres)}.\n\n"
            "**Cómo leer la respuesta**\n"
            "- **Veredicto:** *Positiva* si reduce claramente siniestros o muertes; *Negativa* si los aumenta; "
            "*Neutra* si no hay un cambio claro.\n"
            "- **Rango probable:** entre qué valores está el efecto real con 95 % de probabilidad.\n"
            "- **Confianza:** *alta* si hay varios casos medidos en Bogotá que coinciden; *media* si hay algo de "
            "evidencia y el rango es razonable; *baja* si solo hay estudios de otros países o el rango es muy amplio.\n\n"
            "Solo respondo sobre medidas viales y siniestralidad en Bogotá.")


RE_AYUDA = re.compile(r"\bayuda\b|\bayudame\b|que (?:te )?(?:puedo|podria) (?:preguntar|hacer|consultar|pedir)|"
                      r"como (?:funciona|te uso|se usa|uso esto|usarte|te pregunto|preguntar)|"
                      r"que (?:sabes|haces|eres|puedes hacer|evaluas|medidas)|para que sirve|instrucciones|"
                      r"ejemplos? de (?:pregunta|uso)|\bmenu\b|\bopciones\b|\bguia\b|que te pregunto")


def es_ayuda(texto: str) -> bool:
    return bool(RE_AYUDA.search(plano(texto)))


PROMPT_REDACTOR = """Reescribe la siguiente respuesta para que suene más natural y cercana, en español.
REGLAS: conserva TODAS las cifras, porcentajes y rangos exactamente como están; no agregues datos, cifras ni
fuentes nuevas; conserva el veredicto, la tabla, las marcas [[detalles]] y [[/detalles]] y todo lo que hay entre ellas; no respondas nada fuera de este texto.

{texto}"""


def pulir_con_llm(texto: str, cli) -> str:
    """El LLM solo mejora la redacción. Si cambia o agrega algún número, se descarta."""
    try:
        r = cli.chat_completion(messages=[{"role": "user", "content": PROMPT_REDACTOR.format(texto=texto)}],
                                max_tokens=900, temperature=0.2)
        nuevo = r.choices[0].message.content.strip()
    except Exception:
        return texto
    nums = lambda s: sorted(re.findall(r"\d+(?:[.,]\d+)?", s))
    return nuevo if nums(nuevo) == nums(texto) and "[[detalles]]" in nuevo and "**Confianza" in nuevo else texto


# ════════════════════════════════════════════════════════════
#  ORQUESTADOR
# ════════════════════════════════════════════════════════════
RE_PREGUNTA_NUEVA = re.compile(r"^\W*(cuant[oa]s?|cual(?:es)?|que|quien(?:es)?|como|cuando|por que|existe|hay|"
                               r"dime|explica|muestra|ayuda)\b")


class Agente:
    def __init__(self, usar_llm: bool = True):
        self.ultima = None          # última medida evaluada (para llevarla al mapa)
        self.panel = cargar_panel()
        self.cli = cliente_llm() if usar_llm else None

    def responder(self, mensaje: str, historial: list | None = None) -> str:
        mensaje = (mensaje or "").strip()
        if not mensaje:
            return FUERA_DE_TEMA
        # si el turno anterior fue una pregunta de aclaración, se une con el mensaje previo del usuario
        previo = _aclaracion_pendiente(historial)
        if previo and RE_PREGUNTA_NUEVA.search(plano(mensaje)):
            previo = None               # «¿cuántos vehículos hay?» es otra pregunta, no la respuesta a «¿dónde?»
        texto = f"{previo} {mensaje}" if previo else mensaje
        m = extraer(texto, self.cli)

        self.ultima = None
        if es_ayuda(mensaje):
            return ayuda()
        if re.fullmatch(r"(hola|buenas|buenos dias|buenas tardes|buenas noches|hey|que tal)[ !.]*", plano(mensaje)):
            return ("Hola. Soy VíaSegura IA: estimo si una medida vial reduciría o aumentaría los siniestros en Bogotá, "
                    "con datos de la Secretaría de Movilidad. Cuéntame la medida y dónde se aplicaría, "
                    "o escribe **ayuda** para ver ejemplos.")
        if not m["en_tema"]:
            return FUERA_DE_TEMA
        if m["otra_ciudad"]:
            return FUERA_DE_BOGOTA
        if m["consulta_datos"] and not (m["tipo_medida"] and m.get("direccion")):
            return responder_datos(m, self.panel)
        if not m["tipo_medida"]:
            return ("Puedo ayudarte con eso si me dices **qué medida** quieres evaluar (por ejemplo: bajar el límite de "
                    "velocidad, poner resaltos, instalar cámaras) y **dónde** en Bogotá. También puedo darte cifras de "
                    "siniestros por año o por localidad.")
        falta = faltantes(m)
        if falta:
            return (f"Entiendo que hablas de **{NOMBRES_TIPO[m['tipo_medida']]}**. {MARCA_ACLARACION} que me digas "
                    + " y ".join(falta) + ".")
        if m.get("direccion") == "retirar":
            return (f"Solo tengo evidencia sobre **instalar o aplicar** medidas, no sobre retirarlas, así que no puedo "
                    f"estimar con datos qué pasaría al quitar {NOMBRES_TIPO[m['tipo_medida']]}. "
                    "Si quieres, te digo qué efecto tuvo esa medida cuando se instaló en Bogotá.")
        med, notas = a_medida(m, self.panel)
        res = evaluar({k: v for k, v in med.items() if not k.startswith("_")}, self.panel)
        texto_final = redactar(med, res, notas)
        self.ultima = {"m": m, "med": med, "res": res}
        return pulir_con_llm(texto_final, self.cli) if self.cli else texto_final


def _contenido(x) -> str:
    c = x.get("content") if isinstance(x, dict) else x
    return c if isinstance(c, str) else ""


def _aclaracion_pendiente(historial) -> str | None:
    """Devuelve el mensaje anterior del usuario si el bot acaba de pedir una aclaración."""
    if not historial:
        return None
    if isinstance(historial[-1], dict):            # formato [{"role":…, "content":…}]
        bots = [h for h in historial if h.get("role") == "assistant"]
        users = [h for h in historial if h.get("role") == "user"]
        if bots and users and MARCA_ACLARACION in _contenido(bots[-1]) and historial[-1].get("role") == "assistant":
            return _contenido(users[-1])
    else:                                          # formato [[usuario, bot], …]
        u, b = historial[-1][0], historial[-1][1]
        if b and MARCA_ACLARACION in _contenido(b):
            return _contenido(u)
    return None


# ════════════════════════════════════════════════════════════
#  INTERFACES
# ════════════════════════════════════════════════════════════
EJEMPLOS = ["¿Funcionaría bajar la velocidad a 30 km/h en las zonas escolares de Kennedy?",
            "¿Qué pasa si suben el límite a 60 km/h en la Autopista Norte?",
            "Quieren poner resaltos en la Avenida Ciudad de Cali",
            "¿Sirve instalar cámaras de fotodetección en la Calle 80?",
            "¿Cuántos muertos hubo en siniestros viales en Suba en 2024?"]

PRUEBAS = EJEMPLOS + ["Van a poner controles de policía", "en la avenida boyacá", "¿Cuál es la capital de Francia?",
                      "Dame una receta de ajiaco", "Bajar la velocidad a 40 km/h en Medellín",
                      "hola", "Poner semáforos en Bosa", "Subir la velocidad de 50 a 60 en la NQS con cámaras",
                      "¿Quién ganó el partido de Millonarios?", "Escríbeme un poema sobre el amor",
                      "¿Cuántos accidentes hubo en Bogotá?"]


def interfaz_web(agente: Agente):
    import gradio as gr
    desc = ("Escribe una medida vial para Bogotá y te digo si se espera que **reduzca o aumente** los siniestros, "
            "con datos de la Secretaría Distrital de Movilidad (2015 – hoy). "
            "Solo respondo sobre medidas viales y siniestralidad en Bogotá.")
    def fn(mensaje, historial):
        r = agente.responder(mensaje, historial)
        return (r.replace("[[detalles]]", "<details><summary>Detalles, supuestos y fuentes</summary>\n")
                 .replace("[[/detalles]]", "</details>"))
    try:
        demo = gr.ChatInterface(fn=fn, title="VíaSegura IA", description=desc, examples=EJEMPLOS, type="messages")
    except TypeError:                              # versiones de Gradio sin el parámetro `type`
        demo = gr.ChatInterface(fn=fn, title="VíaSegura IA", description=desc, examples=EJEMPLOS)
    demo.launch(inbrowser=True)


def consola(agente: Agente):
    print("VíaSegura IA (escribe 'salir' para terminar)\n")
    hist = []
    while True:
        try:
            msg = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if plano(msg) in ("salir", "exit", "quit"):
            break
        r = agente.responder(msg, hist)
        hist += [{"role": "user", "content": msg}, {"role": "assistant", "content": r}]
        print(f"\nVíaSegura IA:\n{r.replace('[[detalles]]', '— Detalles —').replace('[[/detalles]]', '')}\n")


def probar(agente: Agente):
    hist = []
    for p in PRUEBAS:
        r = agente.responder(p, hist)
        hist += [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        print(f"\n{'═' * 90}\nTÚ: {p}\n{'─' * 90}\n{r}")


if __name__ == "__main__":
    ag = Agente()
    print(f"Panel cargado · LLM: {'activo (' + MODELO_LLM + ')' if ag.cli else 'no configurado (modo reglas)'}")
    if "--probar" in sys.argv:
        probar(ag)
    elif "--consola" in sys.argv:
        consola(ag)
    else:
        interfaz_web(ag)
