"""
=============================================================
  VÍASEGURA IA — Aplicación web: mapa + chat
=============================================================

Pestaña MAPA
    1. Marque la zona: dibuje una vía (línea), un área (polígono o
       rectángulo) o un punto, o elija un corredor, un barrio, toda Bogotá
       o una o varias localidades (clic en el mapa coroplético). Puede
       combinar varias.
    2. Defina la resolución: escríbala en texto libre (se interpreta sola)
       o elija el tipo de medida y sus valores.
    3. Evaluar: veredicto, efecto estimado, en qué se basa, siniestros de
       la zona en el mapa y las zonas de análisis.

Pestaña CHAT
    Las preguntas sobre normas («¿qué dice el Plan Distrital de Seguridad
    Vial?») las responde el RAG de src/rag (embeddings + FAISS + Claude),
    citando norma y artículo. Lo demás, el mismo chatbot de chatbot.py, más una conversación sobre la propuesta
    del mapa («Conversar sobre esta propuesta»): explica el veredicto y la
    confianza, da cifras de la zona y re-evalúa cambios («¿y a 40 km/h?»,
    «con cámaras», «¿y en la Calle 80?»). Lo evaluado en el chat se puede
    llevar al mapa con «Ver en el mapa».

Uso
    pip install flask
    python src/app_web.py          → abre http://127.0.0.1:5000

Capas de límites (localidades y barrios = sectores catastrales):
    python src/ingesta_sdm.py --solo limites
    Sin esa capa, las localidades se dibujan con las celdas del panel y la
    búsqueda de barrios queda desactivada.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import threading
import webbrowser

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catalogo_resoluciones import LOCALIDADES, plano  # noqa: E402
from chatbot import (NOMBRES_TIPO, Agente, es_ayuda, explicar_confianza, explicar_veredicto,  # noqa: E402
                     _miles, _pct, extraer_reglas, recomendaciones, redactar)
from efecto_intervencion import (CORREDORES, M_LAT, M_LON, celdas_de_linea,  # noqa: E402
                                 celdas_de_localidad, celdas_de_puntos)
from evaluar_medida import evaluar, tipos_con_evidencia  # noqa: E402
from panel_espacial import poligono_celda  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINIESTROS = os.path.join(RAIZ, "out", "panel", "siniestros.csv")
DIR_LIMITES = os.path.join(RAIZ, "data", "raw", "limites")
TAM_CELDA = 500
MESES_BASE = 24
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024          # subida de normas: hasta 60 MB por envío


# ════════════════════════════════════════════════════════════
#  GEOMETRÍA
# ════════════════════════════════════════════════════════════
try:
    from matplotlib.path import Path as _MplPath
except Exception:            # matplotlib es opcional: hay versión en numpy
    _MplPath = None


def _anillos(geom: dict) -> list[list[list[float]]]:
    """Todos los anillos de un Polygon / MultiPolygon GeoJSON."""
    if geom.get("type") == "Polygon":
        return geom["coordinates"]
    if geom.get("type") == "MultiPolygon":
        return [r for p in geom["coordinates"] for r in p]
    return []


def dentro(lon: np.ndarray, lat: np.ndarray, anillos) -> np.ndarray:
    """Punto en polígono por paridad (los huecos quedan fuera automáticamente)."""
    res = np.zeros(len(lon), bool)
    if not len(lon):
        return res
    for ring in anillos:
        r = np.asarray(ring, float)
        if len(r) < 3:
            continue
        caja = (lon >= r[:, 0].min()) & (lon <= r[:, 0].max()) & (lat >= r[:, 1].min()) & (lat <= r[:, 1].max())
        if not caja.any():
            continue
        x, y = lon[caja], lat[caja]
        if _MplPath is not None:
            hit = _MplPath(r).contains_points(np.column_stack([x, y]))
        else:
            hit = np.zeros(len(x), bool)
            x1, y1 = r[:, 0], r[:, 1]
            x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
            for a, b, c, d in zip(x1, y1, x2, y2):
                cruza = ((b > y) != (d > y)) & (x < (c - a) * (y - b) / ((d - b) or 1e-12) + a)
                hit ^= cruza
        res[np.where(caja)[0]] ^= hit
    return res


def distancia_a_linea(lon, lat, coords) -> np.ndarray:
    x, y = np.asarray(lon) * M_LON, np.asarray(lat) * M_LAT
    pts = np.array([[a * M_LON, b * M_LAT] for a, b in coords])
    d = np.full(len(x), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
        d = np.minimum(d, np.hypot(x - (a[0] + t * ab[0]), y - (a[1] + t * ab[1])))
    return d


def distancia_a_lineas(lon, lat, lineas: list, hasta_m: float, bloque: int = 128) -> np.ndarray:
    """Distancia (m) de cada punto a la línea más cercana, calculada solo hasta `hasta_m`: los puntos
    más lejanos quedan en infinito. Vectorizado por bloques de segmentos y con un filtro por recuadro,
    así un corredor de miles de tramos (Av. Boyacá: ~2.000) tarda una fracción de segundo."""
    x, y = np.asarray(lon, float) * M_LON, np.asarray(lat, float) * M_LAT
    d = np.full(len(x), np.inf)
    a_ = [np.asarray(l, float)[:-1] for l in lineas if len(l) > 1]
    b_ = [np.asarray(l, float)[1:] for l in lineas if len(l) > 1]
    if not a_ or not len(x):
        return d
    A = np.concatenate(a_) * [M_LON, M_LAT]
    B = np.concatenate(b_) * [M_LON, M_LAT]
    for i in range(0, len(A), bloque):
        a, b = A[i:i + bloque], B[i:i + bloque]
        xmin, xmax = min(a[:, 0].min(), b[:, 0].min()) - hasta_m, max(a[:, 0].max(), b[:, 0].max()) + hasta_m
        ymin, ymax = min(a[:, 1].min(), b[:, 1].min()) - hasta_m, max(a[:, 1].max(), b[:, 1].max()) + hasta_m
        idx = np.flatnonzero((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax))
        if not len(idx):
            continue
        px, py = x[idx, None], y[idx, None]
        ab = b - a
        den = np.maximum((ab ** 2).sum(1), 1e-9)
        t = np.clip(((px - a[:, 0]) * ab[:, 0] + (py - a[:, 1]) * ab[:, 1]) / den, 0, 1)
        dist = np.hypot(px - (a[:, 0] + t * ab[:, 0]), py - (a[:, 1] + t * ab[:, 1])).min(1)
        d[idx] = np.minimum(d[idx], dist)
    return d


def _simplificar(anillo: list, tol: float = 0.00008) -> list:
    """Douglas-Peucker (tol ≈ 9 m) y 5 decimales: aligera lo que se manda al navegador."""
    if len(anillo) <= 4:
        return [[round(x, 5), round(y, 5)] for x, y in anillo]
    a = np.asarray(anillo, float)
    guardar = np.zeros(len(a), bool)
    m = int(np.argmax(np.hypot(*(a - a[0]).T)))       # anillo cerrado: se parte en el punto más lejano
    guardar[[0, m, -1]] = True
    pila = [(0, m), (m, len(a) - 1)]
    while pila:
        i, j = pila.pop()
        if j <= i + 1:
            continue
        p, q = a[i], a[j]
        d = q - p
        n = np.hypot(*d) or 1e-12
        dist = np.abs(d[0] * (a[i + 1:j, 1] - p[1]) - d[1] * (a[i + 1:j, 0] - p[0])) / n
        k = int(np.argmax(dist))
        if dist[k] > tol:
            guardar[i + 1 + k] = True
            pila += [(i, i + 1 + k), (i + 1 + k, j)]
    return [[round(x, 5), round(y, 5)] for x, y in a[guardar]]


def _redondear(g: dict) -> dict:
    if g.get("type") == "Polygon":
        return {"type": "Polygon", "coordinates": [_simplificar(r) for r in g["coordinates"]]}
    if g.get("type") == "MultiPolygon":
        return {"type": "MultiPolygon", "coordinates": [[_simplificar(r) for r in p] for p in g["coordinates"]]}
    return g


def lineas_de(geom: dict) -> list:
    if geom.get("type") == "LineString":
        return [geom["coordinates"]]
    if geom.get("type") == "MultiLineString":
        return geom["coordinates"]
    return []


def arcgis_a_geojson(g: dict) -> dict | None:
    """rings / paths de ArcGIS → MultiPolygon / MultiLineString GeoJSON."""
    if "rings" in g:
        poligonos = []
        for ring in g["rings"]:
            r = np.asarray(ring, float)
            area = 0.5 * np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1])
            if area <= 0 or not poligonos:      # en ArcGIS el anillo exterior va en sentido horario
                poligonos.append([ring])
            else:
                poligonos[-1].append(ring)       # hueco del polígono anterior
        return {"type": "MultiPolygon", "coordinates": poligonos}
    if "paths" in g:
        return {"type": "MultiLineString", "coordinates": g["paths"]}
    if "x" in g:
        return {"type": "Point", "coordinates": [g["x"], g["y"]]}
    return None


# ════════════════════════════════════════════════════════════
#  DATOS (se cargan una vez al iniciar)
# ════════════════════════════════════════════════════════════
class _CapaPerezosa(dict):
    """{nombre: geometría ArcGIS en texto JSON}; la convierte a GeoJSON solo al pedirla."""

    def __getitem__(self, nombre):
        return arcgis_a_geojson(json.loads(dict.__getitem__(self, nombre)))


class Datos:
    def __init__(self):
        print("Cargando datos…")
        self.agente = Agente()
        self.panel = self.agente.panel
        self.celdas = self.panel.drop_duplicates("celda")[["celda", "lat", "lon", "localidad"]].reset_index(drop=True)
        self.siniestros, self.fin_datos = self._siniestros()
        self.corredores = self._corredores()
        self.localidades = self._capa("Localidad", es_localidad=True)
        self.barrios = self._capa("Sector_catastral")
        self.coropletico = self._coropletico()
        print(f"  panel: {self.panel['celda'].nunique()} celdas · siniestros recientes: {len(self.siniestros):,} · "
              f"corredores: {len(self.corredores)} · localidades: {len(self.localidades)} · barrios: {len(self.barrios)}")

    def _siniestros(self):
        if not os.path.exists(SINIESTROS):
            return pd.DataFrame(columns=["lon", "lat", "victimas", "fatal", "n_heridos", "n_muertos"]), None
        usadas = {"fecha", "GRAVEDAD", "lat", "lon", "Latitud", "Longitud", "n_heridos", "n_muertos", "via",
                  "con_victimas", "fatal"}
        s = pd.read_csv(SINIESTROS, low_memory=False, usecols=lambda c: c.lstrip("﻿") in usadas)
        s.columns = [c.lstrip("﻿") for c in s.columns]
        if "lat" not in s:                                  # versión vieja (panel_espacial.py)
            s = s.rename(columns={"Latitud": "lat", "Longitud": "lon"})
        s["fecha"] = pd.to_datetime(s["fecha"], errors="coerce")
        if "GRAVEDAD" in s:
            g = s["GRAVEDAD"].astype(str).str.upper()
            s["victimas"] = g.isin(["CON HERIDOS", "CON MUERTOS"]).astype(int)
            s["fatal"] = (g == "CON MUERTOS").astype(int)
        else:
            s["victimas"], s["fatal"] = s.get("con_victimas", 0), s.get("fatal", 0)
        fin = s["fecha"].max()
        ultimo_mes = fin.to_period("M").to_timestamp()
        if fin >= ultimo_mes + pd.offsets.MonthEnd(0) - pd.Timedelta(days=1):   # el último mes está completo
            ultimo_mes = ultimo_mes + pd.DateOffset(months=1)
        desde = ultimo_mes - pd.DateOffset(months=MESES_BASE)
        s = s[(s["fecha"] >= desde) & (s["fecha"] < ultimo_mes) & s["lat"].between(4.4, 4.9)]
        cols = ["lon", "lat", "victimas", "fatal", "n_heridos", "n_muertos", "fecha"] + (["via"] if "via" in s else [])
        return s[cols].reset_index(drop=True), ultimo_mes - pd.Timedelta(days=1)

    def _coropletico(self) -> dict:
        """Localidades con sus siniestros al año (últimos 24 meses) para el mapa coroplético."""
        s, f = self.siniestros, 12 / MESES_BASE
        feats = []
        for nombre, geom in self.localidades.items():
            m = dentro(s["lon"].to_numpy(), s["lat"].to_numpy(), _anillos(geom))
            feats.append({"type": "Feature", "geometry": _redondear(geom),
                          "properties": {"nombre": nombre, "victimas_anual": round(float(s.loc[m, "victimas"].sum() * f)),
                                         "muertos_anual": round(float(s.loc[m, "n_muertos"].sum() * f), 1)}})
        return {"type": "FeatureCollection", "features": feats}

    def _corredores(self) -> dict:
        if not os.path.exists(CORREDORES):
            return {}
        geo = json.load(open(CORREDORES, encoding="utf-8"))
        return {f["properties"]["corredor"]: f["geometry"] for f in geo["features"]}

    def _capa(self, patron: str, es_localidad=False) -> dict:
        """Lee una capa de polígonos (data/raw/limites/*<patron>*.csv) → {nombre: geometría}."""
        archivos = glob.glob(os.path.join(DIR_LIMITES, f"*{patron}*.csv"))
        if not archivos:
            return {}
        d = pd.read_csv(archivos[0], low_memory=False)
        textos = [c for c in d.columns if pd.api.types.is_string_dtype(d[c]) and c != "geometria_json"]
        if es_localidad:
            objetivo = {plano(x) for x in LOCALIDADES}
            col = max(textos, key=lambda c: d[c].astype(str).map(plano).isin(objetivo).sum(), default=None)
        else:
            candidatos = [c for c in textos if "NOM" in c.upper()] or textos
            col = max(candidatos, key=lambda c: d[c].nunique(), default=None)
        if col is None or "geometria_json" not in d:
            return {}
        capa = {} if es_localidad else _CapaPerezosa()
        for _, f in d.iterrows():
            try:
                geom = arcgis_a_geojson(json.loads(f["geometria_json"]))
            except (TypeError, ValueError):
                continue
            nombre = str(f[col]).strip().title()
            if not geom or nombre in ("", "Nan"):
                continue
            if es_localidad:
                nombre = next((l for l in LOCALIDADES if plano(l) == plano(nombre) or plano(nombre).endswith(" " + plano(l))), nombre)
            if nombre in capa:                                  # barrios con el mismo nombre en dos localidades
                nombre = f"{nombre} ({len([k for k in capa if k.startswith(nombre)]) + 1})"
            # Los ~1 200 barrios se guardan como texto y se convierten al pedirlos: como listas de
            # Python ocupan ~75 MB, demasiado para un servidor de 512 MB.
            capa[nombre] = geom if es_localidad else f["geometria_json"]
        return capa


D: Datos | None = None


# ════════════════════════════════════════════════════════════
#  ZONA DIBUJADA → CELDAS Y LÍNEA BASE PRECISA
# ════════════════════════════════════════════════════════════
def zona_de_features(features: list[dict]) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    """Devuelve (celdas del panel, máscara de siniestros dentro de la zona, nombres, notas)."""
    celdas, notas, nombres = set(), [], []
    s = D.siniestros
    mascara = np.zeros(len(s), bool)
    clon, clat = D.celdas["lon"].to_numpy(), D.celdas["lat"].to_numpy()
    for f in features:
        g, p = f.get("geometry") or {}, f.get("properties") or {}
        tipo, nombre = p.get("tipo", ""), p.get("nombre")
        if tipo == "ciudad":                         # toda Bogotá: todas las celdas y todos los siniestros
            celdas |= set(D.celdas["celda"])
            mascara |= True
            nombres.append("toda Bogotá")
            continue
        if tipo == "localidad" and nombre:           # la geometría oficial, no la que manda el navegador
            clave = next((k for k in D.localidades if plano(k) == plano(nombre)), None)
            g = D.localidades[clave] if clave else {}
        if nombre:
            nombres.append(nombre)
        if g.get("type") == "Point":
            radio = float(p.get("radio_m") or 150)
            celdas |= set(celdas_de_puntos(D.panel, [g["coordinates"]], radio))
            mascara |= np.hypot((s["lon"].to_numpy() - g["coordinates"][0]) * M_LON,
                                (s["lat"].to_numpy() - g["coordinates"][1]) * M_LAT) <= radio
            nombres.append(nombre or f"punto (radio {radio:g} m)")
        elif lineas_de(g):
            ancho = float(p.get("ancho_m") or 30)
            lineas = lineas_de(g)
            alcance = ancho + TAM_CELDA / 2              # celda cuyo centro está a menos de ancho + media celda
            dc = distancia_a_lineas(D.celdas["lon"], D.celdas["lat"], lineas, alcance)
            celdas |= set(D.celdas.loc[dc <= alcance, "celda"])
            mascara |= distancia_a_lineas(s["lon"], s["lat"], lineas, ancho) <= ancho
            if not nombre:
                largo = sum(np.hypot(np.diff(np.array(l)[:, 0]) * M_LON, np.diff(np.array(l)[:, 1]) * M_LAT).sum()
                            for l in lineas_de(g)) / 1000
                nombres.append(f"vía dibujada ({largo:.1f} km)")
        elif _anillos(g):
            an = _anillos(g)
            sel = dentro(clon, clat, an)
            if not sel.any():                       # polígono más chico que una celda: la celda que lo contiene
                v = np.array(an[0])
                sel = np.zeros(len(clon), bool)
                for x, y in v:
                    sel |= (np.abs(clon - x) * M_LON <= TAM_CELDA / 2) & (np.abs(clat - y) * M_LAT <= TAM_CELDA / 2)
            celdas |= set(D.celdas.loc[sel, "celda"])
            mascara |= dentro(s["lon"].to_numpy(), s["lat"].to_numpy(), an)
            if not nombre:
                nombres.append("área dibujada")
        elif tipo == "localidad" and nombre:        # localidad sin capa de límites: celdas del panel
            cs = celdas_de_localidad(D.panel, nombre)
            celdas |= set(cs)
            mascara |= _mascara_celdas(cs)
    return sorted(celdas), mascara, list(dict.fromkeys(nombres)), notas


def _mascara_celdas(cs: list[str]) -> np.ndarray:
    s = D.siniestros
    filas = np.floor((s["lat"].to_numpy() - 4.45) * M_LAT / TAM_CELDA).astype(int)
    cols = np.floor((s["lon"].to_numpy() + 74.25) * M_LON / TAM_CELDA).astype(int)
    ids = np.char.add(np.char.add("r", filas.astype(str)), np.char.add("_c", cols.astype(str)))
    return np.isin(ids, list(cs))


def linea_base_precisa(mascara: np.ndarray) -> dict:
    s = D.siniestros[mascara]
    f = 12 / MESES_BASE
    return {"periodo": f"últimos {MESES_BASE} meses (hasta {D.fin_datos:%m/%Y})" if D.fin_datos is not None else "",
            "sin_victimas_anual": round(float(s["victimas"].sum() * f), 1),
            "muertos_anual": round(float(s["n_muertos"].sum() * f), 1),
            "heridos_anual": round(float(s["n_heridos"].sum() * f), 1),
            "siniestros_anual": round(len(s) * f, 1)}


def celdas_geojson(celdas: list[str]) -> dict:
    feats = []
    for c in celdas:
        m = re.match(r"r(-?\d+)_c(-?\d+)", c)
        if m:
            feats.append({"type": "Feature", "properties": {"celda": c},
                          "geometry": {"type": "Polygon", "coordinates": poligono_celda(int(m.group(1)), int(m.group(2)), TAM_CELDA)}})
    return {"type": "FeatureCollection", "features": feats}


# ════════════════════════════════════════════════════════════
#  RUTAS
# ════════════════════════════════════════════════════════════
@app.get("/")
def inicio():
    return render_template("index.html")


@app.get("/api/capas")
def capas():
    evaluables = tipos_con_evidencia()
    # solo se ofrecen las medidas que tienen evidencia (medida en Bogotá o internacional citada)
    tipos = [{"id": k, "nombre": v, "con_evidencia": True} for k, v in NOMBRES_TIPO.items() if k in evaluables]
    tipos.sort(key=lambda t: t["nombre"])
    return jsonify(localidades=sorted(LOCALIDADES), corredores=sorted(D.corredores), tipos=tipos,
                   hay_limites=bool(D.localidades), hay_barrios=bool(D.barrios),
                   fin_datos=D.fin_datos.strftime("%d/%m/%Y") if D.fin_datos is not None else None)


CIUDAD = {"type": "Feature", "geometry": None, "properties": {"tipo": "ciudad", "nombre": "toda Bogotá"}}


def feature_de(tipo: str, nombre: str) -> dict | None:
    """Zona por nombre (corredor, localidad o barrio) como Feature GeoJSON."""
    if tipo == "ciudad":
        return json.loads(json.dumps(CIUDAD))
    if tipo == "corredor" and nombre in D.corredores:
        return {"type": "Feature", "geometry": D.corredores[nombre],
                "properties": {"tipo": "corredor", "nombre": nombre, "ancho_m": 40}}
    if tipo == "localidad":
        clave = next((k for k in D.localidades if plano(k) == plano(nombre)), None)
        if clave:
            return {"type": "Feature", "geometry": D.localidades[clave], "properties": {"tipo": "localidad", "nombre": clave}}
        cs = celdas_de_localidad(D.panel, nombre)       # sin capa: el contorno son las celdas del panel
        if cs:
            return {"type": "Feature", "geometry": None,
                    "properties": {"tipo": "localidad", "nombre": nombre, "celdas": celdas_geojson(cs)}}
    if tipo == "barrio" and nombre in D.barrios:
        return {"type": "Feature", "geometry": D.barrios[nombre], "properties": {"tipo": "barrio", "nombre": nombre}}
    return None


@app.post("/api/celdas")
def celdas_zona():
    """Cuadros de 500 m que caen en la zona marcada (o todos, si no hay zona): las unidades de análisis."""
    features = ((request.json or {}).get("zona") or {}).get("features") or []
    celdas = zona_de_features(features)[0] if features else list(D.celdas["celda"])
    return jsonify({"celdas": celdas_geojson(celdas), "n": len(celdas), "toda_la_ciudad": not features})


@app.get("/api/localidades")
def localidades():
    return jsonify(D.coropletico)


@app.get("/api/zona")
def zona():
    f = feature_de(request.args.get("tipo"), request.args.get("nombre", ""))
    return jsonify(f) if f else (jsonify(error="Zona no encontrada"), 404)


@app.get("/api/barrios")
def barrios():
    q = plano(request.args.get("q", ""))
    if len(q) < 2:
        return jsonify([])
    nombres = [n for n in D.barrios if q in plano(n)]
    nombres.sort(key=lambda n: (not plano(n).startswith(q), n))
    return jsonify(nombres[:20])


@app.post("/api/interpretar")
def interpretar():
    texto = (request.json or {}).get("texto", "")
    m = extraer_reglas(texto)
    if not m["en_tema"]:
        return jsonify(error="No reconozco una medida vial en ese texto. Describa la resolución "
                             "(por ejemplo: «reducir el límite de velocidad a 30 km/h»).")
    return jsonify({k: m.get(k) for k in ("tipo_medida", "direccion", "valor_actual", "valor_propuesto",
                                          "acompanamiento", "corredor", "localidad", "entorno_escolar")})


class ErrorPropuesta(Exception):
    pass


def evaluar_propuesta(features: list[dict], r: dict) -> dict:
    """Zona (features GeoJSON) + resolución → resultado completo. Lanza ErrorPropuesta si falta algo."""
    if not features:
        raise ErrorPropuesta("Marque primero una zona en el mapa (vía, área, punto, localidad, corredor o barrio).")
    if not r.get("tipo_medida"):
        raise ErrorPropuesta("Indique el tipo de medida de la resolución.")
    celdas, mascara, nombres, notas = zona_de_features(features)
    if not celdas:
        raise ErrorPropuesta("La zona marcada queda por fuera del área con datos de siniestros de Bogotá.")

    med = {"tipo_medida": r["tipo_medida"], "direccion": r.get("direccion") or "implementar",
           "acompanamiento": list(r.get("acompanamiento") or []), "velocidad_media_actual": None,
           "valor_actual": _num(r.get("valor_actual")), "valor_propuesto": _num(r.get("valor_propuesto")),
           "ubicacion": {"nivel": "celdas", "ids": celdas}, "entorno_escolar": bool(r.get("entorno_escolar"))}
    if med["tipo_medida"] == "zona_escolar":
        med["tipo_medida"], med["entorno_escolar"] = "limite_velocidad", True
        if not med["valor_propuesto"]:
            med["valor_propuesto"] = 30
            notas.append("Tomé la zona escolar como un límite de 30 km/h, como lo define el Decreto 073 de 2021.")
    if med["tipo_medida"] == "limite_velocidad":
        if not med["valor_propuesto"]:
            raise ErrorPropuesta("Indique el nuevo límite de velocidad (km/h).")
        if not med["valor_actual"]:
            med["valor_actual"] = 50
            notas.append("Supuse que el límite actual es 50 km/h (Decreto 073 de 2021), porque no lo indicaste.")
        med["direccion"] = "reducir" if med["valor_propuesto"] < med["valor_actual"] else "aumentar"
        if any(lineas_de(f.get("geometry") or {}) and (f.get("properties") or {}).get("tipo") != "corredor" for f in features):
            notas.append("La evidencia de velocidad medida en Bogotá viene de avenidas principales; en una calle local el efecto puede ser distinto.")

    res = evaluar(med, D.panel)
    res["advertencias"] = [a for a in res.get("advertencias", []) if not a.startswith("La zona tiene pocos")]
    base = linea_base_precisa(mascara)
    if base["siniestros_anual"] > 0 or not res.get("linea_base"):
        res["linea_base"] = base                       # siniestros DENTRO de lo dibujado, no de las celdas
        if res.get("cambio_sin_victimas_pct") is not None:
            res["siniestros_victimas_evitados_anual"] = round(-res["cambio_sin_victimas_pct"] / 100 * base["sin_victimas_anual"], 1)
        if res.get("cambio_muertes_pct") is not None:
            res["muertes_evitadas_anual"] = round(-res["cambio_muertes_pct"] / 100 * base["muertos_anual"], 1)
    else:
        notas.append("No encontré siniestros georreferenciados dentro de la zona dibujada; la línea base usa las celdas de 500 m.")
    med["_zona"] = ", ".join(nombres)[:200] or "zona marcada"
    med["_tipos_zona"] = {(f.get("properties") or {}).get("tipo") or "area" for f in features}
    med["_tipos_zona"] = {"area" if t in ("rect", "barrio") else t for t in med["_tipos_zona"]}
    if "ciudad" in med["_tipos_zona"]:
        notas.append("Evalué la medida para toda la ciudad. Los efectos se midieron en avenidas y tramos concretos; "
                     "aplicarlos a toda la ciudad es una extrapolación.")
    if base["sin_victimas_anual"] < 20:
        notas.append("La zona tiene pocos siniestros con víctimas al año: el número de siniestros evitados es muy incierto.")
    texto = redactar(med, res, notas)

    s = D.siniestros[mascara]
    puntos = s[s["victimas"] == 1].sort_values("fatal", ascending=False).head(4000)
    ver = res.get("veredicto") if res.get("cambio_sin_victimas_pct") is not None else "Sin evidencia"
    return {"markdown": texto, "veredicto": ver, "confianza": res.get("confianza"),
            "linea_base": base, "cambio_pct": res.get("cambio_sin_victimas_pct"), "rango_pct": res.get("rango_pct"),
            "cambio_muertes_pct": res.get("cambio_muertes_pct"), "rango_muertes_pct": res.get("rango_muertes_pct"),
            "celdas": celdas_geojson(celdas), "siniestros": puntos[["lat", "lon", "fatal"]].round(6).values.tolist(),
            "n_siniestros_zona": int(len(s)), "titulo": titulo_propuesta(r, med["_zona"]),
            "_res": res, "_med": med, "_notas": notas}


def _publico(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


@app.post("/api/evaluar")
def evaluar_zona():
    datos = request.json or {}
    try:
        d = evaluar_propuesta((datos.get("zona") or {}).get("features") or [], datos.get("resolucion") or {})
    except ErrorPropuesta as e:
        return jsonify(error=str(e)), 400
    return jsonify(_publico(d))


# ════════════════════════════════════════════════════════════
#  CHAT (con o sin una propuesta en contexto)
# ════════════════════════════════════════════════════════════
ACOMP = {"fotodeteccion": "cámaras", "control_policial": "control policial"}


def titulo_propuesta(r: dict, zona: str) -> str:
    t = r.get("tipo_medida")
    nombre = NOMBRES_TIPO.get(t, t or "medida").split(" (")[0]
    if t in ("limite_velocidad", "zona_escolar") and r.get("valor_propuesto"):
        nombre = ("zona escolar a " if t == "zona_escolar" else "límite a ") + f"{float(r['valor_propuesto']):g} km/h"
    if r.get("acompanamiento"):
        nombre += " con " + " y ".join(ACOMP.get(a, a) for a in r["acompanamiento"])
    return f"{nombre[0].upper() + nombre[1:]} · {zona}"


def propuesta_de_chat(ultima: dict) -> dict | None:
    """Medida evaluada en el chat → zona y resolución para llevarla al mapa."""
    m, med = ultima["m"], ultima["med"]
    ub = med.get("ubicacion") or {}
    if ub.get("nivel") == "celdas" and m.get("localidad") and med.get("entorno_escolar"):
        polis = []
        for c in ub["ids"]:
            g = re.match(r"r(-?\d+)_c(-?\d+)", c)
            if g:
                polis.append(poligono_celda(int(g.group(1)), int(g.group(2)), TAM_CELDA))
        f = {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": polis},
             "properties": {"tipo": "area", "nombre": f"entornos escolares de {m['localidad']}"}}
    elif ub.get("nivel") == "corredor":
        f = feature_de("corredor", ub["nombre"])
    elif ub.get("nivel") == "localidad":
        f = feature_de("localidad", ub["nombre"])
    else:
        f = feature_de("ciudad", "")                   # toda la ciudad
    if not f:
        return None
    escolar = med.get("entorno_escolar") and med["tipo_medida"] == "limite_velocidad" and (med.get("valor_propuesto") or 99) <= 30
    r = {"tipo_medida": "zona_escolar" if escolar else med["tipo_medida"], "direccion": med.get("direccion"),
         "valor_actual": med.get("valor_actual"), "valor_propuesto": med.get("valor_propuesto"),
         "acompanamiento": med.get("acompanamiento") or [], "entorno_escolar": bool(med.get("entorno_escolar"))}
    return {"zona": {"type": "FeatureCollection", "features": [f]}, "resolucion": r,
            "titulo": titulo_propuesta(r, f["properties"]["nombre"])}


RE_POR_QUE = re.compile(r"por ?que|explica|razon|significa|confianza|veredicto|como (?:lo )?(?:calcul|sab|estim)|segur[oa]")
RE_FUENTES = re.compile(r"fuente|en que te bas|de donde|estudio|evidencia|supuesto")
RE_RECOM = re.compile(r"recomiend|recomendaci|que (?:hago|harias|sugier)|alternativ|mejor medida|que mas")
RE_CIFRAS = re.compile(r"cuant|cifra|linea base|datos de la zona|siniestros hay|cuanto")
RE_SIN = re.compile(r"\bsin (?:las |los |el |la )?(camaras?|fotodeteccion|fotomultas?|control|polici\w*|agentes)|"
                    r"\bquit\w* (?:las |los |el |la )?(camaras?|fotodeteccion|control|polici\w*)")


def _bloque_detalles(md: str) -> str:
    g = re.search(r"\[\[detalles\]\]\n?(.*?)\n?\[\[/detalles\]\]", md, re.S)
    return g.group(1).strip() if g else ""


def explicar_en_contexto(d: dict, t: str) -> str:
    res, ver = d["_res"], d["veredicto"]
    if ver == "Sin evidencia":
        return ("No hay evidencia, ni de Bogotá ni internacional, para estimar esta medida, así que no doy veredicto ni "
                "confianza. Prueba con otra medida (escribe **ayuda** para ver cuáles puedo evaluar).")
    L = []
    solo_conf = "confianza" in t and not re.search(r"veredicto|positiv|negativ|neutr", t)
    if not solo_conf:
        L.append(f"**¿Por qué {ver.lower()}?** {explicar_veredicto(res)}")
    exp = explicar_confianza(res)
    L.append(f"**¿Por qué confianza {res['confianza']}?** {exp[0].upper() + exp[1:]}")
    evs = res.get("evidencia", [])
    n_local = sum(e["tipo"] == "local" for e in evs)
    if res["confianza"] != "alta":
        sub = []
        if n_local == 0:
            sub.append("medir el resultado en Bogotá después de implementarla (antes y después, contra zonas parecidas sin la medida)")
        if n_local <= 1:
            sub.append("tener más casos implementados en la ciudad que confirmen el efecto")
        if any("velocidad media" in str(x) for x in res.get("supuestos", [])):
            sub.append("contar con la velocidad real medida en la vía")
        if sub:
            L.append("**Qué subiría la confianza:** " + "; ".join(sub) + ".")
    L.append("Pregunta «¿en qué te basas?» para ver los estudios y supuestos.")
    return "\n\n".join(L)


def cifras_en_contexto(d: dict) -> str:
    b, res = d["linea_base"], d["_res"]
    dec = lambda x: f"{x:.1f}".replace(".", ",")
    per = (b.get("periodo") or "últimos 24 meses").replace("últimos", "promedio de los últimos").replace(" (hasta", ", hasta").rstrip(")")
    L = [f"**En {d['_med']['_zona']}**, por año ({per}):",
         f"- {_miles(b['siniestros_anual'])} siniestros en total, {_miles(b['sin_victimas_anual'])} con víctimas",
         f"- {dec(b['muertos_anual'])} muertos y {_miles(b['heridos_anual'])} heridos"]
    texto = "\n".join(L)
    ev, mu = res.get("siniestros_victimas_evitados_anual"), res.get("muertes_evitadas_anual")
    if ev is not None:
        texto += (f"\n\nCon la medida estimo unos **{_miles(abs(ev))} siniestros con víctimas {'menos' if ev >= 0 else 'más'}** al año"
                  + (f" y {dec(abs(mu))} muertes {'menos' if mu >= 0 else 'más'}" if mu is not None else "") + ".")
    return texto


def modificar_resolucion(msg: str, r: dict, pregunta: bool = False) -> tuple[dict, list[str]]:
    """Cambios a la resolución pedidos en el chat («¿y a 40?», «con cámaras», «mejor resaltos»)."""
    t, m = plano(msg), extraer_reglas(msg)
    r, cambios = json.loads(json.dumps(r)), []
    tipo = m["tipo_medida"]
    # «con cámaras», «y con policía»: acompañamiento de la medida actual, no una medida nueva
    con_cam = re.search(r"\b(?:con|y|mas|agreg\w*|sum\w*)\s+(?:las |los |unas )?(?:camaras?|fotodeteccion|fotomultas?)", t)
    con_pol = re.search(r"\b(?:con|y|mas|agreg\w*|sum\w*)\s+(?:el |la |los )?(?:control(?: policial)?|polici\w*|agentes)", t)
    if tipo and not pregunta and tipo != r.get("tipo_medida") and not (tipo == "limite_velocidad" and r.get("tipo_medida") == "zona_escolar"):
        if not ((tipo == "fotodeteccion" and con_cam) or (tipo == "control_operativo" and con_pol)) and not RE_SIN.search(t):
            r = {"tipo_medida": tipo, "direccion": m.get("direccion") or "implementar", "acompanamiento": [],
                 "valor_actual": None, "valor_propuesto": None, "entorno_escolar": tipo == "zona_escolar"}
            cambios.append(f"medida: {NOMBRES_TIPO.get(tipo, tipo).split(' (')[0]}")
    if r.get("tipo_medida") in ("limite_velocidad", "zona_escolar"):
        if m["valor_propuesto"]:
            nuevo, actual = m["valor_propuesto"], m["valor_actual"]
        elif not (m["corredor"] or m["localidad"]):
            sin_calles = re.sub(r"\b(?:calle|carrera|cra|kr|cl|cll|avenida|av|diagonal|transversal|dg|tv|ak|ac)\.? ?\d+\w*", " ", t)
            nums = [int(x) for x in re.findall(r"\b(\d{2,3})\b", sin_calles) if 10 <= int(x) <= 120]
        else:
            nums = []
        if not m["valor_propuesto"]:
            nuevo, actual = (nums[-1] if nums else None), (nums[0] if len(nums) > 1 else None)
        if nuevo and nuevo != _num(r.get("valor_propuesto")):
            r["valor_propuesto"] = nuevo
            cambios.append(f"nuevo límite {nuevo} km/h")
        if actual and actual != _num(r.get("valor_actual")):
            r["valor_actual"] = actual
            cambios.append(f"límite actual {actual} km/h")
    acomp = list(r.get("acompanamiento") or [])
    quitar = RE_SIN.findall(t)
    for grupo in quitar:
        palabra = "".join(grupo)
        a = "fotodeteccion" if re.match(r"camar|foto", palabra) else "control_policial"
        if a in acomp:
            acomp.remove(a)
            cambios.append(f"sin {ACOMP[a]}")
    pedidos = list(m["acompanamiento"]) + (["fotodeteccion"] if con_cam else []) + (["control_policial"] if con_pol else [])
    if not quitar:
        for a in dict.fromkeys(pedidos):
            if a not in acomp:
                acomp.append(a)
                cambios.append(f"con {ACOMP[a]}")
    r["acompanamiento"] = acomp
    return r, cambios


def responder_con_contexto(msg: str, ctx: dict) -> dict | None:
    """Responde una pregunta sobre la propuesta en contexto. None = que la atienda el chat general."""
    t, m = plano(msg), extraer_reglas(msg)
    features = ((ctx.get("zona") or {}).get("features")) or []
    r = ctx.get("resolucion") or {}
    if es_ayuda(msg) or m["otra_ciudad"] or not features:
        return None
    if m["consulta_datos"] and (m["anio"] or m["toda_la_ciudad"] or m["localidad"] or m["corredor"]):
        return None                                    # pregunta de cifras de otro lugar o año: chat general
    if es_pregunta_de_dato(msg, m) and not re.search(r"\bzona\b|\baqui\b|\balli\b|\besta via\b", t):
        return None                                    # «¿cuántos vehículos hay en Bogotá?» no es sobre la propuesta
    pide_cambio = re.search(r"^\W*(?:y|mejor|que tal|y si|si fuera|en vez|en lugar|ahora|cambia\w*|prueba\w*|evalua\w*|"
                            r"pon\w*|aplica\w*|lleva\w*)\b", t)

    nuevas = features
    if not pide_cambio and m["tipo_medida"] and (m["corredor"] or m["localidad"] or m["toda_la_ciudad"]):
        return None                                    # «resaltos en la Calle 80»: una propuesta nueva, chat general
    lugar = None
    if not pide_cambio:
        pass                                           # mencionar un lugar no basta para mover la propuesta
    elif m["corredor"] and m["corredor"] in D.corredores:
        lugar = feature_de("corredor", m["corredor"])
    elif m["localidad"]:
        lugar = feature_de("localidad", m["localidad"])
    elif m["toda_la_ciudad"]:
        lugar = feature_de("ciudad", "")
    pregunta = bool(RE_POR_QUE.search(t) or RE_FUENTES.search(t) or RE_RECOM.search(t) or RE_CIFRAS.search(t))
    r2, cambios = modificar_resolucion(msg, r, pregunta)
    if lugar:
        nuevas = [lugar]
        cambios.insert(0, f"zona: {lugar['properties']['nombre']}")

    if cambios:
        try:
            d = evaluar_propuesta(nuevas, r2)
        except ErrorPropuesta as e:
            return {"respuesta": str(e)}
        nuevo_ctx = {"zona": {"type": "FeatureCollection", "features": nuevas}, "resolucion": r2, "titulo": d["titulo"]}
        intro = "Cambié la propuesta (" + ", ".join(cambios) + ")."
        try:
            antes = evaluar_propuesta(features, r)
            if antes["cambio_pct"] is not None and d["cambio_pct"] is not None:
                ico = {"Positiva": "🟢", "Negativa": "🔴", "Neutra": "🟡"}
                intro += (f" Antes: {ico.get(antes['veredicto'], '')} {antes['veredicto']} ({_pct(antes['cambio_pct'])} en "
                          f"siniestros con víctimas) → ahora: {ico.get(d['veredicto'], '')} {d['veredicto']} ({_pct(d['cambio_pct'])}).")
        except ErrorPropuesta:
            pass
        return {"respuesta": intro + "\n\n" + d["markdown"],
                "contexto": nuevo_ctx, "propuesta": nuevo_ctx}

    if not pregunta:
        return None
    try:
        d = evaluar_propuesta(features, r)
    except ErrorPropuesta as e:
        return {"respuesta": str(e)}
    if RE_FUENTES.search(t):
        texto = _bloque_detalles(d["markdown"]) or "No hay estudios ni supuestos adicionales para esta propuesta."
    elif RE_RECOM.search(t):
        rec = recomendaciones(d["_med"], d["_res"])
        texto = ("**Recomendaciones para esta propuesta:**\n" + "\n".join(f"- {x}" for x in rec)) if rec else \
            "No tengo una recomendación adicional: la medida ya es de las que mejor evidencia tienen en Bogotá."
    elif RE_CIFRAS.search(t) and not RE_POR_QUE.search(t):
        texto = cifras_en_contexto(d)
    else:
        texto = explicar_en_contexto(d, t)
    return {"respuesta": texto}


# ════════════════════════════════════════════════════════════
#  NORMAS: el RAG (src/rag) — embeddings + FAISS + Claude
# ════════════════════════════════════════════════════════════
RE_NORMA = re.compile(r"decreto|resoluci|\bnormas?\b|normativ|\bley\b|\bleyes\b|articulo|acuerdo distrital|reglament|"
                      r"plan distrital|planes? de gestion|codigo de transito|multa|sancion|que dice|segun (?:la|el)|"
                      r"vigen|derog|comite|obligaci|competencia|quien (?:debe|fija|define|establece|aprueba)|requisito")
RE_PREGUNTA = re.compile(r"^\W*(cual|cuales|que|quien|quienes|como|cuando|donde|existe|hay|se puede|esta permitido)\b")
RE_EVALUAR = re.compile(r"efecto|impacto|sirve|servir|funcionar|funciona |reduc|aument|disminu|evalu|conviene|vale la pena")


RE_CUANTOS = re.compile(r"^\W*(?:y\s+)?(?:cuant[oa]s?|cual es (?:el|la) (?:numero|cantidad|total|poblacion)|numero de|cantidad de)\b")
RE_SINIESTRO = re.compile(r"siniestr|accident|choque|atropell|muert|fallec|herid|lesion|victim")
RE_VIAL = re.compile(r"vehicul|parque automotor|carros?\b|autos?\b|motos?\b|motocicl|bicicl|ciclist|peaton|transit|movilidad|"
                     r"\bvias?\b|calles?\b|semafor|camaras?|fotodetec|velocidad|conductor|licencia|comparendo|multa|"
                     r"transmilenio|sitp|buses|taxis?\b|colegio|resalto|ciclorruta|kilometros")
NO_DATO = ("No tengo ese dato. Puedo responder con estas fuentes:\n"
           "- **Siniestros viales** de la Secretaría Distrital de Movilidad (2010 a la fecha): siniestros, muertos y heridos "
           "por año, localidad o vía. Ej.: «¿Cuántos muertos hubo en Suba en 2024?»\n"
           "- **Efecto de medidas** (límites de velocidad, resaltos, cámaras, carriles preferenciales, zonas escolares…), "
           "medido en Bogotá y en estudios internacionales.\n"
           "- **Normas** cargadas en la pestaña *Normas*.\n\n"
           "Cifras como el número de vehículos registrados (parque automotor) las publican el RUNT y el Observatorio de "
           "Movilidad de la Secretaría de Movilidad.")


def es_pregunta_de_dato(msg: str, m: dict) -> bool:
    """«¿Cuántos vehículos hay en Bogotá?»: pregunta por una cifra que no es de siniestros."""
    t = plano(msg)
    return bool(RE_CUANTOS.search(t)) and not m["consulta_datos"] and not RE_SINIESTRO.search(t) \
        and not (m["direccion"] or m["valor_propuesto"])


def es_pregunta_normativa(msg: str, m: dict) -> bool:
    t = plano(msg)
    if RE_NORMA.search(t):
        return True
    return bool(RE_PREGUNTA.search(t)) and (m["en_tema"] or RE_VIAL.search(t)) and not (m["direccion"] or m["valor_propuesto"] or m["consulta_datos"]
                                               or m["corredor"] or m["localidad"] or RE_EVALUAR.search(t))


class AsistenteNormas:
    """Envuelve el RAG. Se carga en segundo plano para no demorar el arranque de la app."""

    def __init__(self):
        self.rag, self.error, self.listo = None, None, threading.Event()
        threading.Thread(target=self._cargar, daemon=True).start()

    def _cargar(self):
        try:
            from rag.pipeline import RAG
            r = RAG()
            self.rag = r
            if r.hay_indice:
                r.almacen                  # índice FAISS
            r.embedder                     # modelo de embeddings
            r.asegurar_modelo()            # si cambió el modelo (p. ej. poca memoria → modelo ligero), reindexa
            modo = f"Claude ({r.config.modelo_llm})" if r.config.hay_llm else "sin LLM (falta ANTHROPIC_API_KEY)"
            if r.hay_indice:
                print(f"  normas (RAG): {r.almacen.info['n_fragmentos']} fragmentos de "
                      f"{len(r.almacen.info['documentos'])} documentos · {modo}")
            else:
                print(f"  normas (RAG): sin índice todavía (suba e indexe normas en la pestaña Normas) · {modo}")
        except ImportError as e:
            self.error = f"faltan librerías del RAG ({e.name}). Ejecute rag.bat o: pip install -r requirements_rag.txt"
        except Exception as e:
            self.error = str(e)
        finally:
            if self.error:
                print(f"  normas (RAG): no disponible — {self.error}")
            self.listo.set()

    def responder(self, pregunta: str, historial: list) -> dict | None:
        """Respuesta con fuentes, o None si no hay fragmentos relevantes."""
        self.listo.wait(timeout=120)
        if not self.rag:
            return {"respuesta": f"No puedo consultar las normas en este momento: {self.error or 'el RAG sigue cargando'}."}
        if not self.rag.hay_indice:
            return {"respuesta": "Todavía no hay normas indexadas. Suba los PDF en la pestaña **Normas** y pulse **Indexar**."}
        consulta = pregunta
        previas = [h.get("content", "") for h in historial if isinstance(h, dict) and h.get("role") == "user"]
        if previas and len(pregunta.split()) < 6:            # «¿y el artículo 5?»: se apoya en la pregunta anterior
            consulta = f"{previas[-1]} {pregunta}"
        recuperados = self.rag.recuperar(consulta)
        if not recuperados:
            return None
        try:
            texto = self.rag.generador.responder(pregunta, recuperados)
        except Exception as e:                               # API caída, clave inválida, sin saldo, librería vieja…
            import traceback
            traceback.print_exc()
            from rag.llm import GeneradorExtractivo, explicar_error
            texto = (f"⚠️ No pude usar Claude: {explicar_error(e)}\n\n"
                     + GeneradorExtractivo(nota="Mientras tanto, estos son los fragmentos más relevantes:").responder(pregunta, recuperados))
        citadas = {int(n) for n in re.findall(r"\[(\d+)\]", texto)}
        fuentes, vistas = [], set()
        for n, (f, sim) in enumerate(recuperados, 1):
            if citadas and n not in citadas:
                continue
            linea = f"- [{n}] {f.titulo} — {f.seccion}"
            if f.metadatos.get("estado"):
                linea += f" · *{f.metadatos['estado'].rstrip('.')}*"
            if f.metadatos.get("url") and f.doc_id not in vistas:
                linea += f" · [ver norma]({f.metadatos['url']})"
                vistas.add(f.doc_id)
            fuentes.append(linea)
        if texto.startswith("No encuentro"):
            return self.sin_resultado()
        md = texto + ("\n\n**Fuentes**\n" + "\n".join(fuentes) if fuentes else "")
        return {"respuesta": md, "tipo": "normas"}


    def sin_resultado(self) -> dict:
        docs = self.rag.almacen.info.get("documentos", []) if self.rag and self.rag.hay_indice else []
        return {"respuesta": "No encuentro eso en las normas cargadas.\n\nNormas que tengo: " + "; ".join(docs)
                + ".\n\nPara ampliar, copie el PDF de la norma en *data/resoluciones* y ejecute *rag.bat indexar*.",
                "tipo": "normas"}


NORMAS: AsistenteNormas | None = None
EXT_NORMAS = (".pdf", ".docx", ".txt")
TRABAJO = {"activo": False, "mensaje": "", "error": None, "resultados": [], "total": 0, "hechos": 0}
_lock_trabajo = threading.Lock()


def _carpeta_normas() -> str:
    if NORMAS and NORMAS.rag:
        return NORMAS.rag.config.carpeta_documentos
    from rag.config import cargar_config
    return cargar_config().carpeta_documentos


def _archivo_valido(archivo: str) -> str | None:
    """Ruta relativa dentro de la carpeta de normas (evita salir de ella con «..»)."""
    base = os.path.realpath(_carpeta_normas())
    ruta = os.path.realpath(os.path.join(base, archivo or ""))
    if not ruta.startswith(base + os.sep) or not os.path.isfile(ruta):
        return None
    return os.path.relpath(ruta, base).replace(os.sep, "/")


@app.get("/api/normas")
def normas_estado():
    NORMAS.listo.wait(timeout=1)
    r = NORMAS.rag
    base = {"trabajo": TRABAJO, "error": NORMAS.error, "cargando": not NORMAS.listo.is_set()}
    if not r:
        return jsonify({**base, "documentos": []})
    info = r.almacen.info if r.hay_indice else {}
    try:
        docs = r.estado_documentos()
    except Exception as e:
        docs, base["error"] = [], str(e)
    return jsonify({**base, "documentos": docs, "n_fragmentos": info.get("n_fragmentos", 0),
                    "modelo_embeddings": (r._embedder.nombre if r._embedder else r.config.modelo_embeddings),
                    "aviso_embeddings": getattr(r._embedder, "aviso", ""),
                    "llm": r.config.modelo_llm if r.config.hay_llm else None})


@app.post("/api/normas/subir")
def normas_subir():
    destino = os.path.join(_carpeta_normas(), "subidos")
    os.makedirs(destino, exist_ok=True)
    guardados, rechazados = [], []
    for f in request.files.getlist("archivos"):
        nombre = os.path.basename((f.filename or "").replace("\\", "/"))
        tronco, ext = os.path.splitext(nombre)
        if ext.lower() not in EXT_NORMAS:
            rechazados.append(f"{nombre}: solo se aceptan PDF, Word (.docx) o texto (.txt)")
            continue
        tronco = re.sub(r"[^\w\-. ()]", "_", tronco).strip(" .")[:120] or "norma"
        f.save(os.path.join(destino, tronco + ext.lower()))
        guardados.append(f"subidos/{tronco}{ext.lower()}")
    return jsonify(guardados=guardados, rechazados=rechazados)


def _correr_indexacion(archivos: list[str] | None):
    r = NORMAS.rag
    avance = lambda m: TRABAJO.update(mensaje=m)
    try:
        if r._embedder is None:
            avance("Cargando el modelo de embeddings (la primera vez se descarga, ~470 MB)…")
            r.embedder
        if archivos is None:
            res = r.indexar(avance)
            TRABAJO["resultados"] = [{"ok": True, "archivo": "todos", "fragmentos": res["fragmentos"]}]
            TRABAJO["mensaje"] = f"Listo: {len(res['documentos'])} documentos, {res['fragmentos']} fragmentos indexados."
        else:
            for a in archivos:
                TRABAJO["resultados"].append(r.indexar_archivo(a, avance))
                TRABAJO["hechos"] += 1
            ok = [x for x in TRABAJO["resultados"] if x["ok"]]
            TRABAJO["mensaje"] = (f"Listo: {len(ok)} de {len(archivos)} documentos indexados "
                                  f"({sum(x['fragmentos'] for x in ok)} fragmentos).")
    except Exception as e:
        TRABAJO["error"] = f"{type(e).__name__}: {e}"
        TRABAJO["mensaje"] = "La indexación falló."
    finally:
        TRABAJO["activo"] = False


@app.post("/api/normas/indexar")
def normas_indexar():
    if not NORMAS.listo.wait(timeout=30) or not NORMAS.rag:
        return jsonify(error=NORMAS.error or "El RAG todavía está cargando; intente en unos segundos."), 503
    d = request.json or {}
    if d.get("todos"):
        archivos = None
    else:
        archivos = [a for a in (_archivo_valido(x) for x in d.get("archivos") or []) if a]
        if not archivos:
            return jsonify(error="No hay documentos para indexar."), 400
    with _lock_trabajo:
        if TRABAJO["activo"]:
            return jsonify(error="Ya hay una indexación en curso."), 409
        TRABAJO.update(activo=True, mensaje="Empezando…", error=None, resultados=[],
                       total=len(archivos) if archivos else 0, hechos=0)
    threading.Thread(target=_correr_indexacion, args=(archivos,), daemon=True).start()
    return jsonify(ok=True), 202


@app.post("/api/normas/quitar")
def normas_quitar():
    if not NORMAS.rag:
        return jsonify(error=NORMAS.error or "El RAG no está disponible."), 503
    if TRABAJO["activo"]:
        return jsonify(error="Espere a que termine la indexación en curso."), 409
    a = _archivo_valido((request.json or {}).get("archivo"))
    if not a:
        return jsonify(error="Documento no encontrado."), 404
    NORMAS.rag.quitar_archivo(a, borrar=True)
    return jsonify(ok=True)


GENERICA = "Puedo ayudarte con eso si me dices **qué medida**"


@app.post("/api/chat")
def chat():
    d = request.json or {}
    msg, ctx, hist = d.get("mensaje", ""), d.get("contexto"), d.get("historial") or []
    if ctx:
        r = responder_con_contexto(msg, ctx)
        if r is not None:
            return jsonify(r)
    m = extraer_reglas(msg)
    if es_pregunta_de_dato(msg, m) and not m["otra_ciudad"]:
        if not (m["en_tema"] or RE_VIAL.search(plano(msg))):
            return jsonify(respuesta=D.agente.responder(msg, []))        # fuera del tema
        r = NORMAS.responder(msg, hist) if NORMAS else None              # quizá lo dice alguna norma cargada
        if r and not r["respuesta"].startswith(("No encuentro", "No puedo consultar", "Todavía no")):
            return jsonify(r)
        return jsonify(respuesta=NO_DATO)
    normativa = NORMAS and not es_ayuda(msg) and not m["otra_ciudad"] and es_pregunta_normativa(msg, m)
    if normativa:                                  # preguntas sobre normas → RAG
        r = NORMAS.responder(msg, hist)
        return jsonify(r or NORMAS.sin_resultado())
    texto = D.agente.responder(msg, hist)          # evaluar medidas y cifras → motor de efectos
    if NORMAS and not normativa and m["en_tema"] and texto.startswith(GENERICA):
        r = NORMAS.responder(msg, hist)            # en tema pero sin medida concreta: se intenta en las normas
        if r and not r["respuesta"].startswith(("No encuentro", "No puedo consultar")):
            return jsonify(r)
    salida = {"respuesta": texto}
    if D.agente.ultima:
        p = propuesta_de_chat(D.agente.ultima)
        if p:
            try:                       # misma cuenta que el mapa (siniestros dentro de la zona exacta)
                salida["respuesta"] = evaluar_propuesta(p["zona"]["features"], p["resolucion"])["markdown"]
            except ErrorPropuesta:
                pass
            salida["propuesta"] = p
            salida["contexto"] = p
    return jsonify(salida)


def _num(v):
    try:
        x = float(v)
        return x if x > 0 else None
    except (TypeError, ValueError):
        return None


def iniciar() -> Flask:
    """Carga datos y normas una vez. La usan app_web.py (local) y wsgi.py (servidor, p. ej. Render)."""
    global D, NORMAS
    if D is None:
        from entorno import resumen_claves
        for linea in resumen_claves():
            print("  " + linea)
        D = Datos()
        NORMAS = AsistenteNormas()
    return app


if __name__ == "__main__":
    import socket

    def _ocupado(p: int) -> bool:
        with socket.socket() as so:
            so.settimeout(0.5)
            return so.connect_ex(("127.0.0.1", p)) == 0

    # En Windows el servidor de Flask puede abrir un puerto que ya usa otra copia de la app
    # (SO_REUSEADDR): el navegador sigue hablando con la copia vieja y la página no carga.
    puerto = int(os.environ.get("PUERTO", 5000))
    if _ocupado(puerto):
        libre = next((p for p in range(puerto + 1, puerto + 30) if not _ocupado(p)), None)
        print(f"\n⚠️  Ya hay otro programa (probablemente otra ventana de VíaSegura IA) usando el puerto {puerto}.")
        print("   Cierre las otras ventanas negras de la app, o finalice «python.exe» en el Administrador de tareas.")
        if libre is None:
            sys.exit(1)
        print(f"   Mientras tanto, esta copia se abre en el puerto {libre}.\n")
        puerto = libre

    iniciar()
    if "--sin-navegador" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{puerto}")).start()
    print(f"\nVíaSegura IA en http://127.0.0.1:{puerto}  (Ctrl+C para cerrar)")
    app.run(host="127.0.0.1", port=puerto, debug=False, threaded=True)
