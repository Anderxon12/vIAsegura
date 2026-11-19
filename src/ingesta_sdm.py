"""
=============================================================
  INGESTA SDM — Descarga y actualización de datos de movilidad
  Bogotá D.C. · Secretaría Distrital de Movilidad / Datos Abiertos
=============================================================

Qué hace
    1. Consulta el portal Datos Abiertos Bogotá (API CKAN) para cada fuente
       de la lista FUENTES.
    2. Si la fuente tiene un servicio "Esri REST" (ArcGIS), descarga TODAS
       sus capas y tablas completas (siniestros, actores, vehículos, causas…),
       paginando por ObjectID.
    3. Si no, descarga el archivo publicado (XLSX / CSV / ZIP / KML).
    4. Guarda todo en data/raw/<fuente>/ y escribe:
         data/raw/manifiesto.json   qué se bajó, cuándo, filas y columnas
         data/raw/esquema.json      columnas y tipos de cada tabla

Uso
    pip install requests pandas openpyxl
    python src/ingesta_sdm.py                  # descarga todas las fuentes
    python src/ingesta_sdm.py --listar         # muestra recursos sin descargar
    python src/ingesta_sdm.py --solo siniestralidad_bd camaras_salvavidas malla_vial
    python src/ingesta_sdm.py --buscar "fotodeteccion" "resaltos" "zona escolar"
                                               # busca fuentes nuevas en el portal

Notas
    - Es idempotente: cada ejecución reemplaza la descarga anterior de la
      fuente y actualiza el manifiesto. Para actualizar la data, se vuelve
      a ejecutar.
    - Los nombres (slug) de FUENTES fueron verificados en el portal en
      noviembre de 2026. Si alguno cambia, use --buscar para encontrarlo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RAW = os.path.join(RAIZ, "data", "raw")
CKAN = "https://datosabiertos.bogota.gov.co/api/3/action"
HUB = "https://datos.movilidadbogota.gov.co"
CABECERAS = {"User-Agent": "accidentes-bogota-ingesta/1.0 (proyecto academico)"}
PAUSA = 0.25          # segundos entre peticiones, para no saturar el servidor
LOTE_IDS = 1000       # ObjectIDs por petición (se ajusta al máximo del servidor)

SIMUR = "https://sig.simur.gov.co/arcgis/rest/services"
AGOL = "https://services2.arcgis.com/NEwhEo9GGSHXcRXV/arcgis/rest/services"
CATASTRO = "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services"

# Servicios ArcGIS de la SDM: se descargan directo, sin pasar por el portal.
# (encontrados con --buscar en noviembre de 2026)
FUENTES_ARCGIS = {
    "siniestralidad_bd":       f"{SIMUR}/Accidentalidad/AccidentalidadAnalisis/FeatureServer",
    "camaras_salvavidas":      f"{SIMUR}/DatosAbiertos/CamarasSalvavidas/FeatureServer",
    "resaltos_parabolicos":    f"{SIMUR}/DatosAbiertos/Proyecto_ResaltosParabolicos/FeatureServer",
    "malla_vial":              f"{AGOL}/Malla_Vial_Integral_Bogota_D_C/FeatureServer",
    "carril_preferencial_sitp": f"{AGOL}/Carril_Preferencial_SITP_Bogota_D_C/FeatureServer",
    "velocidad_2020_09":       f"{AGOL}/Velocidades_Bitcarrier_Septiembre_2020/FeatureServer",
    "velocidad_2020_10":       f"{AGOL}/Velocidades_Bitcarrier_Octubre_2020/FeatureServer",
    "velocidad_2020_11":       f"{AGOL}/Velocidades_Bitcarrier_Noviembre_2020/FeatureServer",
    "velocidad_2020_12":       f"{AGOL}/Velocidades_Bitcarrier_Diciembre_2020/FeatureServer",
    "velocidad_2021_09":       f"{AGOL}/Velocidades_Bitcarrier_Septiembre_2021/FeatureServer",
    "puntos_criticos_arterial": f"{SIMUR}/DatosAbiertos/PuntosCriticosVulnerables_ViaArterial_2225/FeatureServer",
    "tramos_criticos_arterial": f"{SIMUR}/DatosAbiertos/TramosCriticosVulnerables_ViaArterial_2225/FeatureServer",
    "tramos_criticos_local":   f"{SIMUR}/DatosAbiertos/TramosCriticosVulnerables_Local_2225/FeatureServer",
    "zonas_criticas_local":    f"{SIMUR}/DatosAbiertos/ZonasCriticasVulnerables_Local_2225/FeatureServer",
    # Mapa de Referencia de Catastro: SOLO las capas útiles (el servicio completo pesa ~3,5 GB)
    "ciclorruta":              (f"{CATASTRO}/Mapa_Referencia/Mapa_Referencia/MapServer", [18]),
    "estaciones_transmilenio": (f"{CATASTRO}/Mapa_Referencia/Mapa_Referencia/MapServer", [9]),
    # límites para el mapa de la aplicación: 48 = localidades, 37 = sectores catastrales (barrios)
    "limites":                 (f"{CATASTRO}/Mapa_Referencia/Mapa_Referencia/MapServer", [48, 37]),
}
# Estas fuentes cambian cada mes: se vuelven a descargar siempre. Las demás se
# saltan si ya están descargadas (use --forzar para bajarlas de nuevo).
ACTUALIZAR_SIEMPRE = {"siniestralidad_bd", "camaras_salvavidas", "resaltos_parabolicos"}

# clave local → slug en datosabiertos.bogota.gov.co (portal CKAN)
FUENTES = {
    "colegios":               "colegios-bogota-d-c",
    "siniestros_historico":   "historico-siniestros-bogota-d-c",
    "siniestros_consolidado": "siniestros-viales-consolidados-bogota-d-c",
    "anuario_siniestralidad": "anuario-siniestralidad",
    "factores_riesgo":        "factores-de-riesgos-involucrados-en-la-siniestralidad-vial-de-bogota-dc",
    "conteo_vehiculos":       "conteo-vehiculos-cgt-bogota-d-c",
    "conteo_bicicletas":      "sensores-conteo-bicicleta-bogota-d-c",
}
TODAS = list(FUENTES_ARCGIS) + list(FUENTES)

# formatos de archivo que se descargan cuando no hay servicio Esri REST
FORMATOS_ARCHIVO = ["XLSX", "XLS", "CSV", "GEOJSON", "ZIP", "KML"]

sesion = requests.Session()
sesion.headers.update(CABECERAS)


# ════════════════════════════════════════════════════════════
#  UTILIDADES HTTP
# ════════════════════════════════════════════════════════════
def pedir(url, params=None, metodo="get", intentos=4, **kw):
    """Petición HTTP con reintentos y espera creciente."""
    ultimo = None
    for i in range(intentos):
        try:
            r = sesion.request(metodo, url, params=params if metodo == "get" else None,
                               data=params if metodo == "post" else None, timeout=120, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            time.sleep(PAUSA)
            return r
        except (requests.RequestException, ValueError) as e:
            ultimo = e
            time.sleep(2 ** i)
    raise RuntimeError(f"No se pudo consultar {url}: {ultimo}")


def pedir_json(url, params=None, metodo="get", intentos=4):
    datos = pedir(url, params, metodo, intentos=intentos).json()
    if isinstance(datos, dict) and "error" in datos and "features" not in datos:
        raise RuntimeError(f"El servicio devolvió un error en {url}: {datos['error']}")
    return datos


def nombre_seguro(texto: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(texto)).strip("_")[:80] or "sin_nombre"


# ════════════════════════════════════════════════════════════
#  CKAN (datosabiertos.bogota.gov.co)
# ════════════════════════════════════════════════════════════
def ckan_conjunto(slug: str) -> dict:
    """package_show responde 403 a scripts; package_search sí funciona y trae lo mismo."""
    try:
        return pedir_json(f"{CKAN}/package_show", {"id": slug}, intentos=1)["result"]
    except Exception:
        res = pedir_json(f"{CKAN}/package_search", {"fq": f"name:{slug}", "rows": 1})["result"]["results"]
        if not res:
            res = pedir_json(f"{CKAN}/package_search", {"q": slug.replace("-", " "), "rows": 5})["result"]["results"]
            res = [r for r in res if r.get("name") == slug] or res[:0]
        if not res:
            raise RuntimeError(f"El conjunto '{slug}' no existe en el portal (pruebe --buscar)")
        return res[0]


def ckan_buscar(termino: str, filas: int = 25) -> list[dict]:
    res = pedir_json(f"{CKAN}/package_search", {"q": termino, "rows": filas})["result"]["results"]
    return [{"termino": termino, "slug": p["name"], "titulo": p.get("title"),
             "entidad": (p.get("organization") or {}).get("title"),
             "actualizado": p.get("metadata_modified"),
             "formatos": ", ".join(sorted({(r.get("format") or "").upper() for r in p.get("resources", [])}))}
            for p in res]


def hub_buscar(termino: str, filas: int = 25) -> list[dict]:
    """Búsqueda en el portal ArcGIS Hub de Movilidad (mejor esfuerzo)."""
    try:
        d = pedir_json(f"{HUB}/api/search/v1/collections/dataset/items", {"q": termino, "limit": filas})
        return [{"termino": termino, "slug": f.get("id"), "titulo": f["properties"].get("title"),
                 "entidad": "ArcGIS Hub Movilidad", "actualizado": f["properties"].get("modified"),
                 "formatos": f["properties"].get("url") or ""} for f in d.get("features", [])]
    except Exception as e:  # el Hub cambia de versión con frecuencia
        print(f"   (Hub no disponible para '{termino}': {e})")
        return []


# ════════════════════════════════════════════════════════════
#  ARCGIS REST
# ════════════════════════════════════════════════════════════
def raiz_servicio(url: str) -> str:
    """De .../FeatureServer/0/query?... devuelve .../FeatureServer"""
    m = re.search(r"^(.*?/(?:FeatureServer|MapServer))", url, flags=re.I)
    return m.group(1) if m else url.split("?")[0].rstrip("/")


def es_arcgis(recurso: dict) -> bool:
    url = (recurso.get("url") or "")
    return bool(re.search(r"/(FeatureServer|MapServer)(/|$|\?)", url, flags=re.I))


def geometria_a_columnas(geom: dict | None) -> dict:
    """Punto → lon/lat. Líneas y polígonos → centroide aproximado + JSON."""
    if not geom:
        return {"lon": None, "lat": None}
    if "x" in geom and "y" in geom:
        return {"lon": geom["x"], "lat": geom["y"]}
    partes = geom.get("paths") or geom.get("rings") or []
    pts = [p for parte in partes for p in parte]
    if not pts:
        return {"lon": None, "lat": None}
    return {"lon": sum(p[0] for p in pts) / len(pts), "lat": sum(p[1] for p in pts) / len(pts),
            "geometria_json": json.dumps(geom, separators=(",", ":"))}


def descargar_capa(url_capa: str) -> pd.DataFrame:
    """Descarga todos los registros de una capa o tabla ArcGIS."""
    meta = pedir_json(url_capa, {"f": "json"})
    campo_id = meta.get("objectIdField") or next(
        (c["name"] for c in meta.get("fields", []) if c.get("type") == "esriFieldTypeOID"), "OBJECTID")
    fechas = [c["name"] for c in meta.get("fields", []) if c.get("type") == "esriFieldTypeDate"]
    con_geom = meta.get("type") != "Table"

    ids = pedir_json(f"{url_capa}/query", {"where": "1=1", "returnIdsOnly": "true", "f": "json"}).get("objectIds") or []
    ids.sort()
    lote_n = max(100, min(LOTE_IDS, int(meta.get("maxRecordCount") or LOTE_IDS)))
    filas = []
    for i in range(0, len(ids), lote_n):
        lote = ids[i:i + lote_n]
        d = pedir_json(f"{url_capa}/query", {
            "objectIds": ",".join(map(str, lote)), "outFields": "*", "outSR": 4326,
            "returnGeometry": "true" if con_geom else "false", "f": "json"}, metodo="post")
        for f in d.get("features", []):
            fila = dict(f.get("attributes") or {})
            if con_geom:
                fila.update(geometria_a_columnas(f.get("geometry")))
            filas.append(fila)
        if (i // lote_n) % 20 == 0 or i + lote_n >= len(ids):
            print(f"      {min(i + lote_n, len(ids)):>8,} / {len(ids):,}", flush=True)
    df = pd.DataFrame(filas)
    for c in fechas:                       # epoch en milisegundos → fecha
        if c in df.columns:
            # se deja en UTC, tal como lo publica el servicio: convertir a hora de Bogotá
            # correría un día las fechas guardadas como medianoche
            df[c] = pd.to_datetime(df[c], unit="ms", errors="coerce")
    if campo_id in df.columns and df[campo_id].duplicated().any():
        df = df.drop_duplicates(campo_id)
    return df


def descargar_servicio(url: str, carpeta: str, capas: list[int] | None = None) -> list[dict]:
    """Descarga las capas y tablas de un FeatureServer / MapServer (todas, o solo `capas`)."""
    raiz = raiz_servicio(url)
    meta = pedir_json(raiz, {"f": "json"})
    elementos = [(e, "capa") for e in meta.get("layers", [])] + [(e, "tabla") for e in meta.get("tables", [])]
    if capas is not None:
        elementos = [(e, t) for e, t in elementos if e["id"] in capas]
    # un servicio de mapas con muchas capas puede pesar gigas: se avisa antes
    if capas is None and len(elementos) > 15:
        print(f"   ⚠ el servicio tiene {len(elementos)} capas; se descargan solo las 15 primeras. "
              f"Indique las capas útiles en FUENTES_ARCGIS como (url, [ids]).")
        elementos = elementos[:15]
    if not elementos:                      # la URL ya era una capa suelta
        m = re.search(r"/(\d+)/?$", url.split("?")[0])
        elementos = [({"id": int(m.group(1)) if m else 0, "name": meta.get("name", "capa")}, "capa")]
    salidas = []
    for e, tipo in elementos:
        nombre = nombre_seguro(f"{e['id']}_{e.get('name', tipo)}")
        print(f"   · {tipo} {e['id']}: {e.get('name')}")
        try:
            df = descargar_capa(f"{raiz}/{e['id']}")
        except Exception as ex:
            print(f"      ⚠ no se pudo descargar: {ex}")
            salidas.append({"tabla": nombre, "error": str(ex)})
            continue
        ruta = os.path.join(carpeta, f"{nombre}.csv")
        df.to_csv(ruta, index=False, encoding="utf-8-sig")
        salidas.append(resumen_tabla(nombre, ruta, df, origen=f"{raiz}/{e['id']}"))
    return salidas


# ════════════════════════════════════════════════════════════
#  ARCHIVOS (XLSX / CSV / ZIP / KML)
# ════════════════════════════════════════════════════════════
def descargar_archivo(recurso: dict, carpeta: str) -> dict:
    url = recurso["url"]
    ext = (recurso.get("format") or os.path.splitext(url)[1].lstrip(".") or "dat").lower()
    nombre = nombre_seguro(recurso.get("name") or recurso.get("id")) + "." + ext
    ruta = os.path.join(carpeta, nombre)
    r = pedir(url, stream=True)
    sha = hashlib.sha256()
    with open(ruta, "wb") as f:
        for trozo in r.iter_content(1 << 20):
            f.write(trozo)
            sha.update(trozo)
    info = {"tabla": nombre, "archivo": os.path.relpath(ruta, RAIZ), "origen": url,
            "bytes": os.path.getsize(ruta), "sha256": sha.hexdigest()}
    try:                                   # hojas y columnas, si es tabular
        if ext in ("xlsx", "xls"):
            hojas = pd.read_excel(ruta, sheet_name=None, nrows=200)
            info["hojas"] = {h: list(map(str, d.columns)) for h, d in hojas.items()}
        elif ext == "csv":
            info["columnas"] = list(pd.read_csv(ruta, nrows=200, sep=None, engine="python",
                                                encoding_errors="replace").columns)
    except Exception as ex:
        info["aviso"] = f"no se pudo leer el esquema: {ex}"
    return info


def resumen_tabla(nombre, ruta, df, origen) -> dict:
    info = {"tabla": nombre, "archivo": os.path.relpath(ruta, RAIZ), "origen": origen,
            "filas": int(len(df)), "columnas": {c: str(t) for c, t in df.dtypes.items()}}
    for c in df.columns:                   # rango de fechas, útil para saber el corte
        if "datetime" in str(df[c].dtype) and df[c].notna().any():
            info.setdefault("rango_fechas", {})[c] = [str(df[c].min()), str(df[c].max())]
    return info


# ════════════════════════════════════════════════════════════
#  ORQUESTACIÓN
# ════════════════════════════════════════════════════════════
def procesar_fuente(clave: str, slug: str, solo_listar: bool) -> dict:
    print(f"\n■ {clave}  ({slug})")
    conjunto = ckan_conjunto(slug)
    recursos = conjunto.get("resources", [])
    reg = {"slug": slug, "titulo": conjunto.get("title"),
           "metadatos_actualizados": conjunto.get("metadata_modified"),
           "descargado": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tablas": []}
    for r in recursos:
        print(f"   - [{(r.get('format') or '?').upper():8}] {r.get('name')}  →  {r.get('url')}")
    if solo_listar:
        return reg

    carpeta = os.path.join(DIR_RAW, clave)
    os.makedirs(carpeta, exist_ok=True)
    arcgis = [r for r in recursos if es_arcgis(r)]
    if arcgis:
        hechos = set()
        for r in arcgis:
            raiz = raiz_servicio(r["url"])
            if raiz not in hechos:
                hechos.add(raiz)
                reg["tablas"] += descargar_servicio(r["url"], carpeta)
    # archivos: siempre los XLSX/XLS (traen hojas que el servicio no tiene);
    # el resto solo si no hubo servicio ArcGIS
    for fmt in FORMATOS_ARCHIVO:
        if arcgis and fmt not in ("XLSX", "XLS"):
            continue
        candidatos = [r for r in recursos if (r.get("format") or "").upper() == fmt and not es_arcgis(r)]
        for r in candidatos:
            try:
                print(f"   · archivo {fmt}: {r.get('name')}")
                reg["tablas"].append(descargar_archivo(r, carpeta))
            except Exception as ex:
                print(f"      ⚠ {ex}")
                reg["tablas"].append({"tabla": r.get("name"), "error": str(ex)})
        if candidatos and not arcgis:
            break                          # con el mejor formato disponible basta
    if not reg["tablas"]:
        print("   ⚠ no se encontró ningún recurso descargable")
    return reg


def procesar_arcgis(clave: str, url, solo_listar: bool) -> dict:
    url, capas = (url if isinstance(url, tuple) else (url, None))
    print(f"\n■ {clave}  ({url}{' capas ' + str(capas) if capas else ''})")
    reg = {"servicio": url, "descargado": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tablas": []}
    meta = pedir_json(url, {"f": "json"})
    for e in meta.get("layers", []) + meta.get("tables", []):
        print(f"   - {e['id']}: {e.get('name')}")
    if solo_listar:
        return reg
    carpeta = os.path.join(DIR_RAW, clave)
    os.makedirs(carpeta, exist_ok=True)
    reg["tablas"] = descargar_servicio(url, carpeta, capas)
    for t in reg["tablas"]:   # algunos servicios publican como máximo 1.000.000 de filas
        if t.get("filas") == 1_000_000:
            print(f"   ⚠ {t['tabla']}: exactamente 1.000.000 filas; probablemente el servicio está truncado.")
            t["aviso"] = "posiblemente truncado en 1.000.000 filas"
    return reg


def main():
    ap = argparse.ArgumentParser(description="Ingesta de datos de movilidad de Bogotá (SDM).")
    ap.add_argument("--solo", nargs="+", metavar="FUENTE", help=f"fuentes a procesar: {', '.join(TODAS)}")
    ap.add_argument("--listar", action="store_true", help="solo muestra los recursos, no descarga")
    ap.add_argument("--buscar", nargs="+", metavar="TERMINO", help="busca conjuntos de datos en el portal")
    ap.add_argument("--forzar", action="store_true", help="vuelve a descargar también lo que ya está descargado")
    args = ap.parse_args()
    os.makedirs(DIR_RAW, exist_ok=True)

    if args.buscar:
        filas = []
        for t in args.buscar:
            filas += ckan_buscar(t) + hub_buscar(t)
        cat = pd.DataFrame(filas).drop_duplicates(["slug"]) if filas else pd.DataFrame()
        ruta = os.path.join(DIR_RAW, "catalogo_busqueda.csv")
        cat.to_csv(ruta, index=False, encoding="utf-8-sig")
        print(cat.to_string(index=False, max_colwidth=60) if len(cat) else "Sin resultados.")
        print(f"\n✅ {len(cat)} resultados guardados en {ruta}")
        print("   Para descargar uno, agregue su slug al diccionario FUENTES.")
        return

    claves = args.solo or TODAS
    desconocidas = [c for c in claves if c not in TODAS]
    if desconocidas:
        sys.exit(f"Fuente desconocida: {desconocidas}. Opciones: {TODAS}")

    ruta_man = os.path.join(DIR_RAW, "manifiesto.json")
    manifiesto = json.load(open(ruta_man, encoding="utf-8")) if os.path.exists(ruta_man) else {}
    errores = []
    for clave in claves:
        previo = manifiesto.get(clave, {})
        ya = previo.get("tablas") and all("error" not in t for t in previo["tablas"]) and \
            os.path.isdir(os.path.join(DIR_RAW, clave))
        if ya and not args.forzar and not args.listar and clave not in ACTUALIZAR_SIEMPRE and not args.solo:
            print(f"\n■ {clave}: ya descargado ({previo.get('descargado')}); se salta (use --forzar para actualizar)")
            continue
        try:
            reg = (procesar_arcgis(clave, FUENTES_ARCGIS[clave], args.listar) if clave in FUENTES_ARCGIS
                   else procesar_fuente(clave, FUENTES[clave], args.listar))
            if not args.listar:
                manifiesto[clave] = reg
                with open(ruta_man, "w", encoding="utf-8") as f:
                    json.dump(manifiesto, f, ensure_ascii=False, indent=1, default=str)
        except Exception as ex:
            errores.append(clave)
            print(f"   ✖ {clave}: {ex}")
    if args.listar:
        return

    with open(ruta_man, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, ensure_ascii=False, indent=1, default=str)
    esquema = {c: {t.get("tabla"): t.get("columnas") or t.get("hojas") for t in m.get("tablas", [])}
               for c, m in manifiesto.items()}
    with open(os.path.join(DIR_RAW, "esquema.json"), "w", encoding="utf-8") as f:
        json.dump(esquema, f, ensure_ascii=False, indent=1, default=str)

    print("\n══════════ RESUMEN ══════════")
    for clave in claves:
        for t in manifiesto.get(clave, {}).get("tablas", []):
            det = f"{t['filas']:,} filas" if "filas" in t else (f"{t.get('bytes', 0):,} bytes" if "bytes" in t else f"ERROR {t.get('error')}")
            print(f"  {clave:24} {str(t.get('tabla'))[:40]:40} {det}")
    print(f"\n✅ Manifiesto: {ruta_man}")
    if errores:
        print(f"⚠ Fuentes con error: {errores}. Pruebe --listar o --buscar para revisar el slug.")


if __name__ == "__main__":
    main()
