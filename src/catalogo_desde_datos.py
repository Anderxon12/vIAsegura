"""
=============================================================
  CATÁLOGO DESDE DATOS — Intervenciones con fecha y lugar
  publicadas por la SDM (cámaras salvavidas, resaltos)
=============================================================

Escribe data/intervenciones/catalogo_datos.csv con una fila por
intervención, lista para efecto_intervencion.py --catalogo.

  · Cámaras salvavidas (fotodetección): punto, fecha de instalación y de
    desmonte. Se agrupan por trimestre de instalación (cámaras del mismo
    trimestre se estiman juntas).
  · Resaltos parabólicos: tramo (línea), fecha de intervención (texto,
    se toma la primera fecha que aparece).

Las filas salen con revisado = True porque vienen directo de datos
oficiales; aun así, revise las fechas de los resaltos (vienen en texto).

Uso
    python src/catalogo_desde_datos.py
"""

from __future__ import annotations

import json
import os
import re

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(RAIZ, "data", "raw")
SALIDA = os.path.join(RAIZ, "data", "intervenciones", "catalogo_datos.csv")
URL_CAM = "https://sig.simur.gov.co/arcgis/rest/services/DatosAbiertos/CamarasSalvavidas/FeatureServer"
URL_RES = "https://sig.simur.gov.co/arcgis/rest/services/DatosAbiertos/Proyecto_ResaltosParabolicos/FeatureServer"
MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
         "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}
COLS = ["id_intervencion", "grupo", "tipo_medida", "direccion", "valor_anterior", "valor_nuevo", "unidad",
        "corredor", "localidad", "fecha_inicio", "fecha_sanciones", "fin_ventana", "norma", "fuente", "nota",
        "geometria_geojson", "radio_m", "celdas_ids", "excluir_ids", "meses_transicion", "revisado"]


def camaras() -> list[dict]:
    ruta = os.path.join(RAW, "camaras_salvavidas", "0_CamarasSalvavidas.csv")
    if not os.path.exists(ruta):
        print("   (no hay datos de cámaras)")
        return []
    c = pd.read_csv(ruta)
    c["inst"] = pd.to_datetime(c["FECHA_INSTALACION"], errors="coerce")
    c["desm"] = pd.to_datetime(c["FECHA_DESMONTE"], errors="coerce")
    filas = []
    for _, r in c.dropna(subset=["inst", "lat", "lon"]).iterrows():
        trimestre = str(r["inst"].to_period("Q"))
        filas.append({
            "id_intervencion": f"camara_{int(r['OBJECTID'])}", "grupo": f"fotodeteccion_{trimestre}",
            "tipo_medida": "fotodeteccion", "direccion": "implementar",
            "valor_nuevo": r.get("VELOCIDAD_FISCALIZACION"), "unidad": "km/h fiscalizada",
            "corredor": "", "localidad": "", "fecha_inicio": str(r["inst"].date()),
            "fin_ventana": str((r["desm"] - pd.Timedelta(days=1)).date()) if pd.notna(r["desm"]) else "",
            "fuente": URL_CAM, "nota": f"{r.get('DIRECCION_FOTODETECCION')} · {r.get('TIPO_FOTODETECCION')} · {r.get('CORREDOR')}",
            "geometria_geojson": json.dumps({"type": "Point", "coordinates": [r["lon"], r["lat"]]}),
            "radio_m": 300, "revisado": True})
    print(f"   cámaras: {len(filas)} con fecha de instalación ({c['inst'].isna().sum()} sin fecha, se omiten)")
    return filas


def primera_fecha(texto: str) -> str | None:
    t = str(texto).lower()
    m = re.search(r"(\d{1,2})(?:\s*-\s*\d{1,2})?\s+(?:de\s+)?(" + "|".join(MESES) + r")(?:\s+de)?\s+(\d{4})", t)
    if m:
        return f"{int(m.group(3)):04d}-{MESES[m.group(2)]:02d}-{int(m.group(1)):02d}"
    m = re.search(r"(" + "|".join(MESES) + r")(?:\s+de)?\s+(\d{4})", t)
    return f"{int(m.group(2)):04d}-{MESES[m.group(1)]:02d}-01" if m else None


def resaltos() -> list[dict]:
    ruta = os.path.join(RAW, "resaltos_parabolicos", "0_Proyecto_ResaltosParabolicos.csv")
    if not os.path.exists(ruta):
        print("   (no hay datos de resaltos)")
        return []
    r = pd.read_csv(ruta)
    filas = []
    for _, f in r.iterrows():
        inicio = primera_fecha(f["FECHA_INTER_PR"])
        try:
            g = json.loads(f["geometria_json"])
            # el tramo viene como polígono (rings) o como línea (paths); se usa su contorno
            geo = {"type": "MultiLineString", "coordinates": g.get("paths") or g["rings"]}
        except (TypeError, ValueError, KeyError):
            geo = {"type": "Point", "coordinates": [f["lon"], f["lat"]]}
        filas.append({
            "id_intervencion": f"resaltos_{int(f['OBJECTID'])}", "grupo": f"resaltos_{inicio[:4] if inicio else 'sf'}",
            "tipo_medida": "reductor_velocidad", "direccion": "implementar", "valor_nuevo": f.get("RESALTOS_PR"),
            "unidad": "resaltos", "corredor": "", "localidad": f.get("LOCALIDAD"), "fecha_inicio": inicio or "",
            "fin_ventana": "", "fuente": URL_RES,
            "nota": (f"{f['NOMBRE_PR']} · {f['DIRECCION_PR']} · intervención: {f['FECHA_INTER_PR']} · "
                     f"impacto reportado por la SDM: fallecidos {f['IMPACTO_FALL_PR']} %, lesionados {f['IMPACTO_LES_PR']} %"),
            "geometria_geojson": json.dumps(geo), "radio_m": 100, "revisado": True})
    print(f"   resaltos: {len(filas)} proyectos")
    return filas


URL_CARRIL = "https://services2.arcgis.com/NEwhEo9GGSHXcRXV/arcgis/rest/services/Carril_Preferencial_SITP_Bogota_D_C/FeatureServer"
# Otras medidas en el mismo corredor dentro de la ventana de comparación: se corta la
# ventana antes de que empiecen, o se excluye el tramo si la otra medida empezó antes.
CARRIL_FIN = {"AV. Primero de Mayo": "2019-03-31"}          # 50 km/h desde abril de 2019
CARRIL_EXCLUIR = {"Av. Boyacá": "50 km/h desde octubre de 2018, dos meses antes del carril"}


def carriles() -> list[dict]:
    ruta = os.path.join(RAW, "carril_preferencial_sitp", "0_Carril_Preferencial_SITP_Bogota_D_C.csv")
    if not os.path.exists(ruta):
        print("   (no hay datos de carriles preferenciales)")
        return []
    c = pd.read_csv(ruta)
    filas = []
    for _, r in c.iterrows():
        if pd.isna(r.get("FECHA_IMP_CAR")):
            continue
        corr = str(r["CORREDOR"]).strip()
        excl = next((v for k, v in CARRIL_EXCLUIR.items() if k.lower() in corr.lower()), None)
        try:
            g = json.loads(r["geometria_json"])
            geo = {"type": "MultiLineString", "coordinates": g["paths"]}
        except (TypeError, ValueError, KeyError):
            continue
        filas.append({
            "id_intervencion": f"carril_{int(r['OBJECTID'])}", "grupo": f"carril_{str(r['FECHA_IMP_CAR'])[:4]}",
            "tipo_medida": "carril_preferencial", "direccion": "implementar", "unidad": "", "corredor": "",
            "localidad": "", "fecha_inicio": str(r["FECHA_IMP_CAR"])[:10],
            "fin_ventana": next((v for k, v in CARRIL_FIN.items() if k.lower() in corr.lower()), ""),
            "norma": r.get("RESOLUCION_CARRIL"), "fuente": URL_CARRIL,
            "nota": f"{corr} · {r.get('TRAMO_CARRIL')}" + (f" · EXCLUIDO: {excl}" if excl else ""),
            "geometria_geojson": json.dumps(geo), "radio_m": 50, "revisado": not excl})
    print(f"   carriles preferenciales: {len(filas)} ({sum(not f['revisado'] for f in filas)} excluidos por otra medida en el corredor)")
    return filas


def zona_escolar() -> list[dict]:
    """30 km/h en zonas escolares (Decreto 073 de 2021, desde el 19-mar-2021).

    Tratadas: celdas de 500 m con 3 o más sedes educativas. Controles posibles: celdas
    sin colegios (las de 1–2 colegios se excluyen, porque recibieron la medida en parte).
    Para no confundir el efecto con la pandemia y el regreso a clases, se comparan los
    24 meses antes de marzo de 2020 con los 24 meses desde marzo de 2022 (colegios
    abiertos en los dos periodos); 2020–2021 queda por fuera (meses_transicion = 24)."""
    ruta_col = os.path.join(RAW, "colegios", "0_Colegios.csv")
    ruta_cel = os.path.join(RAIZ, "out", "panel", "celdas.csv")
    if not (os.path.exists(ruta_col) and os.path.exists(ruta_cel)):
        print("   (faltan colegios o el panel para la zona escolar)")
        return []
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from panel_espacial import a_celda
    col = pd.read_csv(ruta_col, usecols=["lon", "lat"]).dropna()
    f, c = a_celda(col["lat"], col["lon"], 500)
    conteo = ("r" + f.astype(str) + "_c" + c.astype(str)).value_counts()
    celdas = set(pd.read_csv(ruta_cel)["celda"])
    tratadas = sorted(k for k, v in conteo.items() if v >= 3 and k in celdas)
    parciales = sorted(k for k, v in conteo.items() if 1 <= v < 3 and k in celdas)
    print(f"   zona escolar: {len(tratadas)} celdas con 3+ colegios (tratadas), {len(parciales)} con 1–2 (fuera de los controles)")
    return [{"id_intervencion": "zona_escolar_2021", "grupo": "zona_escolar_2021", "tipo_medida": "zona_escolar",
             "direccion": "reducir", "valor_anterior": 50, "valor_nuevo": 30, "unidad": "km/h", "corredor": "", "localidad": "",
             "fecha_inicio": "2020-03-01", "meses_transicion": 24, "fin_ventana": "2024-02-29",
             "norma": "Decreto 073 de 2021", "fuente": "https://www.alcaldiabogota.gov.co/sisjur/normas/Norma1.jsp?i=109086",
             "nota": "Medida desde 19-mar-2021; se compara mar-2018/feb-2020 contra mar-2022/feb-2024 para evitar la pandemia",
             "celdas_ids": ";".join(tratadas), "excluir_ids": ";".join(parciales), "revisado": True}]


def main():
    filas = camaras() + resaltos() + carriles() + zona_escolar()
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    pd.DataFrame(filas, columns=COLS).to_csv(SALIDA, index=False, encoding="utf-8-sig")
    print(f"✅ {len(filas)} intervenciones en {SALIDA}")


if __name__ == "__main__":
    main()
