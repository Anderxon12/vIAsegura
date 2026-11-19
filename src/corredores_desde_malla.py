"""
=============================================================
  CORREDORES DESDE LA MALLA VIAL
  Construye data/intervenciones/corredores.geojson
=============================================================

Para qué
    Para medir el efecto de "50 km/h en la Av. Boyacá" hay que saber qué
    celdas del panel toca la Av. Boyacá. Este script toma la Malla Vial
    Integral descargada por ingesta_sdm.py (data/raw/malla_vial/) y junta
    los segmentos de cada corredor en una sola geometría.

Cómo encuentra cada corredor
    Los nombres de vía se buscan con los mismos patrones del catálogo de
    resoluciones (catalogo_resoluciones.CORREDORES): nombre común
    ("Avenida Boyacá") y nomenclatura IDECA ("AK 72"). La columna de nombre
    se elige sola (la que más coincide) o se fuerza con --columna. Si hay
    una columna de jerarquía vial, se prefieren los segmentos arteriales.

Revise el reporte: número de segmentos y longitud por corredor. Un
corredor de 300 km o de 0,5 km indica que el nombre se cruzó con otra vía.
Las líneas dibujadas a mano (propiedad "manual": true) se conservan.

Uso
    python src/corredores_desde_malla.py
    python src/corredores_desde_malla.py --columna MVINOMBRE
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catalogo_resoluciones import CORREDORES, plano  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_MALLA = os.path.join(RAIZ, "data", "raw", "malla_vial")
SALIDA = os.path.join(RAIZ, "data", "intervenciones", "corredores.geojson")
M_LAT, M_LON = 111_320.0, 111_320.0 * np.cos(np.radians(4.65))


# Nombre oficial de cada corredor en la Malla Vial Integral (columnas MVINOMBRE / MVINALTERN).
# Verificado contra la malla descargada en noviembre de 2026. Es más exacto que la
# nomenclatura (AK 72, AC 80…), que cambia a lo largo de la vía y se cruza con otras calles.
NOMBRES_OFICIALES = {
    "Av. Boyacá": ["AVENIDA BOYACA"],
    "Av. Carrera 68": ["AVENIDA DEL CONGRESO EUCARISTICO"],
    "Av. Ciudad de Cali": ["AVENIDA CIUDAD DE CALI"],
    "Calle 80": ["AVENIDA MEDELLIN"],
    "Av. Las Américas": ["AVENIDA DE LAS AMERICAS"],
    "NQS (Av. Carrera 30)": ["AVENIDA CIUDAD DE QUITO"],
    "Av. Primero de Mayo": ["AVENIDA PRIMERO DE MAYO"],
    "Av. Suba": ["AVENIDA ALFREDO D BATEMAN", "AVENIDA SUBA"],
    "Av. Villavicencio": ["AVENIDA CIUDAD DE VILLAVICENCIO"],
    "Calle 13": ["AVENIDA CENTENARIO", "AVENIDA COLON"],
    "Av. Caracas": ["AVENIDA CARACAS"],
    "Autopista Norte": ["AVENIDA PASEO DE LOS LIBERTADORES", "AUTOPISTA NORTE"],
    "Autopista Sur": ["AVENIDA DEL SUR"],
    "Calle 26 (Av. El Dorado)": ["AVENIDA JORGE ELIECER GAITAN", "AVENIDA EL DORADO"],
    "Carrera Séptima": ["AVENIDA ALBERTO LLERAS CAMARGO"],
    "Av. Circunvalar": ["AVENIDA DE LOS CERROS"],
    "Av. Esperanza": ["AVENIDA DE LA ESPERANZA"],
    "Calle 170": ["AVENIDA SAN JOSE"],
    "Calle 127": ["AVENIDA CALLEJAS"],
    "Calle 100": ["AVENIDA CARLOS LLERAS RESTREPO", "AVENIDA ESPANA"],
    "Calle 63": ["AVENIDA JOSE CELESTINO MUTIS"],
    "Av. Bosa": ["AVENIDA BOSA"],
    "Av. Tintal": ["AVENIDA TINTAL"],
}


def cargar_malla() -> pd.DataFrame:
    archivos = [f for f in glob.glob(os.path.join(DIR_MALLA, "*.csv"))]
    if not archivos:
        sys.exit(f"No hay archivos en {DIR_MALLA}. Corra primero: python src/ingesta_sdm.py --solo malla_vial")
    partes = []
    for f in archivos:
        d = pd.read_csv(f, low_memory=False)
        if "geometria_json" in d.columns:
            d["_capa"] = os.path.basename(f)
            partes.append(d)
    if not partes:
        sys.exit("La malla vial descargada no trae geometrías de línea (columna geometria_json).")
    return pd.concat(partes, ignore_index=True)


def columnas_texto(d: pd.DataFrame) -> list[str]:
    return [c for c in d.columns if (d[c].dtype == object or pd.api.types.is_string_dtype(d[c]))
            and c not in ("geometria_json", "_capa")]


def patron_corredor(nombre: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in CORREDORES.get(nombre, [r"(?!x)x"])))


def elegir_columna(d: pd.DataFrame) -> tuple[str, dict]:
    puntajes = {}
    todos = re.compile("|".join(f"(?:{p})" for pats in CORREDORES.values() for p in pats))
    for c in columnas_texto(d):
        v = d[c].dropna().astype(str).map(plano)
        puntajes[c] = int(v.str.contains(todos).sum())
    if not puntajes or max(puntajes.values()) == 0:
        sys.exit("Ninguna columna de texto contiene nombres de corredores. Use --columna.")
    return max(puntajes, key=puntajes.get), puntajes


def columna_jerarquia(d: pd.DataFrame) -> str | None:
    for c in columnas_texto(d):
        v = d[c].dropna().astype(str).map(plano)
        if v.str.contains(r"arterial").mean() > 0.01:
            return c
    return None


def longitud_km(paths) -> float:
    total = 0.0
    for p in paths:
        a = np.array(p, dtype=float)
        if len(a) > 1:
            total += np.hypot(np.diff(a[:, 0]) * M_LON, np.diff(a[:, 1]) * M_LAT).sum()
    return total / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--columna", help="columna con el nombre de la vía (se elige sola si no se da)")
    ap.add_argument("--sin-filtro-arterial", action="store_true", help="no preferir segmentos arteriales")
    args = ap.parse_args()

    d = cargar_malla()
    print(f"Malla vial: {len(d):,} segmentos · columnas de texto: {columnas_texto(d)}")
    if args.columna:
        col, punt = args.columna, {}
    else:
        col, punt = elegir_columna(d)
        print(f"Coincidencias por columna: {punt}")
    print(f"Columna de nombre usada: {col}")
    jer = None if args.sin_filtro_arterial else columna_jerarquia(d)
    if jer:
        print(f"Columna de jerarquía: {jer} (se prefieren segmentos arteriales)")

    nombres = d[col].fillna("").astype(str).map(plano)
    es_arterial = d[jer].fillna("").astype(str).map(plano).str.contains("arterial") if jer else pd.Series(True, index=d.index)

    previos = []
    if os.path.exists(SALIDA):
        previos = [f for f in json.load(open(SALIDA, encoding="utf-8"))["features"] if f["properties"].get("manual")]
    manuales = {f["properties"]["corredor"] for f in previos}

    feats, reporte = list(previos), []
    for corredor in dict.fromkeys(list(NOMBRES_OFICIALES) + list(CORREDORES)):
        if corredor in manuales:
            reporte.append({"corredor": corredor, "segmentos": "manual", "longitud_km": None})
            continue
        oficiales = {plano(n) for n in NOMBRES_OFICIALES.get(corredor, [])}
        if oficiales and {"MVINOMBRE", "MVINALTERN"} & set(d.columns):
            m = pd.Series(False, index=d.index)
            for c in ("MVINOMBRE", "MVINALTERN"):
                if c in d.columns:
                    m |= d[c].fillna("").astype(str).str.strip().map(plano).isin(oficiales)
            metodo = "nombre oficial"
        else:
            m = nombres.str.contains(patron_corredor(corredor))
            if jer and (m & es_arterial).any():
                m = m & es_arterial
            metodo = f"patrones en {col}"
        sel = d[m]
        paths = []
        for g in sel["geometria_json"].dropna():
            try:
                geo = json.loads(g)
            except (TypeError, ValueError):
                continue
            paths += geo.get("paths") or []
        if not paths:
            reporte.append({"corredor": corredor, "segmentos": 0, "longitud_km": 0})
            continue
        km = longitud_km(paths)
        ejemplos = sel[col].astype(str).value_counts().head(4).index.tolist()
        if "MVI_VELREG" in sel.columns:
            vel = sel["MVI_VELREG"].value_counts(normalize=True).round(2).to_dict()
        else:
            vel = {}
        feats.append({"type": "Feature",
                      "properties": {"corredor": corredor, "segmentos": int(len(sel)), "longitud_km": round(km, 1),
                                     "metodo": metodo, "nombres_encontrados": ejemplos,
                                     "velocidad_reglamentaria": {str(k): v for k, v in vel.items()}, "manual": False},
                      "geometry": {"type": "MultiLineString", "coordinates": paths}})
        reporte.append({"corredor": corredor, "segmentos": int(len(sel)), "longitud_km": round(km, 1),
                        "metodo": metodo, "nomenclatura": "; ".join(ejemplos)[:60]})

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    rep = pd.DataFrame(reporte)
    print("\n" + rep.to_string(index=False))
    sospechosos = rep[pd.to_numeric(rep["longitud_km"], errors="coerce").fillna(0).pipe(lambda s: (s > 0) & ((s < 2) | (s > 150)))]
    if len(sospechosos):
        print(f"\n⚠ Revise estos corredores (longitud rara): {sospechosos['corredor'].tolist()}")
    faltan = rep[rep["segmentos"] == 0]["corredor"].tolist()
    if faltan:
        print(f"⚠ Sin segmentos: {faltan}. Ajuste sus patrones en catalogo_resoluciones.CORREDORES "
              f"o dibújelos en geojson.io y agréguelos con la propiedad \"manual\": true.")
    print(f"\n✅ {len(feats)} corredores en {SALIDA}")


if __name__ == "__main__":
    main()
