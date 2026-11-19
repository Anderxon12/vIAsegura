"""
=============================================================
  PANEL ESPACIAL — Siniestros por celda de 500 m y por mes
  Bogotá D.C. · Base: SDM (Base total.csv)
=============================================================

Por qué
    El chatbot no pregunta "¿qué tan grave será este accidente?", sino
    "¿cuántos siniestros habrá en esta zona si aplico la medida?".
    Para eso la unidad de análisis es un lugar en un mes, no un accidente.

Qué hace
    1. Lee data/Base total.csv (una fila por persona) y la lleva a una fila
       por siniestro, con heridos, muertos, actores y causas.
    2. Divide Bogotá en una grilla de celdas de TAM_CELDA metros.
    3. Cuenta, para cada celda y cada mes, los siniestros y sus víctimas
       (incluye los meses con cero siniestros).

Salidas (out/panel/)
    siniestros.csv     una fila por siniestro, con su celda
    panel.csv          celda × mes, listo para modelar
    celdas.geojson     polígonos de las celdas (para mapas y para cruzar
                       con corredores, semáforos, cámaras, etc.)
    reporte_panel.json cifras de control

Objetivo recomendado: n_sin_victimas (siniestros con heridos o muertos).
    Desde 2023 casi no se registran los siniestros de "solo daños", así que
    el total de siniestros NO es comparable entre años; los siniestros con
    víctimas sí (≈ 11.000–13.000 por año en toda la serie).

Uso
    python src/panel_espacial.py
    python src/panel_espacial.py --tam 300      # celdas de 300 m
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADA = os.path.join(RAIZ, "data", "Base total.csv")
SALIDA = os.path.join(RAIZ, "out", "panel")

# Origen de la grilla y caja de Bogotá urbana (con margen)
LAT0, LON0 = 4.45, -74.25
LAT_MIN, LAT_MAX = 4.45, 4.85
LON_MIN, LON_MAX = -74.25, -73.98
M_POR_GRADO_LAT = 111_320.0
M_POR_GRADO_LON = 111_320.0 * np.cos(np.radians(4.65))

COLS = ["Codigo_Accidente", "FechaAcc", "hourAcc", "Localidad", "Latitud", "Longitud",
        "Gravedad_Indicador_Tradicional", "Condicion", "Clase_Acc",
        "Con_Velocidad", "Con_Embriaguez", "Con_Moto", "Con_Peaton", "Con_Bicicleta",
        "Con_Huecos", "Con_Menores", "Con_Persona_Mayor", "Con_Carga"]


# ════════════════════════════════════════════════════════════
#  GRILLA
# ════════════════════════════════════════════════════════════
def a_celda(lat: pd.Series, lon: pd.Series, tam: float) -> tuple[pd.Series, pd.Series]:
    fila = np.floor((lat - LAT0) * M_POR_GRADO_LAT / tam).astype("Int64")
    col = np.floor((lon - LON0) * M_POR_GRADO_LON / tam).astype("Int64")
    return fila, col


def centro_celda(fila, col, tam):
    lat = LAT0 + (np.asarray(fila) + 0.5) * tam / M_POR_GRADO_LAT
    lon = LON0 + (np.asarray(col) + 0.5) * tam / M_POR_GRADO_LON
    return lat, lon


def poligono_celda(fila, col, tam):
    la0 = LAT0 + fila * tam / M_POR_GRADO_LAT
    la1 = LAT0 + (fila + 1) * tam / M_POR_GRADO_LAT
    lo0 = LON0 + col * tam / M_POR_GRADO_LON
    lo1 = LON0 + (col + 1) * tam / M_POR_GRADO_LON
    return [[[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1], [lo0, la0]]]


# ════════════════════════════════════════════════════════════
#  CARGA Y NIVEL SINIESTRO
# ════════════════════════════════════════════════════════════
def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def cargar(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta, sep=None, engine="python", encoding="latin-1",
                     usecols=lambda c: c in COLS)
    df["Latitud"], df["Longitud"] = num(df["Latitud"]), num(df["Longitud"])
    df["fecha"] = pd.to_datetime(df["FechaAcc"], format="%d/%m/%Y", errors="coerce")
    return df


def nivel_siniestro(df: pd.DataFrame) -> pd.DataFrame:
    grav = df["Gravedad_Indicador_Tradicional"].astype(str).str.upper()
    cond = df["Condicion"].astype(str).str.upper()
    df = df.assign(herido=(grav == "HERIDO").astype(int), muerto=(grav == "MUERTO").astype(int),
                   es_peaton=cond.str.startswith("PEAT").astype(int),
                   es_moto=(cond == "MOTOCICLISTA").astype(int),
                   es_ciclista=(cond == "CICLISTA").astype(int))
    banderas = [c for c in df.columns if c.startswith("Con_")]
    for c in banderas:
        df[c] = (df[c].astype(str).str.upper() == "SI").astype(int)
    g = df.groupby("Codigo_Accidente", sort=False)
    acc = g[["fecha", "hourAcc", "Localidad", "Latitud", "Longitud", "Clase_Acc"]].first()
    acc = acc.join(g[["herido", "muerto"]].sum().rename(columns={"herido": "n_heridos", "muerto": "n_muertos"}))
    acc = acc.join(g[["es_peaton", "es_moto", "es_ciclista"] + banderas].max())
    acc["con_victimas"] = ((acc["n_heridos"] + acc["n_muertos"]) > 0).astype(int)
    acc["fatal"] = (acc["n_muertos"] > 0).astype(int)
    return acc.reset_index()


# ════════════════════════════════════════════════════════════
#  PANEL
# ════════════════════════════════════════════════════════════
def construir_panel(acc: pd.DataFrame, tam: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rep = {"siniestros": int(len(acc))}
    valida = (acc["Latitud"].between(LAT_MIN, LAT_MAX) & acc["Longitud"].between(LON_MIN, LON_MAX)
              & acc["fecha"].notna())
    rep["sin_coordenada_o_fecha_valida"] = int((~valida).sum())
    acc = acc[valida].copy()
    acc["fila"], acc["col"] = a_celda(acc["Latitud"], acc["Longitud"], tam)
    acc["celda"] = "r" + acc["fila"].astype(str) + "_c" + acc["col"].astype(str)
    acc["mes"] = acc["fecha"].dt.to_period("M").dt.to_timestamp()

    v = acc["con_victimas"] == 1
    agg = acc.groupby(["celda", "mes"]).agg(
        n_siniestros=("Codigo_Accidente", "size"),
        n_sin_victimas=("con_victimas", "sum"),
        n_sin_fatales=("fatal", "sum"),
        n_heridos=("n_heridos", "sum"),
        n_muertos=("n_muertos", "sum"))
    agg = agg.join(acc[v].groupby(["celda", "mes"]).agg(
        n_vict_peaton=("es_peaton", "sum"), n_vict_moto=("es_moto", "sum"),
        n_vict_ciclista=("es_ciclista", "sum"), n_vict_velocidad=("Con_Velocidad", "sum"),
        n_vict_embriaguez=("Con_Embriaguez", "sum")))

    # celdas: todas las que tuvieron al menos un siniestro en el periodo
    celdas = acc.groupby("celda").agg(fila=("fila", "first"), col=("col", "first"),
                                      localidad=("Localidad", lambda s: s.mode().iloc[0] if s.notna().any() else None),
                                      siniestros_total=("Codigo_Accidente", "size"),
                                      sin_victimas_total=("con_victimas", "sum")).reset_index()
    celdas["lat"], celdas["lon"] = centro_celda(celdas["fila"], celdas["col"], tam)

    meses = pd.date_range(acc["mes"].min(), acc["mes"].max(), freq="MS")
    idx = pd.MultiIndex.from_product([celdas["celda"], meses], names=["celda", "mes"])
    panel = agg.reindex(idx).fillna(0).astype(int).reset_index()
    panel = panel.merge(celdas[["celda", "localidad", "lat", "lon"]], on="celda", how="left")
    panel["anio"] = panel["mes"].dt.year
    panel["mes_num"] = panel["mes"].dt.month

    # el último mes puede venir incompleto (la base termina a mitad de mes)
    ultimo = acc["fecha"].max()
    if ultimo < (ultimo + pd.offsets.MonthEnd(0)) - pd.Timedelta(days=2):
        panel = panel[panel["mes"] < ultimo.to_period("M").to_timestamp()]
        rep["mes_incompleto_excluido"] = str(ultimo.to_period("M"))

    rep.update(tam_celda_m=tam, celdas=int(len(celdas)), meses=int(panel["mes"].nunique()),
               filas_panel=int(len(panel)), rango=[str(panel["mes"].min().date()), str(panel["mes"].max().date())],
               sin_victimas_por_anio=panel.groupby("anio")["n_sin_victimas"].sum().to_dict(),
               pct_celda_mes_con_cero=round(100 * (panel["n_sin_victimas"] == 0).mean(), 1),
               celdas_con_80pct_victimas=int((celdas.sort_values("sin_victimas_total", ascending=False)
                                              ["sin_victimas_total"].cumsum() <= 0.8 * celdas["sin_victimas_total"].sum()).sum()))
    return acc, panel, celdas, rep


def guardar_geojson(celdas: pd.DataFrame, tam: float, ruta: str) -> None:
    feats = [{"type": "Feature",
              "properties": {"celda": r.celda, "localidad": r.localidad,
                             "siniestros_total": int(r.siniestros_total),
                             "sin_victimas_total": int(r.sin_victimas_total)},
              "geometry": {"type": "Polygon", "coordinates": poligono_celda(int(r.fila), int(r.col), tam)}}
             for r in celdas.itertuples()]
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tam", type=float, default=500, help="tamaño de celda en metros (500 por defecto)")
    ap.add_argument("--entrada", default=ENTRADA)
    args = ap.parse_args()
    os.makedirs(SALIDA, exist_ok=True)

    acc = nivel_siniestro(cargar(args.entrada))
    acc, panel, celdas, rep = construir_panel(acc, args.tam)

    acc.drop(columns=["fila", "col"]).to_csv(os.path.join(SALIDA, "siniestros.csv"), index=False, encoding="utf-8-sig")
    panel.to_csv(os.path.join(SALIDA, "panel.csv"), index=False, encoding="utf-8-sig")
    celdas.to_csv(os.path.join(SALIDA, "celdas.csv"), index=False, encoding="utf-8-sig")
    guardar_geojson(celdas, args.tam, os.path.join(SALIDA, "celdas.geojson"))
    with open(os.path.join(SALIDA, "reporte_panel.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    print(f"\n✅ Panel en {SALIDA}/")


if __name__ == "__main__":
    main()
