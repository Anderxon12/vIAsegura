"""
=============================================================
  PANEL SDM — Siniestros 2015 → hoy desde la base de Movilidad
  (Siniestralidad_BD, descargada por ingesta_sdm.py)
=============================================================

Reemplaza a panel_espacial.py como fuente principal: tiene datos hasta el
mes pasado, la vía de cada siniestro (nombre oficial de la malla vial) y
las causas por vehículo.

LIMPIEZA IMPORTANTE (duplicados)
    Entre 2013 y 2020 la base trae cada siniestro dos veces: un formulario
    "A000041337" y una copia "AA000041337" con la misma fecha, hora,
    dirección y gravedad. Sin quitarlas, las muertes de 2015–2019 salen en
    ~1.000/año en vez de ~550 (la cifra oficial). Se elimina toda fila "AA"
    cuyo gemelo "A" existe, en todas las tablas.

Salidas (out/panel/)
    siniestros.csv      una fila por siniestro (depurado), con causas
    panel.csv           celda de 500 m × mes  (mismo formato de panel_espacial)
    panel_vias.csv      avenida × mes (siniestros SOBRE la vía, por nombre oficial)
    celdas.geojson, celdas.csv, reporte_panel.json

Uso
    python src/panel_sdm.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_espacial import (LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, a_celda,  # noqa: E402
                            centro_celda, guardar_geojson)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_BD = os.path.join(RAIZ, "data", "raw", "siniestralidad_bd")
SALIDA = os.path.join(RAIZ, "out", "panel")
TAM = 500
DESDE = "2010-01-01"     # 2010–2014 sirve de "antes" para medidas de 2014–2016 (la serie es consistente desde 2010)

# códigos de causa (tabla VM_ACC_CAUSA)
CAUSAS = {
    "velocidad": ["116"],
    "embriaguez": ["114", "115", "410"],
    "semaforo_rojo": ["142"],
    "desobedecer_senales": ["112"],
    "distancia": ["121"],
    "huecos": ["306"],
    "peaton_cruza": ["409"],
}


def leer(nombre, columnas):
    ruta = os.path.join(DIR_BD, nombre)
    if not os.path.exists(ruta):
        sys.exit(f"Falta {ruta}. Corra: python src/ingesta_sdm.py --solo siniestralidad_bd")
    return pd.read_csv(ruta, low_memory=False, usecols=lambda c: c in columnas)


def formularios_duplicados(formularios: pd.Series) -> set:
    """Formularios 'AA…' cuya versión 'A…' también existe."""
    f = formularios.astype(str)
    aa = f[f.str.startswith("AA")]
    gemelos = aa.str.replace(r"^AA", "A", regex=True)
    existentes = set(f[~f.str.startswith("AA")])
    return set(aa[gemelos.isin(existentes)])


def construir():
    rep = {"fuente": "Siniestralidad_BD (SDM)"}
    acc = leer("2_ACCIDENTE.csv", ["FORMULARIO", "FECHA_OCURRENCIA_ACC", "HORA_OCURRENCIA_ACC", "GRAVEDAD",
                                   "CLASE_ACC", "LOCALIDAD", "MVINOMBRE", "CIV", "DISTANCIA_VIA", "lat", "lon"])
    dup = formularios_duplicados(acc["FORMULARIO"])
    rep["formularios_duplicados_AA_eliminados"] = len(dup)
    acc = acc[~acc["FORMULARIO"].isin(dup)].copy()
    acc["fecha"] = pd.to_datetime(acc["FECHA_OCURRENCIA_ACC"], errors="coerce")
    acc = acc[acc["fecha"] >= DESDE]

    # víctimas
    les = leer("1_LESIONADO.csv", ["FORMULARIO", "CONDICION"])
    mue = leer("0_MUERTO.csv", ["FORMULARIO", "CONDICION"])
    les, mue = les[~les["FORMULARIO"].isin(dup)], mue[~mue["FORMULARIO"].isin(dup)]
    vict = pd.concat([les.assign(herido=1, muerto=0), mue.assign(herido=0, muerto=1)])
    cond = vict["CONDICION"].astype(str).str.upper()
    vict = vict.assign(v_peaton=cond.str.startswith("PEAT").astype(int),
                       v_moto=cond.str.contains("MOTO").astype(int),
                       v_ciclista=cond.str.contains("CICL").astype(int))
    v = vict.groupby("FORMULARIO")[["herido", "muerto", "v_peaton", "v_moto", "v_ciclista"]].sum()
    v = v.rename(columns={"herido": "n_heridos", "muerto": "n_muertos"})
    acc = acc.merge(v, left_on="FORMULARIO", right_index=True, how="left")
    for c in v.columns:
        acc[c] = acc[c].fillna(0).astype(int)

    # causas (cualquier vehículo del siniestro)
    cau = leer("4_VM_ACC_CAUSA.csv", ["FORMULARIO", "CODIGO_CAUSA"])
    cau = cau[~cau["FORMULARIO"].isin(dup)]
    cau["CODIGO_CAUSA"] = cau["CODIGO_CAUSA"].astype(str).str.strip()
    for nombre, codigos in CAUSAS.items():
        con = set(cau.loc[cau["CODIGO_CAUSA"].isin(codigos), "FORMULARIO"])
        acc[f"c_{nombre}"] = acc["FORMULARIO"].isin(con).astype(int)

    g = acc["GRAVEDAD"].astype(str).str.upper()
    acc["con_victimas"] = g.isin(["CON HERIDOS", "CON MUERTOS"]).astype(int)
    acc["fatal"] = (g == "CON MUERTOS").astype(int)
    acc["via"] = acc["MVINOMBRE"].astype(str).str.strip().replace({"SIN_NMG": "", "nan": ""})
    acc["mes"] = acc["fecha"].dt.to_period("M").dt.to_timestamp()

    # último mes completo
    ultimo = acc["fecha"].max()
    corte = ultimo.to_period("M").to_timestamp()
    if ultimo < corte + pd.offsets.MonthEnd(0) - pd.Timedelta(days=1):
        acc = acc[acc["mes"] < corte]
        rep["mes_incompleto_excluido"] = str(corte.date())[:7]
    rep["siniestros"] = int(len(acc))
    rep["sin_victimas_por_anio"] = acc.groupby(acc["fecha"].dt.year)["con_victimas"].sum().to_dict()
    rep["muertos_por_anio"] = acc.groupby(acc["fecha"].dt.year)["n_muertos"].sum().to_dict()
    rep["pct_con_via"] = round(100 * (acc["via"] != "").mean(), 1)
    return acc, rep


def agregar(acc, claves):
    v = acc["con_victimas"] == 1
    base = acc.groupby(claves).agg(
        n_siniestros=("FORMULARIO", "size"), n_sin_victimas=("con_victimas", "sum"),
        n_sin_fatales=("fatal", "sum"), n_heridos=("n_heridos", "sum"), n_muertos=("n_muertos", "sum"))
    extra = acc[v].groupby(claves).agg(
        n_vict_peaton=("v_peaton", lambda s: int((s > 0).sum())), n_vict_moto=("v_moto", lambda s: int((s > 0).sum())),
        n_vict_ciclista=("v_ciclista", lambda s: int((s > 0).sum())),
        n_vict_velocidad=("c_velocidad", "sum"), n_vict_embriaguez=("c_embriaguez", "sum"),
        n_vict_semaforo=("c_semaforo_rojo", "sum"), n_vict_senales=("c_desobedecer_senales", "sum"))
    return base.join(extra)


def completar(agg, unidades, meses, nombre_unidad):
    idx = pd.MultiIndex.from_product([unidades, meses], names=[nombre_unidad, "mes"])
    return agg.reindex(idx).fillna(0).astype(int).reset_index()


def main():
    os.makedirs(SALIDA, exist_ok=True)
    acc, rep = construir()

    # ── celda × mes ─────────────────────────────────────────
    ok = acc["lat"].between(LAT_MIN, LAT_MAX) & acc["lon"].between(LON_MIN, LON_MAX)
    rep["sin_coordenada_valida"] = int((~ok).sum())
    a = acc[ok].copy()
    a["fila"], a["col"] = a_celda(a["lat"], a["lon"], TAM)
    a["celda"] = "r" + a["fila"].astype(str) + "_c" + a["col"].astype(str)
    meses = pd.date_range(acc["mes"].min(), acc["mes"].max(), freq="MS")
    celdas = a.groupby("celda").agg(fila=("fila", "first"), col=("col", "first"),
                                    localidad=("LOCALIDAD", lambda s: s.mode().iloc[0] if s.notna().any() else None),
                                    siniestros_total=("FORMULARIO", "size"),
                                    sin_victimas_total=("con_victimas", "sum")).reset_index()
    celdas["lat"], celdas["lon"] = centro_celda(celdas["fila"], celdas["col"], TAM)
    panel = completar(agregar(a, ["celda", "mes"]), celdas["celda"], meses, "celda")
    panel = panel.merge(celdas[["celda", "localidad", "lat", "lon"]], on="celda", how="left")
    panel["anio"], panel["mes_num"] = panel["mes"].dt.year, panel["mes"].dt.month

    # ── avenida × mes ───────────────────────────────────────
    av = acc[acc["via"].str.startswith(("AVENIDA", "AUTOPISTA"))]
    vias = av.groupby("via").agg(n=("FORMULARIO", "size"), lat=("lat", "median"), lon=("lon", "median"),
                                 localidad=("LOCALIDAD", lambda s: s.mode().iloc[0] if s.notna().any() else None))
    vias = vias[vias["n"] >= 200]                       # avenidas con historia suficiente
    pv = completar(agregar(av[av["via"].isin(vias.index)], ["via", "mes"]), vias.index, meses, "via")
    pv = pv.merge(vias[["lat", "lon", "localidad"]], left_on="via", right_index=True, how="left")
    pv["anio"], pv["mes_num"] = pv["mes"].dt.year, pv["mes"].dt.month

    rep.update(celdas=int(len(celdas)), meses=len(meses), rango=[str(meses.min().date()), str(meses.max().date())],
               avenidas=int(len(vias)), filas_panel=int(len(panel)), filas_panel_vias=int(len(pv)))

    cols_sin = ["FORMULARIO", "fecha", "HORA_OCURRENCIA_ACC", "GRAVEDAD", "CLASE_ACC", "LOCALIDAD", "via", "CIV",
                "DISTANCIA_VIA", "lat", "lon", "n_heridos", "n_muertos", "v_peaton", "v_moto", "v_ciclista"] + \
               [f"c_{c}" for c in CAUSAS]
    acc[cols_sin].to_csv(os.path.join(SALIDA, "siniestros.csv"), index=False, encoding="utf-8-sig")
    panel.to_csv(os.path.join(SALIDA, "panel.csv"), index=False, encoding="utf-8-sig")
    pv.to_csv(os.path.join(SALIDA, "panel_vias.csv"), index=False, encoding="utf-8-sig")
    celdas.to_csv(os.path.join(SALIDA, "celdas.csv"), index=False, encoding="utf-8-sig")
    guardar_geojson(celdas, TAM, os.path.join(SALIDA, "celdas.geojson"))
    with open(os.path.join(SALIDA, "reporte_panel.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    print(f"\n✅ Panel SDM en {SALIDA}/")


if __name__ == "__main__":
    main()
