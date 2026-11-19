"""
=============================================================
  VÍASEGURA — Tramos viales para el modelo de riesgo 2026
=============================================================

El paquete del modelo (Grupo3_Modelo_Final.pkl) trae solo los 2.000 tramos de mayor
riesgo. Para que en la pestaña «Modelos ML → Tramos 2026» se pueda dibujar CUALQUIER vía
y predecir sus tramos, este script arma las variables del modelo para los ~113 mil
segmentos de la malla vial (CIV), con el mismo procedimiento del notebook
vIAsegura_ML/entrega/Grupo3_ModelamientoEvaluacion.ipynb (sección del modelo de tramos):

    - malla vial y dispositivos con las funciones del propio proyecto de ML
      (src.modelado.riesgo_tramos.cargar_malla / dispositivos), sin modificarlas;
    - siniestros con víctimas por CIV y año (siniestros_enriquecidos + ubicación geo);
    - historia 2023-2025, dispositivos vigentes en 2026 y límite vigente.

Guarda en data/ml/:
    tramos_2026.parquet   una fila por CIV: variables del modelo, nombre de la vía,
                          localidad y geometría simplificada (GeoJSON en texto)
    tramos_puntos.npz     11 puntos a lo largo de cada tramo, para saber qué tramos
                          sigue una línea dibujada

Al final comprueba que la predicción coincide con la que trae el .pkl para sus 2.000 tramos.

Uso (una vez, o cuando se actualicen las fases 2 y 4 del ML; tarda 1-3 minutos):
    python src/preparar_tramos_ml.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # vIAsegura_IA
RAIZ_ML = os.environ.get("VIASEGURA_ML") or os.path.join(os.path.dirname(RAIZ), "vIAsegura_ML")
SALIDA = os.path.join(RAIZ, "data", "ml")
ANIO = 2026
N_PUNTOS = 11
VARS_TRAMO = ["largo_m", "lon", "lat", "es_arterial", "vel_reglamentaria", "vel_limite", "carriles", "ancho_calzada",
              "superficie", "n_piezas", "sem_n", "semaforo", "camara", "resalto",
              "y_lag1", "y_lag2", "y_lag3", "y_prom3", "y_fatal_prev3"]


def _simplificar(coords: np.ndarray, tol: float = 0.00002) -> list:
    """Douglas-Peucker (tol ≈ 2 m) y 6 decimales."""
    if len(coords) <= 2:
        return np.round(coords, 6).tolist()
    guardar = np.zeros(len(coords), bool)
    guardar[[0, -1]] = True
    pila = [(0, len(coords) - 1)]
    while pila:
        i, j = pila.pop()
        if j <= i + 1:
            continue
        p, d = coords[i], coords[j] - coords[i]
        n = np.hypot(*d) or 1e-12
        dist = np.abs(d[0] * (coords[i + 1:j, 1] - p[1]) - d[1] * (coords[i + 1:j, 0] - p[0])) / n
        k = int(np.argmax(dist))
        if dist[k] > tol:
            guardar[i + 1 + k] = True
            pila += [(i, i + 1 + k), (i + 1 + k, j)]
    return np.round(coords[guardar], 6).tolist()


def _puntos_a_lo_largo(partes: list[np.ndarray], n: int) -> np.ndarray:
    """n puntos repartidos por igual a lo largo de las piezas del tramo."""
    segs = [(a, b) for p in partes for a, b in zip(p[:-1], p[1:])]
    if not segs:
        c = np.vstack(partes)
        return np.repeat(c[:1], n, axis=0)
    A = np.array([s[0] for s in segs]); B = np.array([s[1] for s in segs])
    L = np.hypot((B[:, 0] - A[:, 0]) * 107_000, (B[:, 1] - A[:, 1]) * 110_600)      # metros aprox. en Bogotá
    acum = np.concatenate([[0], np.cumsum(L)])
    total = acum[-1] or 1e-9
    objetivo = np.linspace(0, total, n)
    i = np.clip(np.searchsorted(acum, objetivo, side="right") - 1, 0, len(segs) - 1)
    t = np.where(L[i] > 0, (objetivo - acum[i]) / np.where(L[i] > 0, L[i], 1), 0)
    return A[i] + (B[i] - A[i]) * t[:, None]


def geometria_malla() -> tuple[dict, dict]:
    """{civ: GeoJSON simplificado} y {civ: puntos a lo largo}, desde la malla vial del ML."""
    from src.enriquecimiento import espacial as E
    piezas: dict[int, list[np.ndarray]] = {}
    for f in E._leer("malla_vial_integral"):
        civ = pd.to_numeric((f.get("properties") or {}).get("MVICIV"), errors="coerce")
        if pd.isna(civ) or civ <= 0:
            continue
        g = f.get("geometry") or {}
        partes = g.get("coordinates") or []
        if g.get("type") == "LineString":
            partes = [partes]
        for p in partes:
            if len(p) >= 2:
                piezas.setdefault(int(civ), []).append(np.asarray([c[:2] for c in p], float))
    geom, puntos = {}, {}
    for civ, ps in piezas.items():
        simples = [_simplificar(p) for p in ps]
        geom[civ] = json.dumps({"type": "MultiLineString", "coordinates": simples} if len(simples) > 1
                               else {"type": "LineString", "coordinates": simples[0]}, separators=(",", ":"))
        puntos[civ] = _puntos_a_lo_largo(ps, N_PUNTOS)
    return geom, puntos


def construir() -> None:
    t0 = time.time()
    if not os.path.isdir(os.path.join(RAIZ_ML, "src")):
        sys.exit(f"No encuentro el proyecto de ML en {RAIZ_ML}")
    sys.path.insert(0, RAIZ_ML)
    from src.modelado import riesgo_tramos as RT            # funciones del proyecto de ML
    inter = os.path.join(RAIZ_ML, "output", "intermedios")

    print("Leyendo la malla vial y los dispositivos (funciones de vIAsegura_ML)…")
    malla, vx, vy, vciv = RT.cargar_malla()
    disp, camaras = RT.dispositivos(malla, vx, vy, vciv)

    print("Contando siniestros con víctimas por tramo y año…")
    base = pd.read_parquet(os.path.join(inter, "siniestros_enriquecidos.parquet"),
                           columns=["id_siniestro", "anio", "es_fatal", "con_victimas"])
    ubic = pd.read_parquet(os.path.join(inter, "siniestros_ubicacion_geo.parquet"), columns=["id_siniestro", "civ_geo"])
    sv = base.loc[(base["con_victimas"] == 1) & base["anio"].between(2015, 2025), ["id_siniestro", "anio", "es_fatal"]]
    sv = sv.merge(ubic, on="id_siniestro")
    sv = sv[sv["civ_geo"].notna() & (sv["civ_geo"] > 0)]
    conteo = (sv.groupby(["civ_geo", "anio"]).agg(y=("id_siniestro", "size"), y_fatal=("es_fatal", "sum"))
              .reset_index().rename(columns={"civ_geo": "civ"}).astype({"civ": "int64"}))

    # mismas reglas del notebook, solo para el año que se predice
    t = malla.merge(disp, on="civ", how="left")
    for k in (1, 2, 3):
        c = conteo[conteo["anio"] == ANIO - k].set_index("civ")
        t[f"y_lag{k}"] = t["civ"].map(c["y"]).fillna(0)
        t[f"_fatal{k}"] = t["civ"].map(c["y_fatal"]).fillna(0)
    t["y_prom3"] = t[["y_lag1", "y_lag2", "y_lag3"]].mean(axis=1)
    t["y_fatal_prev3"] = t[["_fatal1", "_fatal2", "_fatal3"]].sum(axis=1)
    inicio, mitad = pd.Timestamp(f"{ANIO}-01-01"), pd.Timestamp(f"{ANIO}-07-01")
    t["semaforo"] = (t["sem_fecha"].notna() & (t["sem_fecha"] <= inicio)).astype(int)
    t["resalto"] = (t["resalto_fecha"].notna() & (t["resalto_fecha"] <= mitad)).astype(int)
    activas = camaras[(camaras["inst"] <= mitad) & (camaras["desm"] > mitad)]["civ"].unique()
    t["camara"] = t["civ"].isin(activas).astype(int)
    t["vel_limite"] = np.where(t["vel_reglamentaria"] == 30, 30, np.where(t["vel_reglamentaria"] == 60, 50, np.nan))
    for c in VARS_TRAMO:
        t[c] = pd.to_numeric(t[c], errors="coerce").astype("float32")

    # nombre de la vía y localidad (salida de la fase 4 del ML)
    ref = os.path.join(inter, "riesgo_tramos_2026.parquet")
    if os.path.exists(ref):
        r = pd.read_parquet(ref, columns=["civ", "nombre_via", "localidad"]).drop_duplicates("civ")
        t = t.merge(r, on="civ", how="left")
    t["nombre_via"] = t.get("nombre_via", pd.Series(index=t.index, dtype=object)).fillna("")
    t["localidad"] = t.get("localidad", pd.Series(index=t.index, dtype=object)).fillna("")

    print("Simplificando la geometría de cada tramo…")
    geom, puntos = geometria_malla()
    t["geom"] = t["civ"].map(geom)
    t = t[t["geom"].notna()].reset_index(drop=True)

    # comprobación contra el .pkl
    sys.path.insert(0, os.path.join(RAIZ, "src"))
    from modelos_ml import ruta_modelo
    ruta = ruta_modelo()
    if ruta:
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T = joblib.load(ruta)["riesgo_tramos"]
        t["pred_2026"] = T["modelo"].predict(t[T["variables"]])
        ref = T["prediccion_2026"][["civ", "pred_2026"]].merge(t[["civ", "pred_2026"]], on="civ", suffixes=("_pkl", ""))
        dif = (ref["pred_2026"] - ref["pred_2026_pkl"]).abs()
        print(f"Comprobación: {len(ref):,} de {len(T['prediccion_2026']):,} tramos del .pkl encontrados; "
              f"diferencia máxima {dif.max():.4f}, mediana {dif.median():.5f}. "
              f"Total esperado 2026: {t['pred_2026'].sum():,.0f} (el .pkl dice {T['total_esperado_2026']:,.0f}).")

    os.makedirs(SALIDA, exist_ok=True)
    cols = ["civ", "nombre_via", "localidad", "tipo_via"] + VARS_TRAMO + ["geom"]
    t[cols].to_parquet(os.path.join(SALIDA, "tramos_2026.parquet"), index=False)
    civs = t["civ"].to_numpy(np.int64)
    np.savez_compressed(os.path.join(SALIDA, "tramos_puntos.npz"), civ=civs,
                        puntos=np.stack([puntos[c] for c in civs]).astype(np.float32))
    print(f"Listo: {len(t):,} tramos en {os.path.relpath(SALIDA, RAIZ)} ({time.time() - t0:,.0f} s)")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    construir()
