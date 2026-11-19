"""
=============================================================
  EFECTO DE UNA INTERVENCIÓN — Antes y después con grupo control
  (diferencias en diferencias, regresión de Poisson)
  Bogotá D.C.
=============================================================

Pregunta que responde
    "Cuando Bogotá hizo esta medida en estas zonas, ¿cambiaron los
    siniestros con víctimas más de lo que cambiaron en zonas parecidas
    donde no se hizo nada?"

Método
    1. Celdas tratadas: las que toca la intervención (lista de celdas, un
       corredor en GeoJSON con un ancho de influencia, o puntos con radio).
    2. Celdas control: celdas parecidas antes de la medida (mismo nivel y
       tendencia de siniestros), lejos de las tratadas para evitar contagio.
    3. Regresión de Poisson con efectos fijos de celda y de mes:
           E[y] = exp(α_celda + γ_mes + β · tratada × después)
       Efecto = exp(β) − 1, con intervalo del 95 % (errores agrupados por celda).
    4. Diagnósticos: tendencias previas (estudio de eventos) y placebo con
       una fecha falsa anterior a la real.

Uso como módulo (desde el agente):
    from efecto_intervencion import cargar_panel, celdas_de_linea, estimar_efecto
    panel = cargar_panel()
    trat = celdas_de_linea(panel, coords_corredor, ancho_m=150)
    r = estimar_efecto(panel, trat, "2018-10-01")

Uso por consola:
    python src/efecto_intervencion.py --validar                 # prueba con efectos simulados
    python src/efecto_intervencion.py --catalogo                # evalúa el catálogo revisado
    python src/efecto_intervencion.py --celdas r10_c20 r10_c21 --inicio 2018-10-01
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(RAIZ, "out", "panel", "panel.csv")
CATALOGO = os.path.join(RAIZ, "data", "intervenciones", "catalogo_intervenciones.csv")
CORREDORES = os.path.join(RAIZ, "data", "intervenciones", "corredores.geojson")
SALIDA = os.path.join(RAIZ, "out", "efectos")
Y = "n_sin_victimas"
M_LAT = 111_320.0
M_LON = 111_320.0 * np.cos(np.radians(4.65))


from contextlib import contextmanager


@contextmanager
def objetivo(columna: str):
    """Cambia temporalmente la variable que se estima (por defecto n_sin_victimas).
    Ej.: with objetivo("n_sin_fatales"): estimar_efecto(...)"""
    global Y
    previo, Y = Y, columna
    try:
        yield
    finally:
        Y = previo


def cargar_panel(ruta: str = PANEL) -> pd.DataFrame:
    return pd.read_csv(ruta, parse_dates=["mes"])


PANEL_VIAS = os.path.join(RAIZ, "out", "panel", "panel_vias.csv")


def cargar_panel_vias(ruta: str = PANEL_VIAS) -> pd.DataFrame | None:
    """Avenida × mes (siniestros sobre la vía). La columna de unidad se llama
    'celda' para reutilizar el mismo código; su valor es el nombre oficial."""
    if not os.path.exists(ruta):
        return None
    return pd.read_csv(ruta, parse_dates=["mes"]).rename(columns={"via": "celda"})


def estimar_apilado(panel, especificaciones: list[dict], distancia_min_m=1000, minimo_tratadas=3,
                    meses_pre=24, meses_post=24) -> dict:
    """Diferencias en diferencias APILADO (Cengiz et al., 2019): cada grupo de
    corredores que empezó en una fecha se compara con sus propios controles,
    y todos los grupos se estiman juntos con efectos fijos unidad×grupo y
    mes×grupo. Evita usar como control a un corredor que ya fue tratado y da
    más precisión que cada grupo por separado. Errores agrupados por unidad."""
    pilas, usados = [], []
    for i, e in enumerate(especificaciones):
        r = estimar_efecto(panel, e["tratadas"], e["inicio"], meses_pre=meses_pre, meses_post=meses_post,
                           fin=e.get("fin"), excluir=e.get("excluir"), distancia_min_m=distancia_min_m,
                           meses_transicion=e.get("transicion", 1),
                           minimo_tratadas=minimo_tratadas, devolver_datos=True, diagnosticos=False)
        if "error" in r:
            continue
        d = r["datos"].copy()
        inicio = pd.Timestamp(e["inicio"]).to_period("M").to_timestamp()
        d["rel"] = (d["mes"].dt.year - inicio.year) * 12 + d["mes"].dt.month - inicio.month
        d["cluster"] = d["celda"]
        d["celda"] = f"g{i}|" + d["celda"].astype(str)
        d["mes"] = f"g{i}|" + d["mes"].dt.strftime("%Y-%m")
        pilas.append(d)
        usados.append(e.get("etiqueta", f"grupo {i}"))
    if not pilas:
        return {"error": "Ningún grupo tiene datos suficientes."}
    d = pd.concat(pilas, ignore_index=True)
    fit = AjustePoissonFE(d, ["trat_despues"])
    b, se = fit.params["trat_despues"], fit.bse["trat_despues"]
    post = d[(d["tratada"] == 1) & (d["despues"] == 1)]
    r = {"grupos": usados, "n_grupos": len(usados),
         "unidades_tratadas": int(d.loc[d.tratada == 1, "cluster"].nunique()),
         "unidades_control": int(d.loc[d.tratada == 0, "cluster"].nunique()),
         "efecto_pct": round(float(100 * np.expm1(b)), 1),
         "ic95_pct": [round(float(100 * np.expm1(b - 1.96 * se)), 1), round(float(100 * np.expm1(b + 1.96 * se)), 1)],
         "p_valor": round(float(fit.pvalues["trat_despues"]), 4),
         "siniestros_tratadas_antes": int(d[(d.tratada == 1) & (d.despues == 0)][Y].sum()),
         "siniestros_victimas_tratadas_despues": int(post[Y].sum()),
         "siniestros_evitados_estimados": round(float(post[Y].sum() / np.exp(b) - post[Y].sum()), 1)}
    # tendencias previas: semestres relativos, referencia = semestre −1
    d["sem"] = np.floor(d["rel"] / 6).astype(int)
    cols = []
    for s_ in sorted(d.loc[d["despues"] == 0, "sem"].unique()):
        if s_ in (-1,) or s_ >= 0:
            continue
        c = f"pre_m{abs(s_)}"
        d[c] = ((d["sem"] == s_) & (d["tratada"] == 1)).astype(int)
        cols.append(c)
    if cols:
        try:
            fe = AjustePoissonFE(d, cols + ["trat_despues"])
            r["tendencias_previas_p"] = round(fe.wald_p(cols), 4)
            r["previos_pct"] = {c: round(float(100 * np.expm1(fe.params[c])), 1) for c in cols}
        except Exception as ex:
            r["tendencias_previas_error"] = str(ex)
    r["advertencias"] = _advertencias({**r, "diagnosticos": {"tendencias_previas_p": r.get("tendencias_previas_p")}})
    return r


# ════════════════════════════════════════════════════════════
#  ¿QUÉ CELDAS TOCA LA INTERVENCIÓN?
# ════════════════════════════════════════════════════════════
def _celdas_xy(panel):
    c = panel.drop_duplicates("celda")[["celda", "lat", "lon"]].reset_index(drop=True)
    return c, c["lon"].to_numpy() * M_LON, c["lat"].to_numpy() * M_LAT


def celdas_de_linea(panel, coords: list[list[float]], ancho_m: float = 150, tam_celda: float = 500) -> list[str]:
    """Celdas cuyo centro está a menos de ancho_m + media celda de la línea [[lon, lat], ...]."""
    c, x, y = _celdas_xy(panel)
    pts = np.array([[lo * M_LON, la * M_LAT] for lo, la in coords])
    dmin = np.full(len(c), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
        dmin = np.minimum(dmin, np.hypot(x - (a[0] + t * ab[0]), y - (a[1] + t * ab[1])))
    return c.loc[dmin <= ancho_m + tam_celda / 2, "celda"].tolist()


def celdas_de_puntos(panel, puntos: list[list[float]], radio_m: float = 300, tam_celda: float = 500) -> list[str]:
    """Celdas cuyo centro está a menos de radio_m + media celda de algún punto [[lon, lat], ...]
    (cámaras, semáforos, colegios). Así siempre entra la celda que contiene el punto."""
    c, x, y = _celdas_xy(panel)
    sel = np.zeros(len(c), bool)
    for lo, la in puntos:
        sel |= np.hypot(x - lo * M_LON, y - la * M_LAT) <= radio_m + tam_celda / 2
    return c.loc[sel, "celda"].tolist()


def _clave_localidad(x) -> str:
    """'Engativá' / 'ENGATIVA' / 'La Candelaria' / 'CANDELARIA' → misma clave (sin tildes ni artículos)."""
    import re as _re
    import unicodedata as _ud
    t = "".join(ch for ch in _ud.normalize("NFD", str(x).lower()) if _ud.category(ch) != "Mn")
    return _re.sub(r"^(?:la|los|el)\s+", "", t.strip())


def celdas_de_localidad(panel, localidad: str) -> list[str]:
    clave = _clave_localidad(localidad)
    claves = panel["localidad"].map(_clave_localidad)
    return panel.loc[claves == clave, "celda"].unique().tolist()


# ════════════════════════════════════════════════════════════
#  GRUPO CONTROL
# ════════════════════════════════════════════════════════════
def elegir_controles(panel, tratadas, inicio, meses_pre=24, distancia_min_m=1000, por_tratada=3, semilla=0, excluir=()):
    """Por cada celda tratada, las `por_tratada` celdas no tratadas más parecidas en
    nivel y tendencia de siniestros, a más de distancia_min_m.

    El parecido se mide en una ventana ANTERIOR a la que se usa para estimar
    (meses −2·meses_pre a −meses_pre). Si se emparejara con los mismos meses
    que luego se comparan, el ruido de esos meses produce regresión a la media
    y efectos falsos. Si no hay al menos 12 meses en esa ventana, se usa la
    ventana previa normal y se avisa."""
    inicio = pd.Timestamp(inicio)
    ini_ventana = inicio - pd.DateOffset(months=2 * meses_pre)
    fin_ventana = inicio - pd.DateOffset(months=meses_pre)
    ventana = panel[(panel["mes"] >= ini_ventana) & (panel["mes"] < fin_ventana)]
    if ventana["mes"].nunique() >= 12:
        pre = ventana
        elegir_controles.ventana = "independiente"
    else:
        pre = panel[(panel["mes"] >= fin_ventana) & (panel["mes"] < inicio)]
        elegir_controles.ventana = "misma_que_estimacion"
    pre = pre.assign(t=(pre["mes"].dt.year * 12 + pre["mes"].dt.month))
    perfil = pre.groupby("celda").agg(nivel=(Y, "mean"), lat=("lat", "first"), lon=("lon", "first"))
    pend = pre.groupby("celda").apply(lambda d: np.polyfit(d["t"], d[Y], 1)[0] if d[Y].sum() else 0.0)
    perfil["pendiente"] = pend
    perfil["log_nivel"] = np.log(perfil["nivel"] + 0.05)

    trat = perfil.loc[perfil.index.intersection(tratadas)]
    # fuera: las tratadas y las celdas de OTRAS intervenciones del mismo tipo
    # (un corredor tratado meses después contaminaría el grupo control)
    cand = perfil.drop(index=trat.index.union(pd.Index(list(excluir))), errors="ignore")
    # lejos de cualquier tratada
    tx, ty = trat["lon"].to_numpy() * M_LON, trat["lat"].to_numpy() * M_LAT
    cx, cy = cand["lon"].to_numpy() * M_LON, cand["lat"].to_numpy() * M_LAT
    dist = np.min(np.hypot(cx[:, None] - tx[None], cy[:, None] - ty[None]), axis=1) if len(trat) else np.inf
    cand = cand[dist >= distancia_min_m]
    # distancia en nivel (log) y tendencia, estandarizadas
    z = lambda s: (s - perfil[s.name].mean()) / (perfil[s.name].std() or 1)
    ct, cc = trat[["log_nivel", "pendiente"]].apply(z), cand[["log_nivel", "pendiente"]].apply(z)
    usados, elegidos = set(), []
    orden = trat.sort_values("nivel", ascending=False).index  # las más cargadas eligen primero
    for celda in orden:
        d = np.hypot(cc["log_nivel"] - ct.loc[celda, "log_nivel"], 0.5 * (cc["pendiente"] - ct.loc[celda, "pendiente"]))
        n = 0
        for c in d.sort_values().index:
            if c not in usados:
                usados.add(c)
                elegidos.append(c)
                n += 1
                if n == por_tratada:
                    break
    return elegidos, perfil


# ════════════════════════════════════════════════════════════
#  ESTIMACIÓN
# ════════════════════════════════════════════════════════════
class AjustePoissonFE:
    """Regresión de Poisson con efectos fijos de celda y de mes (PPML con efectos
    fijos absorbidos, como ppmlhdfe). No arma columnas ficticias por celda, así
    que usa poca memoria aunque haya miles de celdas. Errores estándar
    agrupados por celda."""

    def __init__(self, d: pd.DataFrame, columnas: list[str], tol=1e-9, max_iter=200):
        from scipy import stats
        self._stats = stats
        d = d[d.groupby("celda")[Y].transform("sum") > 0]          # celdas siempre en 0 no aportan
        d = d[d.groupby("mes")[Y].transform("sum") > 0]
        y = d[Y].to_numpy(float)
        X = d[columnas].to_numpy(float)
        ci = pd.factorize(d["celda"])[0]
        ti = pd.factorize(d["mes"])[0]
        gi = pd.factorize(d["cluster"])[0] if "cluster" in d.columns else ci   # grupos para el error estándar
        nc, nt = ci.max() + 1, ti.max() + 1

        def desmediar(M, w):
            M = M.copy()
            sw_c = np.bincount(ci, w, nc)
            sw_t = np.bincount(ti, w, nt)
            for _ in range(500):
                antes = M.copy()
                for idx, sw, n in ((ci, sw_c, nc), (ti, sw_t, nt)):
                    for k in range(M.shape[1]):
                        M[:, k] -= (np.bincount(idx, w * M[:, k], n) / sw)[idx]
                if np.max(np.abs(M - antes)) < 1e-11:
                    break
            return M

        mu = (y + y.mean()) / 2
        eta = np.log(mu)
        dev_prev = np.inf
        for _ in range(max_iter):
            z = eta + (y - mu) / mu
            w = mu
            ZX = desmediar(np.column_stack([z, X]), w)
            zt, Xt = ZX[:, 0], ZX[:, 1:]
            XtW = Xt.T * w
            beta = np.linalg.solve(XtW @ Xt, XtW @ zt)
            eta = z - (zt - Xt @ beta)
            mu = np.exp(np.clip(eta, -30, 30))
            dev = 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))
            if abs(dev - dev_prev) / max(abs(dev), 1) < tol:
                break
            dev_prev = dev
        # varianza agrupada por celda (sándwich), con X desmediada en la última iteración
        Xt = desmediar(X, mu)
        pan = np.linalg.inv((Xt.T * mu) @ Xt)
        sc = Xt * (y - mu)[:, None]
        G = gi.max() + 1
        S = np.vstack([np.bincount(gi, sc[:, k], G) for k in range(X.shape[1])]).T
        V = pan @ (S.T @ S) @ pan * G / (G - 1)
        self.columnas = columnas
        self.params = pd.Series(beta, index=columnas)
        self.cov = pd.DataFrame(V, index=columnas, columns=columnas)
        self.bse = pd.Series(np.sqrt(np.diag(V)), index=columnas)
        self.pvalues = pd.Series(2 * stats.norm.sf(np.abs(beta / self.bse.to_numpy())), index=columnas)
        self.n_obs, self.n_celdas = len(y), G

    def wald_p(self, cols: list[str]) -> float:
        b = self.params[cols].to_numpy()
        V = self.cov.loc[cols, cols].to_numpy()
        w = float(b @ np.linalg.solve(V, b))
        return float(self._stats.chi2.sf(w, len(cols)))


def _poisson_fe(d, termino):
    return AjustePoissonFE(d, [t.strip() for t in termino.split("+")])


def estimar_efecto(panel, tratadas, inicio, meses_pre=24, meses_post=24, meses_transicion=1,
                   distancia_min_m=1000, por_tratada=3, controles=None, diagnosticos=True,
                   excluir=None, fin=None, minimo_tratadas=3, devolver_datos=False) -> dict:
    """fin: último día que se puede usar después de la medida (p. ej. antes de que
    otra medida para toda la ciudad o la pandemia alteren también a los controles)."""
    inicio = pd.Timestamp(inicio).to_period("M").to_timestamp()
    tratadas = [c for c in dict.fromkeys(tratadas) if c in set(panel["celda"])]
    if len(tratadas) < minimo_tratadas:
        return {"error": f"Solo {len(tratadas)} unidades tratadas dentro del panel; se necesitan al menos {minimo_tratadas}."}
    if controles is None:
        controles, perfil = elegir_controles(panel, tratadas, inicio, meses_pre, distancia_min_m, por_tratada,
                                             excluir=set(excluir or ()) - set(tratadas))
        ventana_emparejamiento = elegir_controles.ventana
    else:
        ventana_emparejamiento = "controles_dados"
    desde = inicio - pd.DateOffset(months=meses_pre)
    hasta = inicio + pd.DateOffset(months=meses_transicion + meses_post)
    if fin is not None and str(fin) not in ("", "nan", "None"):
        hasta = min(hasta, pd.Timestamp(fin).to_period("M").to_timestamp() + pd.DateOffset(months=1))
    d = panel[panel["celda"].isin(tratadas + list(controles)) & (panel["mes"] >= desde) & (panel["mes"] < hasta)].copy()
    if d["mes"].max() < inicio + pd.DateOffset(months=meses_transicion + 6):
        return {"error": "No hay al menos 6 meses de datos después de la medida."}
    d["tratada"] = d["celda"].isin(tratadas).astype(int)
    d["despues"] = (d["mes"] >= inicio + pd.DateOffset(months=meses_transicion)).astype(int)
    d = d[~((d["mes"] >= inicio) & (d["despues"] == 0))]          # excluye meses de transición
    d["trat_despues"] = d["tratada"] * d["despues"]
    if devolver_datos:
        return {"datos": d, "controles": list(controles), "ventana_emparejamiento": ventana_emparejamiento}

    fit = _poisson_fe(d, "trat_despues")
    b, se = fit.params["trat_despues"], fit.bse["trat_despues"]
    tabla = d.groupby(["tratada", "despues"])[Y].mean().unstack()
    pre_t, post_t = d[(d.tratada == 1) & (d.despues == 0)], d[(d.tratada == 1) & (d.despues == 1)]
    meses_post_obs = post_t["mes"].nunique()
    esperado_sin = post_t[Y].sum() / np.exp(b)
    r = {
        "inicio": str(inicio.date()), "celdas_tratadas": len(tratadas), "celdas_control": len(controles),
        "meses_antes": int(d.loc[d.despues == 0, "mes"].nunique()), "meses_despues": int(meses_post_obs),
        "efecto_pct": round(float(100 * (np.exp(b) - 1)), 1),
        "ic95_pct": [round(float(100 * (np.exp(b - 1.96 * se) - 1)), 1), round(float(100 * (np.exp(b + 1.96 * se) - 1)), 1)],
        "p_valor": round(float(fit.pvalues["trat_despues"]), 4),
        "siniestros_victimas_tratadas_despues": int(post_t[Y].sum()),
        "siniestros_evitados_estimados": round(float(esperado_sin - post_t[Y].sum()), 1),
        "promedio_mensual_celda": {"tratadas_antes": round(float(tabla.loc[1, 0]), 3), "tratadas_despues": round(float(tabla.loc[1, 1]), 3),
                                   "control_antes": round(float(tabla.loc[0, 0]), 3), "control_despues": round(float(tabla.loc[0, 1]), 3)},
        "siniestros_tratadas_antes": int(pre_t[Y].sum()),
        "ventana_emparejamiento": ventana_emparejamiento,
    }
    r["significativo_5pct"] = bool(r["p_valor"] < 0.05)
    if diagnosticos:
        r["diagnosticos"] = _diagnosticos(panel, d, tratadas, list(controles), inicio, meses_pre)
    r["advertencias"] = _advertencias(r)
    return r


def _diagnosticos(panel, d, tratadas, controles, inicio, meses_pre):
    out = {}
    # estudio de eventos por semestres relativos a la medida (referencia: semestre -1)
    e = d.copy()
    rel = ((e["mes"].dt.year - inicio.year) * 12 + e["mes"].dt.month - inicio.month)
    e["sem"] = np.floor(rel / 6).astype(int)
    terminos = []
    for s in sorted(e["sem"].unique()):
        if s == -1:
            continue
        col = f"ev_{'m' if s < 0 else 'p'}{abs(s)}"
        e[col] = ((e["sem"] == s) & (e["tratada"] == 1)).astype(int)
        terminos.append((s, col))
    try:
        fe = _poisson_fe(e, " + ".join(c for _, c in terminos))
        out["estudio_eventos_pct"] = {f"semestre_{s:+d}": round(100 * (np.exp(fe.params[c]) - 1), 1) for s, c in terminos}
        previos = [c for s, c in terminos if s < -1]
        if previos:
            out["tendencias_previas_p"] = round(fe.wald_p(previos), 4)
    except Exception as ex:
        out["estudio_eventos_error"] = str(ex)
    # placebo: misma comparación con una fecha falsa 12 meses antes, solo con datos previos
    try:
        falso = inicio - pd.DateOffset(months=12)
        sub = panel[panel["mes"] < inicio]
        pr = estimar_efecto(sub, tratadas, falso, meses_pre=min(meses_pre, 12), meses_post=11,
                            controles=controles, diagnosticos=False)
        out["placebo_12m_antes"] = {k: pr.get(k) for k in ("efecto_pct", "ic95_pct", "p_valor")}
    except Exception as ex:
        out["placebo_error"] = str(ex)
    return out


def _advertencias(r):
    a = []
    if r["siniestros_tratadas_antes"] < 30:
        a.append("Pocos siniestros en la zona antes de la medida: el estimado es muy incierto.")
    dg = r.get("diagnosticos", {})
    if dg.get("tendencias_previas_p") is not None and dg["tendencias_previas_p"] < 0.05:
        a.append("Las zonas tratadas ya venían cambiando distinto antes de la medida: el efecto puede estar sesgado.")
    pl = dg.get("placebo_12m_antes") or {}
    if pl.get("p_valor") is not None and pl["p_valor"] < 0.05:
        a.append("El placebo con fecha falsa dio un efecto significativo: revisar la elección de controles.")
    if r.get("ventana_emparejamiento") == "misma_que_estimacion":
        a.append("Los controles se emparejaron con los mismos meses de la comparación (no había historia suficiente): riesgo de regresión a la media.")
    if r["ic95_pct"][0] < 0 < r["ic95_pct"][1]:
        a.append("El intervalo incluye 0 %: no se puede descartar que la medida no tuvo efecto.")
    return a


# ════════════════════════════════════════════════════════════
#  VALIDACIÓN CON EFECTOS SIMULADOS
# ════════════════════════════════════════════════════════════
def validar(panel, efectos=(-0.30, -0.15, 0.0, 0.15), repeticiones=20, n_celdas=40, semilla=7):
    """Inyecta un efecto conocido en celdas al azar y mide si el método lo recupera."""
    rng = np.random.default_rng(semilla)
    # elegibilidad medida en 2015, antes de cualquier ventana de estimación,
    # para que la propia selección de celdas no induzca regresión a la media
    nivel = panel[panel["anio"] == 2015].groupby("celda")[Y].mean()
    elegibles = nivel[nivel >= 0.3].index.to_numpy()
    filas = []
    for ef in efectos:
        for rep in range(repeticiones):
            inicio = pd.Timestamp(rng.choice(pd.date_range("2019-01-01", "2019-12-01", freq="MS")))
            trat = list(rng.choice(elegibles, n_celdas, replace=False))
            p = panel.copy()
            m = p["celda"].isin(trat) & (p["mes"] >= inicio + pd.DateOffset(months=1))
            if ef < 0:   # reducción: cada siniestro "sobrevive" con probabilidad 1+ef
                p.loc[m, Y] = rng.binomial(p.loc[m, Y].to_numpy(), 1 + ef)
            elif ef > 0:  # aumento: se suman siniestros extra con media y·ef
                p.loc[m, Y] = p.loc[m, Y] + rng.poisson(p.loc[m, Y].to_numpy() * ef)
            r = estimar_efecto(p, trat, inicio, diagnosticos=False)
            if "error" in r:
                continue
            filas.append({"efecto_real_pct": 100 * ef, "estimado_pct": r["efecto_pct"],
                          "ic_inf": r["ic95_pct"][0], "ic_sup": r["ic95_pct"][1], "p": r["p_valor"]})
        print(f"   efecto simulado {100 * ef:+.0f} %: {len([f for f in filas if f['efecto_real_pct'] == 100 * ef])} repeticiones")
    df = pd.DataFrame(filas)
    df["cubre"] = (df["ic_inf"] <= df["efecto_real_pct"]) & (df["efecto_real_pct"] <= df["ic_sup"])
    df["detecta"] = df["p"] < 0.05
    resumen = df.groupby("efecto_real_pct").agg(estimado_promedio=("estimado_pct", "mean"),
                                                sesgo=("estimado_pct", lambda s: s.mean()),
                                                cobertura_ic95=("cubre", "mean"),
                                                tasa_deteccion=("detecta", "mean"),
                                                ancho_ic=("ic_sup", "mean"))
    resumen["sesgo"] = resumen["estimado_promedio"] - resumen.index
    resumen["ancho_ic"] = (df.groupby("efecto_real_pct")["ic_sup"].mean() - df.groupby("efecto_real_pct")["ic_inf"].mean())
    return resumen.round(3), df


# ════════════════════════════════════════════════════════════
#  CATÁLOGO DE INTERVENCIONES (salida de catalogo_resoluciones.py)
# ════════════════════════════════════════════════════════════
CATALOGO_MANUAL = os.path.join(RAIZ, "data", "intervenciones", "catalogo_manual.csv")
CATALOGO_DATOS = os.path.join(RAIZ, "data", "intervenciones", "catalogo_datos.csv")


def _es_si(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "si", "sí")


def cargar_catalogo() -> pd.DataFrame:
    """Une el catálogo manual (verificado) con las filas revisadas del automático."""
    partes = []
    for ruta, origen in ((CATALOGO_MANUAL, "manual"), (CATALOGO_DATOS, "datos")):
        if os.path.exists(ruta):
            m = pd.read_csv(ruta, dtype=str).fillna("")
            partes.append(m[m["revisado"].map(_es_si)].assign(origen=origen))
    if os.path.exists(CATALOGO):
        a = pd.read_csv(CATALOGO, dtype=str).fillna("")
        a = a[a["revisado"].map(_es_si)].copy()
        if len(a):
            a["id_intervencion"] = a["id_norma"] + "_" + a["corredor"].str.replace(r"\W+", "_", regex=True)
            a["grupo"] = a["id_norma"]
            a["fecha_inicio"] = a["fecha_inicio"].where(a["fecha_inicio"] != "", a.get("fecha_expedicion", ""))
            partes.append(a.assign(origen="automatico"))
    if not partes:
        return pd.DataFrame()
    cat = pd.concat(partes, ignore_index=True).fillna("")
    for c in ("fin_ventana", "localidad", "corredor", "valor_anterior", "valor_nuevo", "direccion",
              "geometria_geojson", "radio_m", "celdas_ids", "excluir_ids", "meses_transicion"):
        if c not in cat:
            cat[c] = ""
    return cat


def _celdas_fila(panel, f, lineas) -> list[str] | None:
    if f.get("celdas_ids"):
        validas = set(panel["celda"])
        return [c for c in f["celdas_ids"].split(";") if c in validas]
    if f.get("geometria_geojson"):
        geo = json.loads(f["geometria_geojson"])
        radio = float(f.get("radio_m") or 150)
        if geo["type"] == "Point":
            return celdas_de_puntos(panel, [geo["coordinates"]], radio)
        if geo["type"] == "MultiPoint":
            return celdas_de_puntos(panel, geo["coordinates"], radio)
        partes = [geo["coordinates"]] if geo["type"] == "LineString" else geo["coordinates"]
        return sorted({c for p in partes for c in celdas_de_linea(panel, p, radio)})
    g = lineas.get(f["corredor"]) if f["corredor"] else None
    if g and g["type"] == "LineString":
        return celdas_de_linea(panel, g["coordinates"])
    if g and g["type"] == "MultiLineString":
        return sorted({c for parte in g["coordinates"] for c in celdas_de_linea(panel, parte)})
    loc = f["localidad"]
    if loc and ";" not in loc and loc.upper() != "TODA LA CIUDAD":
        return celdas_de_localidad(panel, loc)
    return None


def fin_grupo(sub: pd.DataFrame, inicio: str) -> str:
    """Fin de la ventana de un grupo: si la mayoría de sus intervenciones terminó
    (p. ej. cámaras desmontadas), la mediana de esas fechas; si no, sin límite.
    Se ignoran fechas de fin anteriores a 6 meses después del inicio (errores de registro)."""
    fines = pd.to_datetime(pd.Series([x for x in sub["fin_ventana"] if x]), errors="coerce").dropna()
    if "origen" in sub and (sub["origen"] != "datos").all():
        # catálogo manual: el fin es una decisión de diseño (p. ej. antes de la pandemia); se respeta
        return str(fines.min().date()) if len(fines) else ""
    minimo = pd.Timestamp(inicio) + pd.DateOffset(months=6)
    validos = fines[fines >= minimo]
    if len(validos) > len(sub) / 2:
        return str(validos.median().date())
    return ""


def evaluar_catalogo(panel):
    cat = cargar_catalogo()
    if cat.empty:
        print("No hay intervenciones revisadas (catalogo_manual.csv o catalogo_intervenciones.csv).")
        return pd.DataFrame()
    geo = json.load(open(CORREDORES, encoding="utf-8")) if os.path.exists(CORREDORES) else {"features": []}
    lineas = {f["properties"].get("corredor"): f["geometry"] for f in geo["features"]}
    cat["celdas"] = pd.Series([_celdas_fila(panel, f, lineas) or [] for _, f in cat.iterrows()], index=cat.index, dtype=object)

    # celdas tocadas por cualquier intervención del mismo tipo (se excluyen de los controles)
    por_tipo = {t: {c for cs in g["celdas"] for c in cs} for t, g in cat.groupby("tipo_medida")}
    campos = ["id_intervencion", "grupo", "tipo_medida", "direccion", "valor_anterior", "valor_nuevo",
              "corredor", "localidad", "fecha_inicio", "fin_ventana", "fuente", "origen"]
    salida = []

    def _transicion(sub) -> int:
        v = [int(float(x)) for x in sub["meses_transicion"] if str(x).strip() not in ("", "nan")]
        return max(v) if v else 1

    def _excluir_extra(sub) -> set:
        return {c for x in sub["excluir_ids"] if x for c in x.split(";")}

    def correr(etiqueta, nivel, base, celdas, inicio, fin, tipo, transicion=1, extra=frozenset()):
        if not celdas:
            if base.get("localidad", "").upper() == "TODA LA CIUDAD":
                msg = "medida para toda la ciudad: no hay grupo control; no se estima con antes/después"
            else:
                msg = "sin geometría: agregue el corredor a corredores.geojson"
            salida.append({**base, "nivel_estimacion": nivel, "error": msg})
            print(f"   – {etiqueta}: {msg}")
            return
        r = estimar_efecto(panel, celdas, inicio, fin=fin, excluir=por_tipo.get(tipo, set()) | set(extra),
                           meses_transicion=transicion)
        fila = {**base, "nivel_estimacion": nivel, "n_celdas": len(celdas),
                **{k: v for k, v in r.items() if k != "diagnosticos"},
                "diagnosticos": json.dumps(r.get("diagnosticos", {}), ensure_ascii=False)}
        for k in ("ic95_pct", "advertencias"):
            if k in fila and not isinstance(fila[k], str):
                fila[k] = json.dumps(fila[k], ensure_ascii=False, default=float)
        salida.append(fila)
        if "error" in r:
            print(f"   – {etiqueta}: {r['error']}")
        else:
            dg = r.get("diagnosticos", {})
            print(f"   ✓ {etiqueta}: {r['efecto_pct']:+.1f} % IC95 {r['ic95_pct']} · p={r['p_valor']} · "
                  f"{r['celdas_tratadas']} celdas · {r['meses_despues']} meses después · "
                  f"tendencias previas p={dg.get('tendencias_previas_p')}")

    print("Por corredor / intervención:")
    for _, f in cat.iterrows():
        if f["tipo_medida"] == "fotodeteccion":
            continue          # una cámara toca 1–2 celdas: se estima por trimestre de instalación
        sub1 = cat.loc[[_]]
        correr(f"{f['id_intervencion']} · {f['corredor'] or f['localidad']}", "corredor",
               {c: f.get(c, "") for c in campos}, f["celdas"], f["fecha_inicio"], f["fin_ventana"], f["tipo_medida"],
               _transicion(sub1), _excluir_extra(sub1))

    print("\nPor grupo (corredores que empezaron a la vez, estimados juntos = más precisión):")
    for g, sub in cat.groupby("grupo"):
        celdas = sorted({c for cs in sub["celdas"] for c in cs})
        if len(sub) < 2 or not celdas:
            continue
        f = sub.iloc[0]
        base = {c: f.get(c, "") for c in campos}
        base.update(id_intervencion=f"grupo_{g}", corredor="; ".join(sub["corredor"].replace("", np.nan).dropna()))
        inicio = min(sub["fecha_inicio"])          # el grupo empieza con la primera intervención
        correr(f"grupo {g} ({len(sub)})", "grupo", base, celdas, inicio, fin_grupo(sub, inicio), f["tipo_medida"],
               _transicion(sub), _excluir_extra(sub))

    # ── por AVENIDA (siniestros sobre la vía) y APILADO ─────────────────
    pv = cargar_panel_vias()
    try:
        from corredores_desde_malla import NOMBRES_OFICIALES
    except Exception:
        NOMBRES_OFICIALES = {}
    unidades_via = set(pv["celda"]) if pv is not None else set()
    cat["vias"] = pd.Series([[n for n in NOMBRES_OFICIALES.get(c, []) if n in unidades_via] for c in cat["corredor"]],
                            index=cat.index, dtype=object)
    vias_por_tipo = {t: {v for vs in g["vias"] for v in vs} for t, g in cat.groupby("tipo_medida")}

    def guardar(etiqueta, nivel, base, r):
        fila = {**base, "nivel_estimacion": nivel, **r}
        for k in ("ic95_pct", "advertencias", "grupos", "previos_pct"):
            if k in fila and not isinstance(fila[k], str):
                fila[k] = json.dumps(fila[k], ensure_ascii=False, default=float)
        salida.append(fila)
        if "error" in r:
            print(f"   – {etiqueta}: {r['error']}")
        else:
            print(f"   ✓ {etiqueta}: {r['efecto_pct']:+.1f} % IC95 {r['ic95_pct']} · p={r['p_valor']} · "
                  f"tendencias previas p={r.get('tendencias_previas_p')}")

    if pv is not None:
        print("\nPor avenida (solo siniestros sobre la vía; controles = otras avenidas):")
        for g, sub in cat.groupby("grupo"):
            vias = sorted({v for vs in sub["vias"] for v in vs})
            if not vias:
                continue
            f = sub.iloc[0]
            base = {c: f.get(c, "") for c in campos}
            base.update(id_intervencion=f"vias_{g}", corredor="; ".join(vias))
            r = estimar_efecto(pv, vias, f["fecha_inicio"], fin=f["fin_ventana"], distancia_min_m=0,
                               minimo_tratadas=1, excluir=vias_por_tipo.get(f["tipo_medida"], set()))
            r = {k: v for k, v in r.items() if k != "diagnosticos"} | {"diagnosticos": json.dumps(r.get("diagnosticos", {}), ensure_ascii=False)} \
                if "error" not in r else r
            if "error" not in r:
                r["tendencias_previas_p"] = json.loads(r["diagnosticos"]).get("tendencias_previas_p")
                if len(vias) < 5:
                    r["advertencias"] = list(r.get("advertencias") or []) + [
                        f"Solo {len(vias)} avenida(s) tratada(s): el error estándar no es confiable; use la estimación apilada."]
            guardar(f"avenidas grupo {g} ({len(vias)})", "grupo_vias", base, r)

    print("\nApilado (todas las fechas juntas, cada una con sus controles):")
    combos = [(o, n, pan, col, dist, minimo, excl)
              for o in ("n_sin_victimas", "n_sin_fatales")
              for n, pan, col, dist, minimo, excl in (("apilado_celdas", panel, "celdas", 1000, 3, por_tipo),
                                                     ("apilado_vias", pv, "vias", 0, 1, vias_por_tipo))]
    for tipo, sub in cat.groupby("tipo_medida"):
      for obj, nivel, pan, col, dist, minimo, excl in combos:
        with objetivo(obj):
            if pan is None:
                continue
            specs = []
            for g, sg in sub.groupby("grupo"):
                unidades = sorted({u for us in sg[col] for u in us})
                if unidades:
                    ini = min(sg["fecha_inicio"])
                    specs.append({"tratadas": unidades, "inicio": ini, "fin": fin_grupo(sg, ini),
                                  "excluir": excl.get(tipo, set()) | _excluir_extra(sg), "etiqueta": g,
                                  "transicion": _transicion(sg)})
            if len(specs) < 2:
                continue
            r = estimar_apilado(pan, specs, distancia_min_m=dist, minimo_tratadas=minimo)
            f = sub.iloc[0]
            base = {c: f.get(c, "") for c in campos}
            base.update(id_intervencion=f"{nivel}_{tipo}_{obj}", grupo=nivel, corredor="varios", fecha_inicio="varias",
                        objetivo=obj)
            guardar(f"{tipo} · {nivel} · {obj}", nivel, base, r)

    out = pd.DataFrame(salida)
    out.to_csv(os.path.join(SALIDA, "efectos_catalogo.csv"), index=False, encoding="utf-8-sig")
    print(f"\n✅ {len(out)} estimaciones en {os.path.join(SALIDA, 'efectos_catalogo.csv')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validar", action="store_true", help="prueba el método con efectos simulados")
    ap.add_argument("--catalogo", action="store_true", help="estima el efecto de cada intervención revisada")
    ap.add_argument("--celdas", nargs="+", help="celdas tratadas (ids del panel)")
    ap.add_argument("--localidad", help="toda una localidad como zona tratada")
    ap.add_argument("--inicio", help="fecha de inicio de la medida AAAA-MM-DD")
    ap.add_argument("--repeticiones", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(SALIDA, exist_ok=True)
    panel = cargar_panel()

    if args.validar:
        print("Validando con efectos simulados (tarda unos minutos)…")
        resumen, detalle = validar(panel, repeticiones=args.repeticiones)
        detalle.to_csv(os.path.join(SALIDA, "validacion_detalle.csv"), index=False)
        resumen.to_csv(os.path.join(SALIDA, "validacion_resumen.csv"))
        print("\n", resumen.to_string())
    elif args.catalogo:
        evaluar_catalogo(panel)
    elif args.inicio and (args.celdas or args.localidad):
        trat = args.celdas or celdas_de_localidad(panel, args.localidad)
        r = estimar_efecto(panel, trat, args.inicio)
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
