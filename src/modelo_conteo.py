"""
=============================================================
  MODELO DE CONTEO — Siniestros con víctimas por celda y mes
  Bogotá D.C.
=============================================================

Para qué sirve en el chatbot
    Da la LÍNEA BASE: cuántos siniestros con víctimas se esperan en una zona
    si no se hace nada. El efecto de una medida (efecto_intervencion.py y
    factores_referencia.py) se aplica sobre esta línea base.

Qué hace
    1. Lee out/panel/panel.csv (salida de panel_espacial.py).
    2. Crea variables solo con información del pasado (rezagos, promedios
       móviles, historia de la celda, estacionalidad, pandemia).
    3. Compara cuatro modelos con partición temporal:
         entrenamiento 2016–2021 · validación 2022 · prueba 2023–2024
         - ingenuo: promedio de los últimos 12 meses de la celda
         - estacional: mismo mes del año anterior
         - GLM Poisson (interpretable)
         - LightGBM con objetivo Poisson
    4. Reentrena el mejor con todos los datos y lo guarda.

Salidas (out/modelo_conteo/)
    metricas.json          error de cada modelo en validación y prueba
    predicciones_prueba.csv
    lgbm_conteo.txt        modelo final
    variables.json         lista y orden de variables

Uso
    pip install pandas numpy statsmodels lightgbm
    python src/modelo_conteo.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(RAIZ, "out", "panel", "panel.csv")
SALIDA = os.path.join(RAIZ, "out", "modelo_conteo")
Y = "n_sin_victimas"


# ════════════════════════════════════════════════════════════
#  VARIABLES (solo pasado → sin fuga de información)
# ════════════════════════════════════════════════════════════
def crear_variables(p: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values(["celda", "mes"]).copy()
    g = p.groupby("celda")[Y]
    for k in (1, 2, 3, 12):
        p[f"lag{k}"] = g.shift(k)
    prev = g.shift(1)
    gp = prev.groupby(p["celda"])
    p["ma3"] = gp.transform(lambda s: s.rolling(3, min_periods=3).mean())
    p["ma12"] = gp.transform(lambda s: s.rolling(12, min_periods=12).mean())
    p["ma24"] = gp.transform(lambda s: s.rolling(24, min_periods=12).mean())
    p["hist_media"] = gp.transform(lambda s: s.expanding(min_periods=12).mean())
    p["muertos_12m"] = p.groupby("celda")["n_muertos"].transform(lambda s: s.shift(1).rolling(12, min_periods=12).sum())
    for c in ("n_vict_peaton", "n_vict_moto", "n_vict_ciclista", "n_vict_velocidad"):
        p[c.replace("n_vict_", "pct12_")] = (
            p.groupby("celda")[c].transform(lambda s: s.shift(1).rolling(12, min_periods=12).sum())
            / (p["ma12"] * 12).replace(0, np.nan)).fillna(0)
    ciudad = p.groupby("mes")[Y].sum().sort_index()
    p["ciudad_lag1"] = p["mes"].map(ciudad.shift(1)) / p["celda"].nunique()
    p["ciudad_lag12"] = p["mes"].map(ciudad.shift(12)) / p["celda"].nunique()
    p["covid_fuerte"] = p["mes"].between("2020-04-01", "2020-08-01").astype(int)
    p["covid_parcial"] = p["mes"].between("2020-09-01", "2021-06-01").astype(int)
    p["tendencia"] = (p["mes"].dt.year - 2015) + (p["mes"].dt.month - 1) / 12
    p["mes_sin"] = np.sin(2 * np.pi * p["mes_num"] / 12)
    p["mes_cos"] = np.cos(2 * np.pi * p["mes_num"] / 12)
    p["localidad"] = p["localidad"].fillna("Sin dato").astype("category")
    return p


VARS = ["lag1", "lag2", "lag3", "lag12", "ma3", "ma12", "ma24", "hist_media", "muertos_12m",
        "pct12_peaton", "pct12_moto", "pct12_ciclista", "pct12_velocidad",
        "ciudad_lag1", "ciudad_lag12", "covid_fuerte", "covid_parcial", "tendencia",
        "mes_sin", "mes_cos", "mes_num", "lat", "lon", "localidad"]


# ════════════════════════════════════════════════════════════
#  MÉTRICAS
# ════════════════════════════════════════════════════════════
def devianza_poisson(y, mu):
    mu = np.clip(mu, 1e-9, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.mean(t - (y - mu)))


def metricas(df: pd.DataFrame, col: str) -> dict:
    y, mu = df[Y].to_numpy(float), df[col].to_numpy(float)
    ciudad = df.groupby("mes")[[Y, col]].sum()
    loc = df.groupby(["localidad", "mes"], observed=True)[[Y, col]].sum()
    # ¿las celdas que el modelo marca como más riesgosas concentran los siniestros?
    anual = df.groupby("celda")[[Y, col]].sum()
    top = anual[col].rank(ascending=False, method="first") <= 0.1 * len(anual)
    return {
        "MAE_celda_mes": round(float(np.mean(np.abs(y - mu))), 4),
        "RMSE_celda_mes": round(float(np.sqrt(np.mean((y - mu) ** 2))), 4),
        "devianza_poisson": round(devianza_poisson(y, mu), 4),
        "calibracion_pred_sobre_obs": round(float(mu.sum() / max(y.sum(), 1)), 3),
        "MAPE_ciudad_mes_pct": round(float(100 * np.mean(np.abs(ciudad[col] - ciudad[Y]) / ciudad[Y])), 2),
        "MAPE_localidad_mes_pct": round(float(100 * np.mean(np.abs(loc[col] - loc[Y]) / loc[Y].clip(lower=1))), 2),
        "pct_siniestros_en_top10pct_celdas": round(float(100 * anual.loc[top, Y].sum() / anual[Y].sum()), 1),
    }


# ════════════════════════════════════════════════════════════
#  MODELOS
# ════════════════════════════════════════════════════════════
def ajustar_glm(tr):
    import statsmodels.api as sm
    X = glm_X(tr)
    return sm.GLM(tr[Y], X, family=sm.families.Poisson()).fit(), list(X.columns)


def glm_X(d, columnas=None):
    import statsmodels.api as sm
    X = pd.DataFrame({
        "log_ma12": np.log(d["ma12"] + 0.05), "log_hist": np.log(d["hist_media"] + 0.05),
        "log_ma3": np.log(d["ma3"] + 0.05), "log_lag12": np.log(d["lag12"] + 0.5),
        "log_ciudad12": np.log(d["ciudad_lag12"]), "covid_fuerte": d["covid_fuerte"],
        "covid_parcial": d["covid_parcial"], "tendencia": d["tendencia"]}, index=d.index)
    X = X.join(pd.get_dummies(d["mes_num"], prefix="m", drop_first=True, dtype=float))
    X = sm.add_constant(X)
    return X if columnas is None else X.reindex(columns=columnas, fill_value=0)


def ajustar_lgbm(tr, va=None, n_arboles=None):
    import lightgbm as lgb
    params = dict(objective="poisson", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=42)
    dtr = lgb.Dataset(tr[VARS], tr[Y], categorical_feature=["localidad"])
    if va is not None:
        dva = lgb.Dataset(va[VARS], va[Y], reference=dtr)
        return lgb.train(params, dtr, 3000, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(params, dtr, n_arboles)


def main():
    os.makedirs(SALIDA, exist_ok=True)
    p = pd.read_csv(PANEL, parse_dates=["mes"])
    p = crear_variables(p)
    p = p[p["mes"] >= "2016-01-01"].copy()            # 2015 solo alimenta los rezagos
    p["ma24"] = p["ma24"].fillna(p["ma12"])
    # partición temporal relativa al último año con datos (con la base SDM: 2016–2022 · 2023 · 2024+)
    ultimo = int(p["anio"].max())
    anio_val = ultimo - 3 if ultimo >= 2025 else ultimo - 2
    tr = p[p["anio"] < anio_val]
    va = p[p["anio"] == anio_val]
    te = p[p["anio"] > anio_val]
    NOMBRE_VAL, NOMBRE_TEST = f"validacion_{anio_val}", f"prueba_{anio_val + 1}_{ultimo}"
    print(f"Entrenamiento {len(tr):,} · validación {len(va):,} · prueba {len(te):,} filas celda-mes")

    res = {}
    for nombre, d in ((NOMBRE_VAL, va), (NOMBRE_TEST, te)):
        d = d.copy()
        d["pred_ingenuo"] = d["ma12"]
        d["pred_estacional"] = d["lag12"]
        res.setdefault("ingenuo_promedio_12m", {})[nombre] = metricas(d, "pred_ingenuo")
        res.setdefault("estacional_mismo_mes", {})[nombre] = metricas(d, "pred_estacional")

    glm, cols_glm = ajustar_glm(tr)
    lgbm = ajustar_lgbm(tr, va)
    print(f"LightGBM: {lgbm.best_iteration} árboles (parada temprana en {anio_val})")
    for nombre, d in ((NOMBRE_VAL, va), (NOMBRE_TEST, te)):
        d = d.copy()
        d["pred_glm"] = glm.predict(glm_X(d, cols_glm))
        d["pred_lgbm"] = lgbm.predict(d[VARS], num_iteration=lgbm.best_iteration)
        res.setdefault("glm_poisson", {})[nombre] = metricas(d, "pred_glm")
        res.setdefault("lightgbm_poisson", {})[nombre] = metricas(d, "pred_lgbm")
        if nombre == NOMBRE_TEST:
            d[["celda", "mes", "localidad", "lat", "lon", Y, "pred_glm", "pred_lgbm", "ma12"]].to_csv(
                os.path.join(SALIDA, "predicciones_prueba.csv"), index=False, encoding="utf-8-sig")

    # coeficientes GLM (interpretables) e importancia LightGBM
    res["glm_coeficientes"] = {k: round(float(v), 4) for k, v in glm.params.items() if not k.startswith("m_")}
    res["glm_razon_dispersion"] = round(float(glm.pearson_chi2 / glm.df_resid), 3)
    imp = pd.Series(lgbm.feature_importance("gain"), index=VARS)
    res["lgbm_importancia_pct"] = (100 * imp / imp.sum()).round(1).sort_values(ascending=False).head(12).to_dict()

    # modelo final con todo 2016–2024
    final = ajustar_lgbm(p, n_arboles=lgbm.best_iteration)
    final.save_model(os.path.join(SALIDA, "lgbm_conteo.txt"))
    with open(os.path.join(SALIDA, "variables.json"), "w", encoding="utf-8") as f:
        json.dump({"objetivo": Y, "variables": VARS, "arboles": lgbm.best_iteration,
                   "localidades": list(p["localidad"].cat.categories)}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(SALIDA, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    tabla = pd.DataFrame({m: v[NOMBRE_TEST] for m, v in res.items() if isinstance(v, dict) and NOMBRE_TEST in v}).T
    print(f"\n{NOMBRE_TEST}:\n", tabla.to_string())
    print(f"\n✅ Resultados en {SALIDA}/")


if __name__ == "__main__":
    main()
