"""
=============================================================
  PREPROCESAMIENTO — Proyecto grupal ML Probabilístico, Parte 1
  Predicción de la gravedad de accidentes de tránsito · Bogotá
  CRISP-DM fase 2 (entendimiento de datos) y fase 3 (preparación)
=============================================================

Entrada:
    out/accidentes_con_sensor.csv   (salida de app.py: SDM + lluvia IDEAM por Voronoi)

Salidas (carpeta out/procesado/):
    accidentes_nivel_accidente.csv   base limpia, una fila por accidente
    X_train.csv (2015–2020), X_val.csv (2021), X_test.csv (2022)   matrices listas para modelar
    X_monitoreo.csv (2023–2024, cambio de registro)                 solo para monitoreo
    y_*.csv                          variable objetivo (0 = solo daños, 1 = con heridos, 2 = con muertos)
    reporte_preprocesamiento.json    cifras del EDA y de cada paso

Uso:
    pip install pandas numpy holidays
    python src/preprocesamiento.py
"""

import json
import os
import re
import unicodedata

import holidays
import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADA = os.path.join(RAIZ, "out", "accidentes_con_sensor.csv")
SALIDA = os.path.join(RAIZ, "out", "procesado")
os.makedirs(SALIDA, exist_ok=True)
R = {}  # reporte


def sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def canonizar(serie: pd.Series, canonicos: list[str]) -> pd.Series:
    """Repara texto dañado por la codificación (p. ej. 'Engativï' → 'Engativá')
    comparando el prefijo ASCII legible con una lista de valores canónicos."""
    mapa = {}
    for v in serie.dropna().unique():
        pref = re.match(r"[A-Za-z0-9 ,.\-]*", str(v)).group(0).strip().lower()
        cand = [c for c in canonicos if sin_tildes(c).lower().startswith(pref)] if pref else []
        mapa[v] = cand[0] if len(cand) == 1 else (v if v in canonicos else np.nan)
    return serie.map(mapa)


# ════════════════════════════════════════════════════════════
#  1. CARGA
# ════════════════════════════════════════════════════════════
df = pd.read_csv(ENTRADA, low_memory=False)
R["entrada"] = {"filas": len(df), "columnas": df.shape[1],
                "accidentes": int(df["Codigo_Accidente"].nunique())}
R["nulos_entrada_pct"] = {c: round(100 * df[c].isna().mean(), 1) for c in df.columns}

# ════════════════════════════════════════════════════════════
#  2. LIMPIEZA A NIVEL PERSONA
# ════════════════════════════════════════════════════════════
limpieza = {}

# 2.1 Duplicados
limpieza["duplicados_fila_completa"] = int(df.duplicated().sum())
limpieza["duplicados_id"] = int(df["Id"].duplicated().sum())
df = df.drop_duplicates()

# 2.2 Codificación de texto
LOCALIDADES = ["Antonio Nariño", "Barrios Unidos", "Bosa", "Candelaria", "Chapinero",
               "Ciudad Bolívar", "Engativá", "Fontibón", "Kennedy", "Los Mártires",
               "Puente Aranda", "Rafael Uribe Uribe", "San Cristóbal", "Santa Fe", "Suba",
               "Sumapaz", "Teusaquillo", "Tunjuelito", "Usaquén", "Usme"]
reparados = 0
for col, canon in {
    "Localidad": LOCALIDADES,
    "Clase_Acc": ["Choque", "Atropello", "Caida de ocupante", "Volcamiento", "Otro", "Incendio", "Autolesión"],
    "Elemento_Choque": ["Vehículo", "Objeto Fijo", "Semoviente", "Tren"],
    "Condicion": ["CONDUCTOR", "MOTOCICLISTA", "PEATÓN", "PASAJERO", "CICLISTA", "SIN INFORMACION"],
    "Gravedad_Indicador_Tradicional.1": ["Solo Daños", "Con Heridos", "Con Muertos"],
}.items():
    antes = df[col].copy()
    df[col] = canonizar(df[col].replace("0", np.nan), canon)
    reparados += int((antes.notna() & (antes != df[col])).sum())
limpieza["valores_texto_reparados"] = reparados

# 2.3 Banderas Con_*: "SI" / vacío → 1 / 0
FLAGS = [c for c in df.columns if c.startswith("Con_")]
for c in FLAGS:
    df[c] = (df[c].astype(str).str.upper() == "SI").astype(int)
limpieza["banderas_recodificadas"] = len(FLAGS)

# 2.4 Edad fuera de rango (0 o > 100 años) → faltante
mala_edad = (df["Edad"] <= 0) | (df["Edad"] > 100)
limpieza["edades_invalidas"] = int(mala_edad.sum())
limpieza["edad_max_original"] = float(df["Edad"].max())
df.loc[mala_edad, "Edad"] = np.nan

# 2.5 "SIN INFORMACION" y "0" como faltantes
df["Sexo"] = df["Sexo"].replace("SIN INFORMACION", np.nan)
df["Vehiculo_Viajaba_Clasificado"] = df["Vehiculo_Viajaba_Clasificado"].where(
    df["Vehiculo_Viajaba_Clasificado"].isin(
        ["LIVIANO", "MOTOCICLETA", "TRANSPORTE DE PASAJEROS", "TAXI", "CARGA",
         "BICICLETA", "SERVICIO ESPECIAL"]))

# 2.6 Minutos redondeados (inconsistencia de registro, se documenta)
limpieza["pct_minuto_00_o_30"] = round(100 * df["minuAcc"].isin([0, 30]).mean(), 1)

# 2.7 Variables excluidas
EXCLUIDAS = {
    "fuga_de_informacion": ["Gravedad_Indicador_Tradicional", "Gravedad_Indicador_30d",
                            "Muerte_Posterior", "Fecha_CambioGravedad"],
    "identificadores": ["Id", "Formulario", "Codigo_Vehiculo", "Ccodigo_Victima", "sensor_idx"],
    "redundantes": ["FechaAcc", "mesAcc", "Dia_Semana_Acc", "NombreEstacion", "Condicion_Especifica",
                    "Tipo_de_vehiculo", "Servicio", "Modalidad", "Tipo_SITP"],
    "casi_vacias": ["Tipo_Objeto_Fijo"],
}
R["excluidas"] = EXCLUIDAS
R["limpieza"] = limpieza

# ════════════════════════════════════════════════════════════
#  3. AGREGACIÓN A NIVEL ACCIDENTE
# ════════════════════════════════════════════════════════════
df["Fecha"] = pd.to_datetime(df["Fecha"])
g = df.groupby("Codigo_Accidente", sort=False)
primero = g[["Fecha", "AnnoAcc", "DD_Mes_Acc", "hourAcc", "minuAcc", "Localidad", "Latitud",
             "Longitud", "Clase_Acc", "Elemento_Choque", "dist_km", "ValorObservado",
             "ClaseLluvia", "Gravedad_Indicador_Tradicional.1"]].first()

cond = pd.crosstab(df["Codigo_Accidente"], df["Condicion"])
veh = pd.crosstab(df["Codigo_Accidente"], df["Vehiculo_Viajaba_Clasificado"])
acc = primero.join([
    g.size().rename("n_personas"),
    g["Edad"].agg(edad_media="mean", edad_min="min", edad_max="max"),
    (g["Sexo"].apply(lambda s: (s == "MASCULINO").mean())).rename("pct_hombres"),
    g[FLAGS].max(),
    cond.reindex(columns=["PEATÓN", "MOTOCICLISTA", "CICLISTA", "PASAJERO"], fill_value=0)
        .set_axis(["n_peatones", "n_motociclistas", "n_ciclistas", "n_pasajeros"], axis=1),
    veh.set_axis(["n_veh_" + sin_tildes(c).lower().replace(" ", "_") for c in veh.columns], axis=1),
])
acc["n_vehiculos"] = acc[[c for c in acc.columns if c.startswith("n_veh_")]].sum(axis=1)
acc = acc.rename(columns={"Gravedad_Indicador_Tradicional.1": "gravedad", "AnnoAcc": "anio",
                          "DD_Mes_Acc": "dia_mes", "hourAcc": "hora", "minuAcc": "minuto",
                          "ValorObservado": "lluvia_mm_h"})
acc = acc.reset_index()

# ════════════════════════════════════════════════════════════
#  4. NUEVAS VARIABLES
# ════════════════════════════════════════════════════════════
co_holidays = holidays.Colombia(years=range(2015, 2025))
acc["mes"] = acc["Fecha"].dt.month
acc["dia_semana"] = acc["Fecha"].dt.dayofweek.map(
    {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"})
acc["fin_de_semana"] = acc["Fecha"].dt.dayofweek.isin([5, 6]).astype(int)
acc["festivo"] = acc["Fecha"].dt.date.map(lambda d: d in co_holidays).astype(int)
acc["hora_sin"] = np.sin(2 * np.pi * acc["hora"] / 24)
acc["hora_cos"] = np.cos(2 * np.pi * acc["hora"] / 24)
acc["mes_sin"] = np.sin(2 * np.pi * acc["mes"] / 12)
acc["mes_cos"] = np.cos(2 * np.pi * acc["mes"] / 12)
acc["franja"] = pd.cut(acc["hora"], [-1, 5, 11, 17, 23],
                       labels=["Madrugada", "Mañana", "Tarde", "Noche"]).astype(str)
acc["oscuridad"] = ((acc["hora"] >= 18) | (acc["hora"] < 6)).astype(int)
acc["llueve"] = (acc["lluvia_mm_h"] > 0).astype(int)
# distancia (km) a la Plaza de Bolívar como proxy de centralidad
acc["dist_centro_km"] = np.hypot((acc["Latitud"] - 4.5981) * 111.32,
                                 (acc["Longitud"] + 74.0760) * 111.32 * np.cos(np.radians(4.6)))
CLAVE = ["Con_Embriaguez", "Con_Velocidad", "Con_Moto", "Con_Peaton", "Con_Bicicleta",
         "Con_Huecos", "Con_Menores", "Con_Persona_Mayor", "Con_Carga"]
acc["n_factores_riesgo"] = acc[CLAVE].sum(axis=1)
acc["actor_vulnerable"] = ((acc["n_peatones"] + acc["n_ciclistas"] + acc["n_motociclistas"]) > 0).astype(int)

# categorías raras (< 100 accidentes) → "Otra"
for c in ["Localidad", "Clase_Acc", "Elemento_Choque"]:
    acc[c] = acc[c].fillna("Sin dato")
    vc = acc[c].value_counts()
    acc[c] = acc[c].where(~acc[c].isin(vc[vc < 100].index), "Otra")

# variable objetivo
Y_MAP = {"Solo Daños": 0, "Con Heridos": 1, "Con Muertos": 2}
acc["y"] = acc["gravedad"].map(Y_MAP)
assert acc["y"].notna().all()
acc.to_csv(os.path.join(SALIDA, "accidentes_nivel_accidente.csv"), index=False, encoding="utf-8-sig")

NUM = ["hora_sin", "hora_cos", "mes_sin", "mes_cos", "dia_mes", "Latitud", "Longitud",
       "dist_centro_km", "dist_km", "lluvia_mm_h", "n_personas", "n_vehiculos", "n_peatones",
       "n_motociclistas", "n_ciclistas", "n_pasajeros", "edad_media", "edad_min", "edad_max",
       "pct_hombres", "n_factores_riesgo"] + \
      [c for c in acc.columns if c.startswith("n_veh_")]
BIN = FLAGS + ["fin_de_semana", "festivo", "oscuridad", "llueve", "actor_vulnerable"]
CAT = ["Localidad", "Clase_Acc", "Elemento_Choque", "dia_semana", "franja", "ClaseLluvia"]
R["variables_modelo"] = {"numericas": NUM, "binarias": BIN, "categoricas": CAT,
                         "n_numericas": len(NUM), "n_binarias": len(BIN), "n_categoricas": len(CAT)}

# ════════════════════════════════════════════════════════════
#  5. EDA SOBRE LA BASE A NIVEL ACCIDENTE
# ════════════════════════════════════════════════════════════
eda = {"accidentes": len(acc)}
eda["objetivo"] = acc["gravedad"].value_counts().to_dict()
DESC = ["hora", "Latitud", "Longitud", "dist_centro_km", "dist_km", "lluvia_mm_h", "n_personas",
        "n_vehiculos", "edad_media", "n_factores_riesgo"]
d = {}
for c in DESC:
    s = acc[c]
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    d[c] = {"media": round(s.mean(), 2), "mediana": round(s.median(), 2), "std": round(s.std(), 2),
            "min": round(s.min(), 2), "max": round(s.max(), 2),
            "nulos_pct": round(100 * s.isna().mean(), 2),
            "atipicos_iqr": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
            "asimetria": round(s.skew(), 2)}
eda["numericas"] = d
eda["nulos_nivel_accidente_pct"] = {c: round(100 * acc[c].isna().mean(), 2)
                                    for c in NUM + CAT if acc[c].isna().any()}
CORR = ["hora", "Latitud", "Longitud", "dist_centro_km", "n_personas", "n_vehiculos",
        "n_peatones", "n_motociclistas", "edad_media", "n_factores_riesgo", "y"]
eda["correlacion_spearman"] = acc[CORR].corr(method="spearman").round(2).to_dict()
fatal = (acc["y"] == 2).astype(int)
eda["corr_con_fatal"] = acc[NUM + BIN].corrwith(fatal).round(3).sort_values(key=abs, ascending=False).head(12).to_dict()


def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    n = ct.values.sum()
    exp = np.outer(ct.sum(axis=1), ct.sum(axis=0)) / n
    chi2 = ((ct.values - exp) ** 2 / exp).sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


eda["cramers_v"] = {c: round(cramers_v(acc[c], acc["y"]), 3) for c in CAT}
eda["fatal_por_franja"] = (fatal.groupby(acc["franja"]).mean() * 100).round(2).to_dict()
eda["fatal_por_festivo"] = (fatal.groupby(acc["festivo"]).mean() * 100).round(2).to_dict()
eda["fatal_por_n_factores"] = (fatal.groupby(acc["n_factores_riesgo"].clip(upper=3)).mean() * 100).round(2).to_dict()
eda["accidentes_por_n_factores"] = acc["n_factores_riesgo"].clip(upper=3).value_counts().sort_index().to_dict()
eda["fatal_actor_vulnerable"] = (fatal.groupby(acc["actor_vulnerable"]).mean() * 100).round(2).to_dict()
eda["personas_hist"] = acc["n_personas"].clip(upper=6).value_counts().sort_index().to_dict()
R["eda"] = eda

# ════════════════════════════════════════════════════════════
#  6. PARTICIÓN TEMPORAL Y TRANSFORMACIÓN
# ════════════════════════════════════════════════════════════
# Desde 2023 casi no se registran accidentes de "solo daños" (7 % vs ~60 %): cambio en el registro.
# Por eso 2023–2024 NO se usan como prueba; quedan aparte como conjunto de monitoreo.
R["gravedad_por_anio"] = pd.crosstab(acc["anio"], acc["gravedad"]).to_dict()
split = np.select([acc["anio"] <= 2020, acc["anio"] == 2021, acc["anio"] == 2022],
                  ["train", "val", "test"], default="monitoreo")
tr = acc[split == "train"]
# parámetros ajustados SOLO con entrenamiento (evita fuga de información)
medianas = tr[NUM].median()
medias = tr[NUM].fillna(medianas).mean()
desv = tr[NUM].fillna(medianas).std().replace(0, 1)
categorias = {c: sorted(tr[c].fillna("Sin dato").unique()) for c in CAT}


def transformar(parte: pd.DataFrame) -> pd.DataFrame:
    num = (parte[NUM].fillna(medianas) - medias) / desv                 # imputación + estandarización
    cat = [pd.get_dummies(pd.Categorical(parte[c].fillna("Sin dato"), categories=categorias[c]),
                          prefix=c, dtype=int) for c in CAT]            # one-hot con categorías de train
    return pd.concat([num.reset_index(drop=True), parte[BIN].reset_index(drop=True)] + cat, axis=1)


R["particion"] = {}
for s in ["train", "val", "test", "monitoreo"]:
    m = split == s
    X = transformar(acc.loc[m])
    X.to_csv(os.path.join(SALIDA, f"X_{s}.csv"), index=False)
    acc.loc[m, ["y"]].to_csv(os.path.join(SALIDA, f"y_{s}.csv"), index=False)
    R["particion"][s] = {"filas": int(m.sum()),
                         "clases_pct": (acc.loc[m, "y"].value_counts(normalize=True).sort_index() * 100).round(2).to_dict()}
cols = X.columns
R["matriz_final"] = {"columnas": len(cols), "nulos_restantes": int(X.isna().sum().sum())}
pesos = len(acc[split == "train"]) / (3 * acc.loc[split == "train", "y"].value_counts().sort_index())
R["pesos_clase_train"] = pesos.round(2).to_dict()

with open(os.path.join(SALIDA, "reporte_preprocesamiento.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
print(f"✅ {len(acc):,} accidentes · {len(cols)} columnas modelables · reporte en {SALIDA}/")
