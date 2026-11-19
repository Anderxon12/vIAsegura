"""
=============================================================
  VORONOI CLIMÁTICO — Asignación de sensores a accidentes
  Bogotá D.C. — IDEAM / SDM
=============================================================

Requisitos:
    pip install dash plotly pandas numpy scipy

Uso:
    1. Pon Precipitación.csv y Base total.csv en la carpeta data/.
    2. python src/app.py   (exporta out/accidentes_con_sensor.csv)
    3. http://127.0.0.1:8050

─── Clasificación de lluvia (ValorObservado mm/h) ──────────
  Soleado      →  = 0  (sin registro o valor cero)
  Llovizna     →  > 0  y  <= 0.1
  Ligera       →  > 0.1 y  <  2
  Moderada     →  >= 2  y  <= 15
  Fuerte       →  > 15  y  <= 30
  Muy fuerte   →  > 30  y  <= 60
  Torrencial   →  > 60
=============================================================
"""

import os
import pandas as pd
import numpy as np
from scipy.spatial import Voronoi, cKDTree
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import warnings
warnings.filterwarnings("ignore")

# Rutas relativas a la raíz del proyecto (funciona desde cualquier carpeta)
RAIZ            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_CLIMA      = os.path.join(RAIZ, "data", "Precipitación.csv")
RUTA_ACCIDENTES = os.path.join(RAIZ, "data", "Base total.csv")
OUT_DIR         = os.path.join(RAIZ, "out")
RUTA_SALIDA     = os.path.join(OUT_DIR, "accidentes_con_sensor.csv")


# ════════════════════════════════════════════════════════════
#  CLASIFICACIÓN DE LLUVIA
# ════════════════════════════════════════════════════════════

LLUVIA_ORDEN = ["Soleado", "Llovizna", "Ligera", "Moderada",
                "Fuerte", "Muy fuerte", "Torrencial"]

LLUVIA_COLOR_HEX = {
    "Soleado":    "#FFD166",
    "Llovizna":   "#A8DADC",
    "Ligera":     "#70B8FF",
    "Moderada":   "#3A86FF",
    "Fuerte":     "#1D3FA5",
    "Muy fuerte": "#9B2FE8",
    "Torrencial": "#FF0066",
}

LLUVIA_COLOR_RGB = {
    k: [int(v[1:3], 16), int(v[3:5], 16), int(v[5:7], 16)]
    for k, v in LLUVIA_COLOR_HEX.items()
}

MATRIZ_COLOR = {
    ("Llovizna",   "Baja"):  "#CCE5FF",
    ("Llovizna",   "Media"): "#66B2FF",
    ("Llovizna",   "Alta"):  "#0066CC",
    ("Ligera",     "Baja"):  "#C8F7C5",
    ("Ligera",     "Media"): "#5DBD58",
    ("Ligera",     "Alta"):  "#1A6B15",
    ("Moderada",   "Baja"):  "#FFF3CD",
    ("Moderada",   "Media"): "#FFB347",
    ("Moderada",   "Alta"):  "#CC6600",
    ("Fuerte",     "Baja"):  "#FADADD",
    ("Fuerte",     "Media"): "#FF6B6B",
    ("Fuerte",     "Alta"):  "#CC0000",
    ("Muy fuerte", "Baja"):  "#E8D5F5",
    ("Muy fuerte", "Media"): "#C06BE8",
    ("Muy fuerte", "Alta"):  "#6A0DAD",
    ("Torrencial", "Baja"):  "#FFD6E7",
    ("Torrencial", "Media"): "#FF3399",
    ("Torrencial", "Alta"):  "#990033",
    ("Soleado",    "Baja"):  "#FFF9C4",
    ("Soleado",    "Media"): "#FFD700",
    ("Soleado",    "Alta"):  "#B8860B",
}


def calcular_nivel_acc(acc_df):
    conteo = acc_df.groupby("sensor_id").size()
    q33    = conteo.quantile(0.33)
    q66    = conteo.quantile(0.66)
    def nivel(n):
        if n <= q33:  return "Baja"
        if n <= q66:  return "Media"
        return "Alta"
    return conteo.map(nivel).to_dict()


def clasificar_lluvia(val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "Soleado"
    if v == 0 or pd.isna(v):
        return "Soleado"
    elif v <= 0.1:
        return "Llovizna"
    elif v < 2:
        return "Ligera"
    elif v <= 15:
        return "Moderada"
    elif v <= 30:
        return "Fuerte"
    elif v <= 60:
        return "Muy fuerte"
    else:
        return "Torrencial"


# ════════════════════════════════════════════════════════════
#  CONSTANTES GLOBALES
# ════════════════════════════════════════════════════════════

MESES_NOMBRE = {
    1: "Enero",     2: "Febrero",   3: "Marzo",      4: "Abril",
    5: "Mayo",      6: "Junio",     7: "Julio",       8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

MAX_TABLA = 2_000


# ════════════════════════════════════════════════════════════
#  ESTILOS
# ════════════════════════════════════════════════════════════

FONT   = "'IBM Plex Mono','Fira Code',monospace"
BG     = "#0D1117"; BG2 = "#131720"
BORDER = "#1e2745"; MUTED = "#6272a4"; TEXT = "#cdd6f4"
CYAN   = "#00E5FF"; PURPLE = "#BD93F9"; PINK = "#FF79C6"
GREEN  = "#50FA7B"; ORANGE = "#FFB86C"

DD_STYLE  = {"fontSize": "0.82rem", "backgroundColor": BG2, "color": TEXT}
SEK_STYLE = {
    "display": "flex", "gap": "28px", "padding": "8px 32px 14px",
    "alignItems": "flex-start", "flexWrap": "wrap",
    "borderBottom": f"1px solid {BORDER}"
}

TAB_STYLE = {
    "backgroundColor": BG2,
    "color": MUTED,
    "borderBottom": f"1px solid {BORDER}",
    "fontFamily": FONT,
    "fontSize": "0.80rem",
    "fontWeight": "600",
    "letterSpacing": "0.08em",
    "textTransform": "uppercase",
    "padding": "10px 20px",
}
TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "backgroundColor": BG,
    "color": CYAN,
    "borderTop": f"2px solid {CYAN}",
    "borderBottom": f"1px solid {BG}",
}


# ════════════════════════════════════════════════════════════
#  1. CARGA Y LIMPIEZA
# ════════════════════════════════════════════════════════════

def _fix_decimal(series: pd.Series) -> pd.Series:
    return (series.astype(str)
                  .str.replace(",", ".", regex=False)
                  .str.strip()
                  .pipe(pd.to_numeric, errors="coerce"))


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def cargar_clima(path=RUTA_CLIMA):
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    df.columns = df.columns.str.strip()
    df["Latitud"]        = _fix_decimal(df["Latitud"])
    df["Longitud"]       = _fix_decimal(df["Longitud"])
    df["ValorObservado"] = _fix_decimal(df["ValorObservado"])
    df["Fecha"]          = _parse_date(df["Fecha"])
    df["NombreEstacion"] = df["NombreEstacion"].astype(str).str.strip()
    df["_hora_int"]      = (df["Hora"].astype(str)
                                       .str.split(":").str[0]
                                       .pipe(pd.to_numeric, errors="coerce"))
    df = df.dropna(subset=["Latitud", "Longitud", "ValorObservado"])

    catalogo = (df.sort_values("Fecha", ascending=False)
                  .drop_duplicates(subset=["NombreEstacion", "Latitud", "Longitud"])
                  .reset_index(drop=True)
                  .copy())
    catalogo["sensor_id"] = [f"S{i:02d}" for i in range(len(catalogo))]

    sid_map = catalogo.set_index("NombreEstacion")["sensor_id"].to_dict()
    df["sensor_id"] = df["NombreEstacion"].map(sid_map)

    return catalogo, df


def cargar_accidentes(path=RUTA_ACCIDENTES) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    df.columns = df.columns.str.strip()
    df["Latitud"]  = _fix_decimal(df["Latitud"])
    df["Longitud"] = _fix_decimal(df["Longitud"])
    fecha_col = "FechaAcc" if "FechaAcc" in df.columns else "Fecha_Acc"
    df["Fecha"] = _parse_date(df[fecha_col])
    df["hourAcc"] = pd.to_numeric(df["hourAcc"], errors="coerce").fillna(0).astype(int)
    df["minuAcc"] = pd.to_numeric(df["minuAcc"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["Latitud", "Longitud"])
    df = df[(df["Latitud"] != 0) & (df["Longitud"] != 0)]
    df = df.reset_index(drop=True)
    return df


def cargar_datos():
    if os.path.exists(RUTA_ACCIDENTES) and os.path.exists(RUTA_CLIMA):
        catalogo, clima_full = cargar_clima(RUTA_CLIMA)
        accidentes = cargar_accidentes(RUTA_ACCIDENTES)
        print(f"✅  Cargados: {len(catalogo)} sensores | {len(clima_full):,} lecturas | {len(accidentes):,} accidentes")
    else:
        print("⚠️  CSVs no encontrados → datos de ejemplo (Bogotá, 608 000 accidentes)")
        catalogo, clima_full, accidentes = _datos_ejemplo()
    return catalogo, clima_full, accidentes


def _datos_ejemplo():
    np.random.seed(7)
    n_s = 18
    nombres = [
        "IDEAM PUENTE ARANDA", "IDEAM VENECIA", "IDEAM LA PICOTA",
        "IDEAM SUBA", "IDEAM USAQUEN", "IDEAM CHAPINERO",
        "IDEAM KENNEDY", "IDEAM BOSA", "IDEAM FONTIBON",
        "IDEAM USME", "IDEAM SAN CRISTOBAL", "IDEAM RAFAEL URIBE",
        "IDEAM ANTONIO NARIÑO", "IDEAM BARRIOS UNIDOS", "IDEAM TEUSAQUILLO",
        "IDEAM SANTAFE", "IDEAM CANDELARIA", "FUNDACION ANA RESTREPO",
    ]
    vals = np.concatenate([
        np.zeros(5),
        np.random.uniform(0.01, 0.1, 2),
        np.random.uniform(0.11, 1.9, 2),
        np.random.uniform(2, 15, 4),
        np.random.uniform(15.1, 30, 2),
        np.random.uniform(30.1, 60, 2),
        np.random.uniform(60.1, 90, 1),
    ])
    np.random.shuffle(vals)
    clima = pd.DataFrame({
        "sensor_id":      [f"S{i:02d}" for i in range(n_s)],
        "NombreEstacion": nombres,
        "Latitud":        np.random.uniform(4.48, 4.82, n_s),
        "Longitud":       np.random.uniform(-74.20, -74.00, n_s),
        "Fecha":          pd.to_datetime("2024-12-31"),
        "_hora_int":      np.random.randint(0, 24, n_s),
        "ValorObservado": np.round(vals[:n_s], 2),
    })
    n_a = 608_000
    fechas_base = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        np.random.randint(0, 365, n_a), unit="D")
    vehiculos = np.random.choice(
        ["Automovil", "Bus", "Moto", "Bicicleta", "Camion", "Taxi", "Patineta"],
        n_a, p=[0.30, 0.15, 0.28, 0.10, 0.07, 0.08, 0.02]
    )
    si_no = lambda p: np.where(np.random.random(n_a) < p, "SI", "")
    accidentes = pd.DataFrame({
        "Id":            range(n_a),
        "Codigo_Accidente": [f"A{i:06d}" for i in range(n_a)],
        "Latitud":   np.random.uniform(4.48, 4.82, n_a),
        "Longitud":  np.random.uniform(-74.20, -74.00, n_a),
        "Fecha":     fechas_base,
        "AnnoAcc":   fechas_base.year,
        "mesAcc":    fechas_base.strftime("%B"),
        "DD_Mes_Acc": fechas_base.day,
        "hourAcc":   np.random.randint(0, 24, n_a),
        "minuAcc":   np.random.choice([0, 15, 30, 45], n_a),
        "Localidad": np.random.choice([
            "KENNEDY", "SUBA", "ENGATIVA", "BOSA", "USME",
            "CHAPINERO", "FONTIBON", "RAFAEL URIBE", "TUNJUELITO", "CIUDAD BOLIVAR"
        ], n_a),
        "Clase_Acc": np.random.choice(
            ["Choque", "Atropello", "Caida de ocupante", "Volcamiento", "Otro"],
            n_a, p=[0.45, 0.25, 0.15, 0.08, 0.07]
        ),
        "Gravedad_Indicador_Tradicional": np.random.choice(
            ["ILESO", "HERIDO", "MUERTO"], n_a, p=[0.55, 0.40, 0.05]
        ),
        "Condicion": np.random.choice(
            ["CONDUCTOR", "PASAJERO", "PEATON", "CICLISTA"], n_a, p=[0.50, 0.20, 0.20, 0.10]
        ),
        "Tipo_de_vehiculo": vehiculos,
        "Con_Embriaguez": si_no(0.08),
        "Con_Velocidad":  si_no(0.12),
        "Con_Moto":       si_no(0.28),
        "Con_Peaton":     si_no(0.20),
        "Con_Bicicleta":  si_no(0.10),
        "Con_Huecos":     si_no(0.05),
        "Con_Menores":    si_no(0.04),
        "Con_Carga":      si_no(0.06),
    })
    return clima, pd.DataFrame(), accidentes


# ════════════════════════════════════════════════════════════
#  2. ASIGNACIÓN VORONOI
# ════════════════════════════════════════════════════════════

def asignar_sensor_voronoi(catalogo_df, acc_df, clima_full_df=None):
    sens = catalogo_df[["Latitud", "Longitud"]].values
    acc  = acc_df[["Latitud", "Longitud"]].values
    tree = cKDTree(sens)
    dists, idxs = tree.query(acc)

    res = acc_df.copy()
    res["sensor_idx"]     = idxs
    res["sensor_id"]      = catalogo_df.iloc[idxs]["sensor_id"].values
    res["NombreEstacion"] = catalogo_df.iloc[idxs]["NombreEstacion"].values
    res["dist_km"]        = np.round(dists * 111.32, 3)

    if clima_full_df is not None and len(clima_full_df) > 0:
        print("  → Construyendo lookup de precipitación…")

        cf = clima_full_df.copy()
        cf["_date_key"] = cf["Fecha"].dt.normalize()
        cf["_hora_int"] = cf["_hora_int"].fillna(-1).astype(int)

        lk_exact = (cf.groupby(["sensor_id", "_date_key", "_hora_int"])["ValorObservado"]
                      .mean().reset_index()
                      .rename(columns={"_hora_int": "_hora_key"}))

        lk_day = (cf.groupby(["sensor_id", "_date_key"])["ValorObservado"]
                    .mean().reset_index()
                    .rename(columns={"ValorObservado": "_vo_day"}))

        rows = []
        for delta in (-1, 0, 1):
            tmp = lk_exact.copy()
            tmp["_hora_key"] = tmp["_hora_key"] + delta
            tmp["_vo_near"]  = tmp["ValorObservado"]
            rows.append(tmp[["sensor_id", "_date_key", "_hora_key", "_vo_near"]])
        lk_near = (pd.concat(rows)
                     .groupby(["sensor_id", "_date_key", "_hora_key"])["_vo_near"]
                     .mean().reset_index())

        res["_date_key"] = res["Fecha"].dt.normalize()
        res["_hora_key"] = res["hourAcc"].astype(int)

        res = res.merge(lk_exact, on=["sensor_id", "_date_key", "_hora_key"], how="left")

        mask_nan = res["ValorObservado"].isna()
        if mask_nan.any():
            fill_near = res.loc[mask_nan, ["sensor_id", "_date_key", "_hora_key"]].merge(
                lk_near, on=["sensor_id", "_date_key", "_hora_key"], how="left")
            res.loc[mask_nan, "ValorObservado"] = fill_near["_vo_near"].values

        mask_nan2 = res["ValorObservado"].isna()
        if mask_nan2.any():
            fill_day = res.loc[mask_nan2, ["sensor_id", "_date_key"]].merge(
                lk_day, on=["sensor_id", "_date_key"], how="left")
            res.loc[mask_nan2, "ValorObservado"] = fill_day["_vo_day"].values

        res["ValorObservado"] = res["ValorObservado"].fillna(0.0)
        res = res.drop(columns=["_date_key", "_hora_key"])

        n_no_soleado = (res["ValorObservado"] > 0).sum()
        print(f"  → Cruce OK: {n_no_soleado:,} accidentes con lluvia > 0 "
              f"({round(100*n_no_soleado/len(res),1)}%)")
    else:
        res["ValorObservado"] = catalogo_df.iloc[idxs]["ValorObservado"].values

    res["ClaseLluvia"] = res["ValorObservado"].apply(clasificar_lluvia)
    return res


# ════════════════════════════════════════════════════════════
#  3. POLÍGONOS VORONOI
# ════════════════════════════════════════════════════════════

def voronoi_poligonos(puntos, margen=0.12):
    lat_min = puntos[:, 0].min() - margen
    lat_max = puntos[:, 0].max() + margen
    lon_min = puntos[:, 1].min() - margen
    lon_max = puntos[:, 1].max() + margen
    fantasmas = np.array([
        [lat_min - 10, (lon_min + lon_max) / 2],
        [lat_max + 10, (lon_min + lon_max) / 2],
        [(lat_min + lat_max) / 2, lon_min - 10],
        [(lat_min + lat_max) / 2, lon_max + 10],
    ])
    vor = Voronoi(np.vstack([puntos, fantasmas]))
    polys = []
    for i in range(len(puntos)):
        reg = vor.regions[vor.point_region[i]]
        if -1 in reg or len(reg) == 0:
            polys.append(None)
            continue
        v = vor.vertices[reg].copy()
        v[:, 0] = np.clip(v[:, 0], lat_min, lat_max)
        v[:, 1] = np.clip(v[:, 1], lon_min, lon_max)
        polys.append(v.tolist())
    return polys


# ════════════════════════════════════════════════════════════
#  4. MAPA
# ════════════════════════════════════════════════════════════

MAX_PUNTOS_MAPA = 40_000

def _rgb_a_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def construir_mapa(acc_df=None, sens_df=None, mostrar_voronoi=True, colorear_por="ClaseLluvia"):
    fig = go.Figure()

    if mostrar_voronoi and sens_df is not None and len(sens_df) >= 3:
        s = sens_df.copy()
        s["Latitud"]  = pd.to_numeric(s["Latitud"],  errors="coerce")
        s["Longitud"] = pd.to_numeric(s["Longitud"], errors="coerce")
        s = s.dropna(subset=["Latitud", "Longitud"])
        pts  = s[["Latitud", "Longitud"]].values
        pols = voronoi_poligonos(pts)
        cats = s["ValorObservado"].apply(clasificar_lluvia).values

        _niv_dict_vor = {}
        if colorear_por == "Matriz" and acc_df is not None and len(acc_df) > 0:
            _niv_dict_vor = calcular_nivel_acc(acc_df)
        _svo      = s["ValorObservado"].fillna(0)
        _svo_min  = _svo.min()
        _svo_max  = _svo.max()
        _svo_rng  = (_svo_max - _svo_min) if _svo_max > _svo_min else 1.0

        for i, poly in enumerate(pols):
            if poly is None:
                continue
            if colorear_por == "ValorObservado":
                _nv = (_svo.iloc[i] - _svo_min) / _svo_rng
                r, g, b = int(255*_nv), int(80*(1-_nv)), int(255*(1-_nv))
            elif colorear_por == "Matriz" and _niv_dict_vor:
                _sid   = s["sensor_id"].iloc[i]
                _nivel = _niv_dict_vor.get(_sid, "Baja")
                hex_c  = MATRIZ_COLOR.get((cats[i], _nivel), "#8888aa")
                r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
            else:
                hex_c = LLUVIA_COLOR_HEX.get(cats[i], "#8888aa")
                r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
            lats_p = [v[0] for v in poly] + [poly[0][0]]
            lons_p = [v[1] for v in poly] + [poly[0][1]]
            fig.add_trace(go.Scattermap(
                lat=lats_p, lon=lons_p,
                mode="lines", fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.20)",
                line=dict(color=f"rgba({r},{g},{b},0.55)", width=1),
                hoverinfo="skip", showlegend=False,
            ))

    GRAV_INFO = [
        ("ILESO",  "#50FA7B"),
        ("HERIDO", "#FFB86C"),
        ("MUERTO", "#FF5555"),
    ]

    if acc_df is not None and len(acc_df) > 0:
        df = acc_df.sample(min(MAX_PUNTOS_MAPA, len(acc_df)), random_state=42).copy()
        df["Latitud"]  = pd.to_numeric(df["Latitud"],  errors="coerce")
        df["Longitud"] = pd.to_numeric(df["Longitud"], errors="coerce")
        df = df.dropna(subset=["Latitud", "Longitud"])

        if colorear_por == "ClaseLluvia":
            df["_color"] = df["ClaseLluvia"].map(LLUVIA_COLOR_HEX).fillna("#8888aa")
        elif colorear_por == "Matriz":
            nivel_acc_dict = calcular_nivel_acc(acc_df)
            df["NivelAcc"] = df["sensor_id"].map(nivel_acc_dict).fillna("Baja") \
                             if "sensor_id" in df.columns else "Baja"
            df["_color"] = df.apply(lambda row: MATRIZ_COLOR.get(
                (row.get("ClaseLluvia", "Soleado"), row.get("NivelAcc", "Baja")), "#888888"
            ), axis=1)
        elif colorear_por == "ValorObservado":
            _vals = df["ValorObservado"].fillna(0)
            _vmin, _vmax = _vals.min(), _vals.max()
            _vrng = (_vmax - _vmin) if _vmax > _vmin else 1.0
            norm = (_vals - _vmin) / _vrng
            df["_color"] = norm.apply(
                lambda v: _rgb_a_hex([int(255*v), int(80*(1-v)), int(255*(1-v))]))
        elif colorear_por == "dist_km":
            _vals = df["dist_km"].fillna(0)
            _vmin, _vmax = _vals.min(), _vals.max()
            _vrng = (_vmax - _vmin) if _vmax > _vmin else 1.0
            norm = (_vals - _vmin) / _vrng
            df["_color"] = norm.apply(
                lambda v: _rgb_a_hex([int(255*v), int(200*(1-v)), int(50*(1-v))]))
        elif colorear_por == "hourAcc":
            df["_color"] = df["hourAcc"].fillna(0).apply(
                lambda h: "#1e50c8" if h < 6 else
                          "#ffdc32" if h < 12 else
                          "#ff8c1e" if h < 18 else "#8c32c8")
        else:
            df["_color"] = "#3a86ff"

        df["_grav"] = df["Gravedad_Indicador_Tradicional"].fillna("ILESO") \
                      if "Gravedad_Indicador_Tradicional" in df.columns else "ILESO"
        df["_loc"]  = df["Localidad"].fillna("—")  if "Localidad"  in df.columns else "—"
        df["_lluv"] = df["ClaseLluvia"].fillna("—")
        df["_hora"] = df["hourAcc"].fillna(0).astype(int) if "hourAcc" in df.columns else 0

        for grav_val, grav_color in GRAV_INFO:
            sub = df[df["_grav"] == grav_val]
            if len(sub) == 0:
                continue
            fig.add_trace(go.Scattermap(
                lat=sub["Latitud"].tolist(),
                lon=sub["Longitud"].tolist(),
                mode="markers",
                marker=dict(size=6, color=sub["_color"].tolist(), opacity=0.8),
                customdata=list(zip(sub["_loc"], sub["_grav"], sub["_lluv"], sub["_hora"])),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Gravedad: <b>%{customdata[1]}</b><br>"
                    "Lluvia: %{customdata[2]}<br>"
                    "Hora: %{customdata[3]}:00 h"
                    "<extra></extra>"
                ),
                name=grav_val.title(),
                showlegend=True,
                legendrank=10 + ["ILESO","HERIDO","MUERTO"].index(grav_val),
            ))
    else:
        fig.add_trace(go.Scattermap(lat=[], lon=[]))

    if sens_df is not None and len(sens_df) > 0:
        s = sens_df.copy()
        s["Latitud"]  = pd.to_numeric(s["Latitud"],  errors="coerce")
        s["Longitud"] = pd.to_numeric(s["Longitud"], errors="coerce")
        s = s.dropna(subset=["Latitud", "Longitud"])
        fig.add_trace(go.Scattermap(
            lat=s["Latitud"].tolist(),
            lon=s["Longitud"].tolist(),
            mode="markers",
            marker=dict(size=16, color="red", opacity=1.0),
            text=s["NombreEstacion"].fillna("—").tolist(),
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="Sensores",
            showlegend=True,
            legendrank=13,
        ))

    if colorear_por == "ClaseLluvia":
        leyenda = [(cat, LLUVIA_COLOR_HEX[cat]) for cat in LLUVIA_ORDEN]
    elif colorear_por == "Matriz":
        leyenda = [
            ("Soleado · Baja",    MATRIZ_COLOR.get(("Soleado",    "Baja"),  "#888")),
            ("Moderada · Media",  MATRIZ_COLOR.get(("Moderada",   "Media"), "#888")),
            ("Fuerte · Alta",     MATRIZ_COLOR.get(("Fuerte",     "Alta"),  "#888")),
            ("Torrencial · Alta", MATRIZ_COLOR.get(("Torrencial", "Alta"),  "#888")),
        ]
    elif colorear_por == "ValorObservado":
        if acc_df is not None and len(acc_df) > 0:
            _lv = acc_df["ValorObservado"].fillna(0)
            _l0, _l3 = _lv.min(), _lv.max()
            _l1, _l2 = _l0 + (_l3-_l0)/3, _l0 + 2*(_l3-_l0)/3
        else:
            _l0, _l1, _l2, _l3 = 0, 20, 40, 60
        leyenda = [
            (f"{_l0:.1f} mm/h", _rgb_a_hex([0, 80, 255])),
            (f"{_l1:.1f} mm/h", _rgb_a_hex([85, 53, 210])),
            (f"{_l2:.1f} mm/h", _rgb_a_hex([170, 26, 165])),
            (f"{_l3:.1f}+ mm/h", _rgb_a_hex([255, 0, 0])),
        ]
    elif colorear_por == "dist_km":
        if acc_df is not None and len(acc_df) > 0:
            _ld = acc_df["dist_km"].fillna(0)
            _d0, _d3 = _ld.min(), _ld.max()
            _d1, _d2 = _d0 + (_d3-_d0)/3, _d0 + 2*(_d3-_d0)/3
        else:
            _d0, _d1, _d2, _d3 = 0, 3, 7, 10
        leyenda = [
            (f"{_d0:.1f} km", _rgb_a_hex([0, 200, 50])),
            (f"{_d1:.1f} km", _rgb_a_hex([77, 140, 35])),
            (f"{_d2:.1f} km", _rgb_a_hex([179, 57, 13])),
            (f"{_d3:.1f}+ km", _rgb_a_hex([255, 0, 0])),
        ]
    elif colorear_por == "hourAcc":
        leyenda = [("00-05 h madrugada","#1e50c8"), ("06-11 h mañana","#ffdc32"),
                   ("12-17 h tarde","#ff8c1e"), ("18-23 h noche","#8c32c8")]
    else:
        leyenda = []

    for i, (nombre, color) in enumerate(leyenda):
        fig.add_trace(go.Scattermap(
            lat=[], lon=[], mode="markers",
            marker=dict(size=10, color=color),
            name=nombre,
            legendgroup="color",
            legendgrouptitle_text="Color",
            showlegend=True,
            legendrank=100 + i,
        ))

    fig.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=4.6534, lon=-74.0836),
            zoom=11,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
        paper_bgcolor=BG,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(13,17,23,0.85)",
            font=dict(color=TEXT, size=11, family=FONT),
            bordercolor=BORDER, borderwidth=1,
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
        hoverlabel=dict(bgcolor="#0D1117", font_color=TEXT, font_family=FONT),
    )
    return fig


# ════════════════════════════════════════════════════════════
#  5. INICIALIZACIÓN
# ════════════════════════════════════════════════════════════

clima_df, clima_full_df, accidentes_df = cargar_datos()

print("⏳  Asignando sensores Voronoi + cruce temporal…")
acc_asignado = asignar_sensor_voronoi(clima_df, accidentes_df, clima_full_df)
os.makedirs(OUT_DIR, exist_ok=True)
acc_asignado.to_csv(RUTA_SALIDA, index=False, encoding="utf-8-sig")
print(f"💾  Exportado → out/accidentes_con_sensor.csv({len(acc_asignado):,} filas)")

gravedades_unicas = sorted(acc_asignado["Gravedad_Indicador_Tradicional"].dropna().unique())
clases_unicas     = sorted(acc_asignado["Clase_Acc"].dropna().unique()) if "Clase_Acc" in acc_asignado.columns else []
condicion_unicas  = sorted(acc_asignado["Condicion"].dropna().unique()) if "Condicion" in acc_asignado.columns else []
vehiculo_unicas   = sorted(acc_asignado["Tipo_de_vehiculo"].dropna().unique()) if "Tipo_de_vehiculo" in acc_asignado.columns else []
lluvia_presentes  = [c for c in LLUVIA_ORDEN if c in acc_asignado["ClaseLluvia"].unique()]

annos = sorted(acc_asignado["Fecha"].dt.year.dropna().unique().astype(int).tolist())
meses = list(range(1, 13))
dias  = list(range(1, 32))

FIG_MAPA_INICIAL = construir_mapa(acc_asignado, clima_df)
print("✅  Todo listo — iniciando servidor.\n")


# ════════════════════════════════════════════════════════════
#  6. PRE-CÓMPUTO PESTAÑA 1 — KPIs
# ════════════════════════════════════════════════════════════

_DIA_MAP = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue",
            4: "Vie", 5: "Sáb", 6: "Dom"}
_DIA_ORD = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

_anual = (acc_asignado
          .groupby(acc_asignado["Fecha"].dt.year)
          .size()
          .reset_index(name="Accidentes"))
_anual.columns = ["Año", "Accidentes"]

FIG_ANUAL = px.line(
    _anual, x="Año", y="Accidentes",
    title="Tendencia anual de accidentes",
    template="plotly_dark",
    markers=True,
    color_discrete_sequence=[CYAN],
)
FIG_ANUAL.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
    title_font_color=CYAN, height=400,
    xaxis=dict(dtick=1),
)

_tmp3 = pd.DataFrame({
    "hourAcc":   acc_asignado["hourAcc"],
    "DiaSemana": acc_asignado["Fecha"].dt.dayofweek.map(_DIA_MAP),
})
_conteo_hd = (_tmp3.groupby(["hourAcc", "DiaSemana"])
              .size().reset_index(name="Accidentes"))

FIG_HORA_DIA = px.density_heatmap(
    _conteo_hd, x="hourAcc", y="DiaSemana", z="Accidentes",
    histfunc="sum",
    color_continuous_scale="Viridis",
    title="Accidentes por hora y día de semana",
    template="plotly_dark",
    labels={"hourAcc": "Hora del día", "DiaSemana": "Día"},
    category_orders={"DiaSemana": _DIA_ORD},
)
FIG_HORA_DIA.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
    title_font_color=CYAN, height=380,
    xaxis=dict(dtick=1),
)

_conteo_mes = (pd.DataFrame({
    "mes_num": acc_asignado["Fecha"].dt.month,
}).assign(mes_nom=lambda d: d["mes_num"].map(MESES_NOMBRE))
 .groupby(["mes_num", "mes_nom"]).size()
 .reset_index(name="Accidentes")
 .sort_values("mes_num"))

FIG_MES = px.bar(
    _conteo_mes, x="mes_nom", y="Accidentes",
    title="Accidentes por mes",
    template="plotly_dark",
    color="Accidentes",
    color_continuous_scale="Blues",
    labels={"mes_nom": "Mes"},
)
FIG_MES.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
    title_font_color=CYAN, height=380,
    coloraxis_showscale=False,
)

KPI_TOTAL       = len(acc_asignado)
KPI_FATALES     = int((acc_asignado["Gravedad_Indicador_Tradicional"] == "MUERTO").sum())
KPI_TASA_FAT    = f"{round(KPI_FATALES / KPI_TOTAL * 100, 2):.2f}%" if KPI_TOTAL else "—"
KPI_LOCALIDADES = (int(acc_asignado["Localidad"].nunique())
                   if "Localidad" in acc_asignado.columns else 0)

_por_anno = acc_asignado.groupby(acc_asignado["Fecha"].dt.year).size()

KPI_PROM_ANUAL  = f"{round(_por_anno.mean()):,}" if len(_por_anno) else "—"

_anno_max       = int(_por_anno.idxmax()) if len(_por_anno) else None
KPI_ANNO_MAX    = str(_anno_max) if _anno_max else "—"
KPI_ANNO_MAX_LBL = (f"Año con más accidentes · {int(_por_anno.max()):,} casos"
                    if len(_por_anno) else "Año con más accidentes")

if 2019 in _por_anno.index and 2020 in _por_anno.index:
    _v = round((_por_anno[2020] - _por_anno[2019]) / _por_anno[2019] * 100, 1)
    KPI_VAR_1920 = f"{'+' if _v >= 0 else ''}{_v}%"
    KPI_VAR_COLOR = ORANGE if _v >= 0 else GREEN
else:
    KPI_VAR_1920  = "N/D"
    KPI_VAR_COLOR = MUTED

_anni_tend = [y for y in [2021, 2022, 2023, 2024] if y in _por_anno.index]
if len(_anni_tend) >= 2:
    _tp = round((_por_anno[_anni_tend[-1]] - _por_anno[_anni_tend[0]])
                / _por_anno[_anni_tend[0]] * 100, 1)
    KPI_TEND      = f"{'+' if _tp >= 0 else ''}{_tp}% {'↑' if _tp >= 0 else '↓'}"
    KPI_TEND_COLOR = ORANGE if _tp >= 0 else GREEN
elif len(_anni_tend) == 1:
    KPI_TEND      = f"{_por_anno[_anni_tend[0]]:,}"
    KPI_TEND_COLOR = CYAN
else:
    KPI_TEND      = "N/D"
    KPI_TEND_COLOR = MUTED

if "Con_Moto" in acc_asignado.columns and KPI_TOTAL:
    _moto_si    = (acc_asignado["Con_Moto"].astype(str).str.upper() == "SI").sum()
    KPI_PCT_MOTO = f"{round(_moto_si / KPI_TOTAL * 100, 1)}%"
else:
    KPI_PCT_MOTO = "N/D"

KPI_HORA_MODAL = (f"{int(acc_asignado['hourAcc'].mode().iloc[0]):02d}:00 h"
                  if len(acc_asignado) else "—")

KPI_LOCALIDAD_TOP = (acc_asignado["Localidad"].value_counts().index[0].title()
                     if "Localidad" in acc_asignado.columns and len(acc_asignado) else "—")

_KPI_ROW = {"display": "flex", "gap": "14px",
            "padding": "8px 32px 4px", "flexWrap": "wrap"}
_KPI_HDR = {"color": MUTED, "fontSize": "0.68rem", "fontWeight": "700",
            "letterSpacing": "0.12em", "textTransform": "uppercase",
            "padding": "14px 32px 2px", "borderTop": f"1px solid {BORDER}",
            "marginTop": "4px"}

_grav_counts = (acc_asignado["Gravedad_Indicador_Tradicional"]
                .value_counts().reset_index())
_grav_counts.columns = ["Gravedad", "Accidentes"]

FIG_GRAVEDAD = px.pie(
    _grav_counts, names="Gravedad", values="Accidentes",
    color="Gravedad",
    color_discrete_map={"ILESO": "#50FA7B", "HERIDO": "#FFB86C", "MUERTO": "#FF5555"},
    title="Distribución por gravedad",
    template="plotly_dark",
)
FIG_GRAVEDAD.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
    title_font_color=CYAN, height=320, autosize=True,
)

if "Localidad" in acc_asignado.columns:
    _loc_top = acc_asignado["Localidad"].value_counts().head(10).reset_index()
    _loc_top.columns = ["Localidad", "Accidentes"]
    _loc_top = _loc_top.sort_values("Accidentes", ascending=True)
else:
    _loc_top = pd.DataFrame({"Localidad": [], "Accidentes": []})

FIG_LOCALIDAD_TOP = px.bar(
    _loc_top, x="Accidentes", y="Localidad",
    orientation="h",
    color_discrete_sequence=[PURPLE],
    title="Top 10 localidades",
    template="plotly_dark",
    labels={"Localidad": ""},
)
FIG_LOCALIDAD_TOP.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
    title_font_color=CYAN, height=320, autosize=True,
)

_hora_counts = acc_asignado.groupby("hourAcc").size().reset_index(name="Accidentes")

FIG_HORA_KPI = px.bar(
    _hora_counts, x="hourAcc", y="Accidentes",
    color="Accidentes",
    color_continuous_scale="Blues",
    title="Accidentes por hora del día",
    template="plotly_dark",
    labels={"hourAcc": "Hora"},
)
_hora_modal_int = int(KPI_HORA_MODAL.split(":")[0]) if KPI_HORA_MODAL != "—" else 0
FIG_HORA_KPI.add_vline(
    x=_hora_modal_int,
    line_dash="dash", line_color=CYAN, line_width=2,
    annotation_text=f"Moda: {KPI_HORA_MODAL}",
    annotation_font_color=CYAN,
    annotation_position="top right",
)
FIG_HORA_KPI.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
    title_font_color=CYAN, height=320, autosize=True,
    coloraxis_showscale=False,
    xaxis=dict(dtick=1),
)

_n_moto = (
    (acc_asignado["Con_Moto"].astype(str).str.upper() == "SI").sum()
    if "Con_Moto" in acc_asignado.columns else 0
)

FIG_MOTO_VS_RESTO = px.pie(
    names=["Con moto", "Sin moto"],
    values=[_n_moto, KPI_TOTAL - _n_moto],
    color_discrete_sequence=["#FFB86C", "#6272a4"],
    title="Participación de motos",
    template="plotly_dark",
)
FIG_MOTO_VS_RESTO.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
    title_font_color=CYAN, height=320, autosize=True,
)


# ════════════════════════════════════════════════════════════
#  7. PRE-CÓMPUTO PESTAÑA 4 — CLIMA
# ════════════════════════════════════════════════════════════

_clima_grav = (acc_asignado
    .groupby(["ClaseLluvia", "Gravedad_Indicador_Tradicional"])
    .size().reset_index(name="Accidentes"))
_clima_grav["ClaseLluvia"] = pd.Categorical(
    _clima_grav["ClaseLluvia"], categories=LLUVIA_ORDEN, ordered=True)
_clima_grav = _clima_grav.sort_values("ClaseLluvia")

FIG_CLIMA_GRAV = px.bar(
    _clima_grav, x="ClaseLluvia", y="Accidentes",
    color="Gravedad_Indicador_Tradicional",
    barmode="group",
    color_discrete_map={"ILESO": "#50FA7B", "HERIDO": "#FFB86C", "MUERTO": "#FF5555"},
    title="Accidentes por categoría de lluvia y gravedad",
    template="plotly_dark",
    labels={"ClaseLluvia": "Categoría de lluvia",
            "Gravedad_Indicador_Tradicional": "Gravedad"},
    category_orders={"ClaseLluvia": LLUVIA_ORDEN,
                     "Gravedad_Indicador_Tradicional": ["ILESO", "HERIDO", "MUERTO"]},
)
FIG_CLIMA_GRAV.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
    title_font_color=CYAN, height=420,
    legend=dict(bgcolor="rgba(13,17,23,0.88)", font=dict(color=TEXT, size=11),
                bordercolor=BORDER, borderwidth=1),
)

_fat_lluvia = (acc_asignado
    .groupby("ClaseLluvia")
    .apply(lambda d: pd.Series({
        "Total":    len(d),
        "Muertos":  (d["Gravedad_Indicador_Tradicional"] == "MUERTO").sum(),
        "PctFatal": round(
            100 * (d["Gravedad_Indicador_Tradicional"] == "MUERTO").sum() / len(d), 2
        ) if len(d) else 0,
    }))
    .reset_index())
_fat_lluvia["ClaseLluvia"] = pd.Categorical(
    _fat_lluvia["ClaseLluvia"], categories=LLUVIA_ORDEN, ordered=True)
_fat_lluvia = _fat_lluvia.sort_values("ClaseLluvia")

FIG_CLIMA_FATAL = go.Figure()
FIG_CLIMA_FATAL.add_trace(go.Bar(
    x=_fat_lluvia["ClaseLluvia"],
    y=_fat_lluvia["Total"],
    name="Total accidentes",
    marker_color=PURPLE,
    opacity=0.65,
    yaxis="y1",
))
FIG_CLIMA_FATAL.add_trace(go.Scatter(
    x=_fat_lluvia["ClaseLluvia"],
    y=_fat_lluvia["PctFatal"],
    name="% fatalidad",
    mode="lines+markers",
    line=dict(color="#FF5555", width=2.5),
    marker=dict(size=8, color="#FF5555"),
    yaxis="y2",
))
FIG_CLIMA_FATAL.update_layout(
    title="Volumen de accidentes y tasa de fatalidad por categoría de lluvia",
    paper_bgcolor=BG, plot_bgcolor=BG2,
    font_color=TEXT, title_font_color=CYAN, height=400,
    yaxis=dict(title="Total accidentes", gridcolor=BORDER, color=PURPLE),
    yaxis2=dict(title="% fatalidad", overlaying="y", side="right",
                color="#FF5555", showgrid=False, ticksuffix="%"),
    xaxis=dict(categoryorder="array", categoryarray=LLUVIA_ORDEN,
               gridcolor=BORDER, color=TEXT),
    legend=dict(bgcolor="rgba(13,17,23,0.88)", font=dict(color=TEXT, size=11),
                bordercolor=BORDER, borderwidth=1),
    barmode="overlay",
    hovermode="x unified",
)

_clima_mes = (acc_asignado
    .groupby([acc_asignado["Fecha"].dt.month.rename("mes"), "ClaseLluvia"])
    .agg(Accidentes=("ClaseLluvia", "count"), ValorMedio=("ValorObservado", "mean"))
    .reset_index())
_clima_mes["mes_nom"] = _clima_mes["mes"].map(MESES_NOMBRE)
_clima_mes["ClaseLluvia"] = pd.Categorical(
    _clima_mes["ClaseLluvia"], categories=LLUVIA_ORDEN, ordered=True)
_clima_mes = _clima_mes.sort_values(["mes", "ClaseLluvia"])

FIG_SCATTER_CLIMA = px.scatter(
    _clima_mes, x="mes_nom", y="ValorMedio",
    color="ClaseLluvia",
    size="Accidentes", size_max=40,
    hover_name="mes_nom",
    hover_data={"ValorMedio": ":.2f", "Accidentes": True, "mes_nom": False},
    color_discrete_map=LLUVIA_COLOR_HEX,
    title="Precipitación media por mes y categoría de lluvia",
    template="plotly_dark",
    labels={"mes_nom": "Mes", "ValorMedio": "Precipitación media (mm/h)",
            "Accidentes": "Nº Accidentes"},
    category_orders={"ClaseLluvia": LLUVIA_ORDEN,
                     "mes_nom": list(MESES_NOMBRE.values())},
)
FIG_SCATTER_CLIMA.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
    title_font_color=CYAN, height=480,
    xaxis=dict(tickangle=-30),
    legend=dict(bgcolor="rgba(13,17,23,0.88)", font=dict(color=TEXT, size=11),
                bordercolor=BORDER, borderwidth=1),
)

_clima_hora = (acc_asignado
    .groupby(["ClaseLluvia", "hourAcc"])
    .size().reset_index(name="Accidentes"))
_clima_hora["ClaseLluvia"] = pd.Categorical(
    _clima_hora["ClaseLluvia"], categories=LLUVIA_ORDEN, ordered=True)
_clima_hora = _clima_hora.sort_values(["ClaseLluvia", "hourAcc"])

FIG_CLIMA_HORA = px.density_heatmap(
    _clima_hora, x="hourAcc", y="ClaseLluvia", z="Accidentes",
    histfunc="sum",
    color_continuous_scale="Plasma",
    title="Accidentes por hora del día y categoría de lluvia",
    template="plotly_dark",
    labels={"hourAcc": "Hora del día", "ClaseLluvia": "Categoría de lluvia"},
    category_orders={"ClaseLluvia": LLUVIA_ORDEN},
)
FIG_CLIMA_HORA.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
    title_font_color=CYAN, height=380,
    xaxis=dict(dtick=1),
)


# ════════════════════════════════════════════════════════════
#  8. PRE-CÓMPUTO PESTAÑA 5 — FACTORES DE RIESGO
# ════════════════════════════════════════════════════════════

_FACTOR_COLS = [c for c in [
    "Con_Embriaguez", "Con_Velocidad", "Con_Moto", "Con_Peaton",
    "Con_Bicicleta",  "Con_Huecos",   "Con_Menores", "Con_Carga",
] if c in acc_asignado.columns]

_GRAV_ORD    = ["ILESO", "HERIDO", "MUERTO"]
_GRAV_COLORS = {"ILESO": "#50FA7B", "HERIDO": "#FFB86C", "MUERTO": "#FF5555"}
_grav_present = [g for g in _GRAV_ORD
                 if g in acc_asignado["Gravedad_Indicador_Tradicional"].unique()]

_factor_label = {
    "Con_Embriaguez": "Embriaguez",
    "Con_Velocidad":  "Velocidad",
    "Con_Moto":       "Moto",
    "Con_Peaton":     "Peatón",
    "Con_Bicicleta":  "Bicicleta",
    "Con_Huecos":     "Huecos",
    "Con_Menores":    "Menores",
    "Con_Carga":      "Carga",
}

print("⏳  Pre-computando máscaras de factores…")
_factor_masks = {}
for f in _FACTOR_COLS:
    _factor_masks[f] = acc_asignado[f].astype(str).str.upper() == "SI"

_es_muerto = acc_asignado["Gravedad_Indicador_Tradicional"] == "MUERTO"
_grav_serie = acc_asignado["Gravedad_Indicador_Tradicional"]
print("✅  Máscaras listas.")

if _FACTOR_COLS and _grav_present:
    _pct_rows = []
    for f in _FACTOR_COLS:
        mask  = _factor_masks[f]
        total = mask.sum()
        row   = {"Factor": _factor_label.get(f, f)}
        for g in _grav_present:
            row[g] = (round(100 * (_grav_serie[mask] == g).sum() / total, 1)
                      if total else 0.0)
        _pct_rows.append(row)
    _pct_df = pd.DataFrame(_pct_rows).set_index("Factor")

    FIG_FACTORES = px.imshow(
        _pct_df, text_auto=True,
        color_continuous_scale="RdYlGn_r",
        title="% de gravedad por factor de riesgo",
        template="plotly_dark",
        labels={"color": "%", "x": "Gravedad", "y": "Factor"},
        aspect="auto",
    )
    FIG_FACTORES.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
        title_font_color=CYAN, height=380,
    )
else:
    FIG_FACTORES = go.Figure().update_layout(
        title="Sin columnas Con_* disponibles",
        paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
        title_font_color=CYAN, height=380,
    )

_mort_rows = []
for f in _FACTOR_COLS:
    mask     = _factor_masks[f]
    total_s  = mask.sum()
    total_n  = (~mask).sum()
    tasa_con = round(100 * _es_muerto[mask].sum() / total_s, 2) if total_s else 0
    tasa_sin = round(100 * _es_muerto[~mask].sum() / total_n, 2) if total_n else 0
    _mort_rows.append({
        "Factor":      _factor_label.get(f, f),
        "Con factor":  tasa_con,
        "Sin factor":  tasa_sin,
        "Diferencia":  round(tasa_con - tasa_sin, 2),
    })
_mort_df = pd.DataFrame(_mort_rows).sort_values("Con factor", ascending=True)

FIG_MORTALIDAD = go.Figure()
FIG_MORTALIDAD.add_trace(go.Bar(
    y=_mort_df["Factor"],
    x=_mort_df["Sin factor"],
    name="Sin factor",
    orientation="h",
    marker_color="#6272a4",
    opacity=0.8,
))
FIG_MORTALIDAD.add_trace(go.Bar(
    y=_mort_df["Factor"],
    x=_mort_df["Con factor"],
    name="Con factor activo",
    orientation="h",
    marker_color="#FF5555",
    opacity=0.9,
))
FIG_MORTALIDAD.update_layout(
    title="Tasa de mortalidad (% MUERTO): con vs. sin factor",
    barmode="overlay",
    paper_bgcolor=BG, plot_bgcolor=BG2,
    font_color=TEXT, title_font_color=CYAN, height=380,
    xaxis=dict(title="% accidentes mortales", gridcolor=BORDER,
               ticksuffix="%", color=TEXT),
    yaxis=dict(gridcolor=BORDER, color=TEXT),
    legend=dict(bgcolor="rgba(13,17,23,0.88)", font=dict(color=TEXT, size=11),
                bordercolor=BORDER, borderwidth=1),
    hovermode="y unified",
)

if _FACTOR_COLS:
    _bool_matrix = pd.concat(
        [_factor_masks[f].rename(f) for f in _FACTOR_COLS], axis=1)
    _n_factores_serie = _bool_matrix.sum(axis=1)
    _combo = acc_asignado[["Gravedad_Indicador_Tradicional"]].copy()
    _combo["n_factores"] = _n_factores_serie
    _combo_grp = (_combo
        .groupby(["n_factores", "Gravedad_Indicador_Tradicional"])
        .size().reset_index(name="Acc"))
    _combo_tot = _combo_grp.groupby("n_factores")["Acc"].transform("sum")
    _combo_grp["Pct"] = round(100 * _combo_grp["Acc"] / _combo_tot, 1)
    _combo_grp["n_factores"] = _combo_grp["n_factores"].astype(str) + " factor(es)"

    FIG_COMBO = px.bar(
        _combo_grp, x="n_factores", y="Pct",
        color="Gravedad_Indicador_Tradicional",
        barmode="stack",
        color_discrete_map=_GRAV_COLORS,
        title="Gravedad según número de factores simultáneos",
        template="plotly_dark",
        labels={"n_factores": "Factores activos",
                "Pct": "% del total",
                "Gravedad_Indicador_Tradicional": "Gravedad"},
        category_orders={"Gravedad_Indicador_Tradicional": _GRAV_ORD},
        text="Pct",
    )
    FIG_COMBO.update_traces(texttemplate="%{text:.0f}%", textposition="inside",
                            textfont_size=10)
    FIG_COMBO.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
        title_font_color=CYAN, height=400,
        yaxis=dict(title="% de accidentes", gridcolor=BORDER, ticksuffix="%"),
        xaxis=dict(gridcolor=BORDER, color=TEXT),
        legend=dict(bgcolor="rgba(13,17,23,0.88)", font=dict(color=TEXT, size=11),
                    bordercolor=BORDER, borderwidth=1),
    )
else:
    FIG_COMBO = go.Figure().update_layout(
        title="Sin factores disponibles", paper_bgcolor=BG,
        plot_bgcolor=BG2, font_color=TEXT, title_font_color=CYAN, height=400)

if _FACTOR_COLS and _grav_present:
    _radar_df = []
    _grav_masks = {g: (_grav_serie == g) for g in _grav_present}
    for g in _grav_present:
        n = _grav_masks[g].sum()
        for f in _FACTOR_COLS:
            pct = round(100 * (_factor_masks[f] & _grav_masks[g]).sum() / n, 1) if n else 0
            _radar_df.append({"Gravedad": g, "Factor": _factor_label.get(f, f), "Pct": pct})
    _radar_df = pd.DataFrame(_radar_df)

    FIG_RADAR = go.Figure()
    for g in _grav_present:
        sub  = _radar_df[_radar_df["Gravedad"] == g]
        cats = sub["Factor"].tolist() + [sub["Factor"].iloc[0]]
        vals = sub["Pct"].tolist()    + [sub["Pct"].iloc[0]]
        col  = _GRAV_COLORS[g]
        r_int, g_int, b_int = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        FIG_RADAR.add_trace(go.Scatterpolar(
            r=vals, theta=cats,
            fill="toself", name=g,
            line_color=col,
            fillcolor=f"rgba({r_int},{g_int},{b_int},0.10)",
            opacity=0.85,
        ))
    FIG_RADAR.update_layout(
        title="Perfil de factores por gravedad (% de presencia)",
        polar=dict(
            bgcolor=BG2,
            radialaxis=dict(visible=True, color=MUTED, gridcolor=BORDER,
                            ticksuffix="%", range=[0, _radar_df["Pct"].max() * 1.15]),
            angularaxis=dict(color=TEXT, gridcolor=BORDER),
        ),
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font_color=TEXT, title_font_color=CYAN, height=440,
        legend=dict(bgcolor="rgba(13,17,23,0.88)", font=dict(color=TEXT, size=11),
                    bordercolor=BORDER, borderwidth=1),
        template="plotly_dark",
    )
else:
    FIG_RADAR = go.Figure().update_layout(
        title="Sin datos para radar", paper_bgcolor=BG,
        plot_bgcolor=BG2, font_color=TEXT, title_font_color=CYAN, height=440)

if _FACTOR_COLS and "Clase_Acc" in acc_asignado.columns:
    _inc_rows = []
    for f in _FACTOR_COLS:
        mask = _factor_masks[f]
        if not mask.any():
            continue
        top_clase = acc_asignado.loc[mask, "Clase_Acc"].value_counts().head(3)
        n_f = mask.sum()
        for clase, cnt in top_clase.items():
            _inc_rows.append({
                "Factor": _factor_label.get(f, f),
                "Clase":  clase,
                "Pct":    round(100 * cnt / n_f, 1),
            })
    _inc_df = pd.DataFrame(_inc_rows) if _inc_rows else pd.DataFrame(
        columns=["Factor", "Clase", "Pct"])

    FIG_CLASE_FACTOR = px.bar(
        _inc_df, x="Factor", y="Pct", color="Clase",
        barmode="stack",
        title="Top 3 clases de accidente por factor (% dentro del factor)",
        template="plotly_dark",
        labels={"Factor": "Factor de riesgo", "Pct": "%", "Clase": "Clase"},
        text="Pct",
    )
    FIG_CLASE_FACTOR.update_traces(texttemplate="%{text:.0f}%",
                                   textposition="inside", textfont_size=10)
    FIG_CLASE_FACTOR.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
        title_font_color=CYAN, height=400,
        yaxis=dict(title="% de accidentes", gridcolor=BORDER, ticksuffix="%"),
        xaxis=dict(gridcolor=BORDER, color=TEXT, tickangle=-20),
        legend=dict(bgcolor="rgba(13,17,23,0.88)", font=dict(color=TEXT, size=11),
                    bordercolor=BORDER, borderwidth=1),
    )
else:
    FIG_CLASE_FACTOR = go.Figure().update_layout(
        title="Sin datos de clase disponibles", paper_bgcolor=BG,
        plot_bgcolor=BG2, font_color=TEXT, title_font_color=CYAN, height=400)

_factor_kpis = {}
for f in _FACTOR_COLS:
    mask = _factor_masks[f]
    n    = int(mask.sum())
    pct  = round(100 * n / len(acc_asignado), 1) if len(acc_asignado) else 0
    mort = round(100 * _es_muerto[mask].sum() / n, 2) if n else 0
    _factor_kpis[f] = {"n": n, "pct": pct, "mortalidad": mort,
                       "label": _factor_label.get(f, f)}

_factor_mas_frecuente = (max(_factor_kpis, key=lambda f: _factor_kpis[f]["n"])
                         if _factor_kpis else None)
_factor_mas_letal     = (max(_factor_kpis, key=lambda f: _factor_kpis[f]["mortalidad"])
                         if _factor_kpis else None)


# ════════════════════════════════════════════════════════════
#  9. HELPERS UI
# ════════════════════════════════════════════════════════════

def card(label, value, color=CYAN):
    return html.Div([
        html.Div(value, style={
            "fontSize": "1.9rem", "fontWeight": "800",
            "color": color, "letterSpacing": "-1px", "lineHeight": "1.1"
        }),
        html.Div(label, style={
            "fontSize": "0.70rem", "color": MUTED,
            "marginTop": "5px", "letterSpacing": "0.08em",
            "textTransform": "uppercase"
        }),
    ], style={
        "background": f"linear-gradient(135deg,#161b2e 0%,{BG} 100%)",
        "border": f"1px solid {BORDER}", "borderRadius": "12px",
        "padding": "14px 20px", "flex": "1", "minWidth": "130px",
    })


def lbl(txt):
    return html.Label(txt, style={
        "color": MUTED, "fontSize": "0.71rem",
        "textTransform": "uppercase", "letterSpacing": "0.07em",
        "display": "block", "marginBottom": "5px"
    })


def seccion_titulo(txt, icono="▸"):
    return html.Div(
        f"{icono} {txt}",
        style={
            "color": CYAN, "fontSize": "0.72rem", "fontWeight": "700",
            "letterSpacing": "0.12em", "textTransform": "uppercase",
            "padding": "10px 32px 4px", "marginTop": "4px",
            "borderTop": f"1px solid {BORDER}"
        }
    )


# ════════════════════════════════════════════════════════════
#  10. LAYOUT
# ════════════════════════════════════════════════════════════

app = Dash(__name__, title="Voronoi Climático · Bogotá")

_header = html.Div([
    html.Span("◈ ", style={"color": CYAN, "fontSize": "1.6rem"}),
    html.Div([
        html.H1("VORONOI CLIMÁTICO",
                style={"margin": "0", "fontSize": "1.35rem", "fontWeight": "900",
                       "letterSpacing": "0.18em", "color": "#E8EAF6"}),
        html.P(
            "Asignación de sensores IDEAM a registros de accidentalidad · Bogotá D.C.",
            style={"margin": "0", "fontSize": "0.73rem", "color": MUTED}
        ),
    ]),
    html.Div("⚡ Plotly · Scattermap", style={
        "marginLeft": "auto",
        "background": "#0a1628",
        "border": f"1px solid {CYAN}",
        "borderRadius": "6px",
        "padding": "4px 12px",
        "fontSize": "0.72rem",
        "color": CYAN,
        "fontWeight": "700",
        "letterSpacing": "0.10em",
        "whiteSpace": "nowrap",
    }),
], style={"display": "flex", "alignItems": "center", "gap": "14px",
          "padding": "22px 32px 14px", "borderBottom": f"1px solid {BORDER}"})

app.layout = html.Div(
    style={"fontFamily": FONT, "backgroundColor": BG, "minHeight": "100vh", "color": TEXT},
    children=[

    _header,

    dcc.Tabs(
        id="tabs-main",
        value="tab-kpi",
        style={"backgroundColor": BG, "borderBottom": f"1px solid {BORDER}"},
        children=[

        # ── PESTAÑA 1: KPIs ───────────────────────────────────
        dcc.Tab(label="📊 KPIs", value="tab-kpi",
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE,
                children=[

            html.P("▸ Resumen general", style=_KPI_HDR),
            html.Div([
                card("Total siniestros",      f"{KPI_TOTAL:,}",    CYAN),
                card("Casos fatales",         f"{KPI_FATALES:,}",  ORANGE),
                card("Tasa de fatalidad",     KPI_TASA_FAT,        PINK),
                card("Localidades cubiertas", f"{KPI_LOCALIDADES}", PURPLE),
            ], style=_KPI_ROW),

            html.P("▸ Tendencia temporal", style=_KPI_HDR),
            html.Div([
                card("Promedio anual",      KPI_PROM_ANUAL,  CYAN),
                card(KPI_ANNO_MAX_LBL,      KPI_ANNO_MAX,    ORANGE),
                card("Variación 2019→2020", KPI_VAR_1920,    KPI_VAR_COLOR),
                card("Tendencia 2021-2024", KPI_TEND,        KPI_TEND_COLOR),
            ], style=_KPI_ROW),

            html.P("▸ Factores y patrones", style=_KPI_HDR),
            html.Div([
                card("Involucra moto",            KPI_PCT_MOTO,      ORANGE),
                card("Hora modal de mayor frec.", KPI_HORA_MODAL,    CYAN),
                card("Localidad con más casos",   KPI_LOCALIDAD_TOP, PURPLE),
            ], style={**_KPI_ROW, "paddingBottom": "12px"}),

            dcc.Graph(
                figure=FIG_ANUAL,
                config={"displayModeBar": False},
                style={"padding": "8px 32px 0"},
            ),

            html.Div([
                dcc.Graph(figure=FIG_GRAVEDAD,
                          config={"displayModeBar": False},
                          style={"height": "320px", "minHeight": "320px"}),
                dcc.Graph(figure=FIG_LOCALIDAD_TOP,
                          config={"displayModeBar": False},
                          style={"height": "320px", "minHeight": "320px"}),
                dcc.Graph(figure=FIG_HORA_KPI,
                          config={"displayModeBar": False},
                          style={"height": "320px", "minHeight": "320px"}),
                dcc.Graph(figure=FIG_MOTO_VS_RESTO,
                          config={"displayModeBar": False},
                          style={"height": "320px", "minHeight": "320px"}),
            ], id="grid-kpi", style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                                     "gap": "16px", "padding": "8px 32px 32px"}),
        ]),

        # ── PESTAÑA 2: Mapa Voronoi ───────────────────────────
        dcc.Tab(label="🗺 Mapa Voronoi", value="tab-mapa",
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE,
                children=[

            html.Div(id="contador",
                     style={"padding": "6px 32px 6px", "color": MUTED,
                            "fontSize": "0.78rem",
                            "borderBottom": f"1px solid {BORDER}"}),

            seccion_titulo("Filtros de Fecha y Hora", "📅"),
            html.Div([
                html.Div([
                    lbl("Año"),
                    dcc.Dropdown(id="dd-anno",
                        options=[{"label": str(a), "value": a} for a in annos],
                        value=None, multi=True, placeholder="Todos",
                        style={"minWidth": "140px", **DD_STYLE}),
                ]),
                html.Div([
                    lbl("Mes"),
                    dcc.Dropdown(id="dd-mes",
                        options=[{"label": MESES_NOMBRE[m], "value": m} for m in meses],
                        value=None, multi=True, placeholder="Todos",
                        style={"minWidth": "170px", **DD_STYLE}),
                ]),
                html.Div([
                    lbl("Día del mes"),
                    dcc.Dropdown(id="dd-dia",
                        options=[{"label": str(d), "value": d} for d in dias],
                        value=None, multi=True, placeholder="Todos",
                        style={"minWidth": "150px", **DD_STYLE}),
                ]),
                html.Div([
                    lbl("Hora (rango)"),
                    dcc.RangeSlider(
                        id="sl-hora",
                        min=0, max=23, step=1,
                        value=[0, 23],
                        marks={h: {"label": f"{h}h",
                                   "style": {"color": MUTED, "fontSize": "0.65rem"}}
                               for h in [0, 3, 6, 9, 12, 15, 18, 21, 23]},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"minWidth": "280px", "flex": "2"}),
            ], style=SEK_STYLE),

            seccion_titulo("Filtros de Accidente", "🚗"),
            html.Div([
                html.Div([
                    lbl("Gravedad"),
                    dcc.Checklist(id="chk-gravedad",
                        options=[{"label": f" {g}", "value": g} for g in gravedades_unicas],
                        value=gravedades_unicas, inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"color": TEXT, "marginRight": "14px",
                                    "fontSize": "0.83rem"}),
                ]),
                html.Div([
                    lbl("Clase de accidente"),
                    dcc.Dropdown(id="dd-clase",
                        options=[{"label": c, "value": c} for c in clases_unicas],
                        value=None, multi=True, placeholder="Todas",
                        style={"minWidth": "220px", **DD_STYLE}),
                ]),
                html.Div([
                    lbl("Tipo de vehículo"),
                    dcc.Dropdown(id="dd-vehiculo",
                        options=[{"label": v, "value": v} for v in vehiculo_unicas],
                        value=None, multi=True, placeholder="Todos",
                        style={"minWidth": "220px", **DD_STYLE}),
                ]),
                html.Div([
                    lbl("Condición"),
                    dcc.Dropdown(id="dd-condicion",
                        options=[{"label": c, "value": c} for c in condicion_unicas],
                        value=None, multi=True, placeholder="Todas",
                        style={"minWidth": "200px", **DD_STYLE}),
                ]),
            ], style=SEK_STYLE),

            seccion_titulo("Filtros Climáticos y Visualización", "🌧"),
            html.Div([
                html.Div([
                    lbl("Clasificación de lluvia"),
                    dcc.Checklist(id="chk-lluvia",
                        options=[
                            {"label": html.Span(
                                [html.Span("■ ", style={"color": LLUVIA_COLOR_HEX[c]}),
                                 f" {c}"],
                            ), "value": c}
                            for c in lluvia_presentes
                        ],
                        value=lluvia_presentes, inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={
                            "color": TEXT, "marginRight": "14px", "fontSize": "0.83rem",
                            "display": "inline-flex", "alignItems": "center"
                        }),
                ], style={"flex": "3"}),
                html.Div([
                    lbl("Colorear puntos por"),
                    dcc.Dropdown(id="dd-color",
                        options=[
                            {"label": "☁ Clase de lluvia (categoría)",    "value": "ClaseLluvia"},
                            {"label": "🎯 Matriz lluvia × accidentalidad", "value": "Matriz"},
                            {"label": "💧 Valor observado (mm/h)",         "value": "ValorObservado"},
                            {"label": "📏 Distancia al sensor (km)",       "value": "dist_km"},
                            {"label": "🕐 Hora del accidente",             "value": "hourAcc"},
                        ],
                        value="ClaseLluvia", clearable=False,
                        style={"minWidth": "280px", **DD_STYLE}),
                ]),
                html.Div([
                    lbl("Regiones Voronoi"),
                    dcc.Checklist(id="chk-voronoi",
                        options=[{"label": " Mostrar relleno", "value": "on"}],
                        value=["on"],
                        inputStyle={"marginRight": "5px"},
                        labelStyle={"color": TEXT, "fontSize": "0.83rem"}),
                ]),
            ], style=SEK_STYLE),

            html.Div([
                dcc.Graph(
                    id="mapa-voronoi",
                    figure=FIG_MAPA_INICIAL,
                    config={"scrollZoom": True, "displayModeBar": False},
                    style={"width": "100%", "height": "650px"},
                ),
            ], style={
                "margin": "0 32px",
                "borderRadius": "10px",
                "overflow": "hidden", "border": f"1px solid {BORDER}",
            }),

            html.Div([
                html.Span("☁ ", style={"color": CYAN}),
                html.Span("Clasificación: ", style={"color": MUTED, "fontSize": "0.78rem"}),
                *[
                    html.Span(
                        f"■ {cat}  ",
                        style={"color": LLUVIA_COLOR_HEX[cat],
                               "fontSize": "0.78rem", "fontWeight": "700"}
                    )
                    for cat in LLUVIA_ORDEN
                ],
                html.Span("  |  valores en mm/h",
                          style={"color": MUTED, "fontSize": "0.73rem"}),
            ], style={"padding": "8px 32px 4px", "flexWrap": "wrap",
                      "display": "flex", "alignItems": "center"}),

            html.Div([
                dcc.Graph(
                    id="fig-clima-acc",
                    config={"displayModeBar": False},
                ),
            ], style={"padding": "8px 32px 32px"}),
        ]),

        # ── PESTAÑA 3: Temporalidad ───────────────────────────
        dcc.Tab(label="🕐 Temporalidad", value="tab-temporal",
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE,
                children=[

            seccion_titulo("Filtros de Fecha y Hora", "📅"),
            html.Div([
                html.Div([
                    lbl("Año"),
                    dcc.Dropdown(id="dd-temp-anno",
                        options=[{"label": str(a), "value": a} for a in annos],
                        value=None, multi=True, placeholder="Todos",
                        style={"minWidth": "140px", **DD_STYLE}),
                ]),
                html.Div([
                    lbl("Mes"),
                    dcc.Dropdown(id="dd-temp-mes",
                        options=[{"label": MESES_NOMBRE[m], "value": m} for m in meses],
                        value=None, multi=True, placeholder="Todos",
                        style={"minWidth": "170px", **DD_STYLE}),
                ]),
                html.Div([
                    lbl("Día del mes"),
                    dcc.Dropdown(id="dd-temp-dia",
                        options=[{"label": str(d), "value": d} for d in dias],
                        value=None, multi=True, placeholder="Todos",
                        style={"minWidth": "150px", **DD_STYLE}),
                ]),
                html.Div([
                    lbl("Hora (rango)"),
                    dcc.RangeSlider(
                        id="sl-temp-hora",
                        min=0, max=23, step=1,
                        value=[0, 23],
                        marks={h: {"label": f"{h}h",
                                   "style": {"color": MUTED, "fontSize": "0.65rem"}}
                               for h in [0, 3, 6, 9, 12, 15, 18, 21, 23]},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"minWidth": "280px", "flex": "2"}),
            ], style=SEK_STYLE),

            seccion_titulo("Filtros de Accidente", "🚗"),
            html.Div([
                html.Div([
                    lbl("Gravedad"),
                    dcc.Checklist(id="chk-temp-gravedad",
                        options=[{"label": f" {g}", "value": g} for g in gravedades_unicas],
                        value=gravedades_unicas, inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"color": TEXT, "marginRight": "14px",
                                    "fontSize": "0.83rem"}),
                ]),
                html.Div([
                    lbl("Clase de accidente"),
                    dcc.Dropdown(id="dd-temp-clase",
                        options=[{"label": c, "value": c} for c in clases_unicas],
                        value=None, multi=True, placeholder="Todas",
                        style={"minWidth": "220px", **DD_STYLE}),
                ]),
                html.Div([
                    lbl("Tipo de vehículo"),
                    dcc.Dropdown(id="dd-temp-vehiculo",
                        options=[{"label": v, "value": v} for v in vehiculo_unicas],
                        value=None, multi=True, placeholder="Todos",
                        style={"minWidth": "220px", **DD_STYLE}),
                ]),
                html.Div([
                    lbl("Condición"),
                    dcc.Dropdown(id="dd-temp-condicion",
                        options=[{"label": c, "value": c} for c in condicion_unicas],
                        value=None, multi=True, placeholder="Todas",
                        style={"minWidth": "200px", **DD_STYLE}),
                ]),
            ], style=SEK_STYLE),

            html.Div([
                dcc.Graph(id="fig-temp-anual-grav", config={"displayModeBar": False}),
            ], style={"padding": "20px 32px 0"}),

            html.Div([
                dcc.Graph(id="fig-temp-hora-dia", figure=FIG_HORA_DIA,
                          config={"displayModeBar": False}),
                dcc.Graph(id="fig-temp-mes", figure=FIG_MES,
                          config={"displayModeBar": False}),
            ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                      "gap": "16px", "padding": "20px 32px 0"}),

            html.Div([
                dcc.Graph(id="fig-temp-scatter-anios", config={"displayModeBar": False}),
                dcc.Graph(id="fig-temp-correlacion",   config={"displayModeBar": False}),
            ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                      "gap": "16px", "padding": "16px 32px 32px"}),
        ]),

        # ── PESTAÑA 4: Clima ──────────────────────────────────
        dcc.Tab(label="🌧 Clima", value="tab-clima",
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE,
                children=[

            html.Div([
                card("Accidentes bajo lluvia",
                     f"{int((acc_asignado['ClaseLluvia'] != 'Soleado').sum()):,}", CYAN),
                card("% bajo alguna lluvia",
                     f"{round(100 * (acc_asignado['ClaseLluvia'] != 'Soleado').mean(), 1)}%", PURPLE),
                card("Categoría más frecuente",
                     acc_asignado['ClaseLluvia'].value_counts().index[0], ORANGE),
                card("Categoría más letal",
                     _fat_lluvia.loc[_fat_lluvia['PctFatal'].idxmax(), 'ClaseLluvia']
                     if len(_fat_lluvia) else "—", CYAN),
            ], style={**_KPI_ROW, "padding": "16px 32px 8px"}),

            html.Div([
                dcc.Graph(figure=FIG_CLIMA_GRAV, config={"displayModeBar": False}),
                dcc.Graph(figure=FIG_CLIMA_FATAL, config={"displayModeBar": False}),
            ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                      "gap": "16px", "padding": "8px 32px 0"}),

            html.Div([
                dcc.Graph(figure=FIG_SCATTER_CLIMA, config={"displayModeBar": False}),
                dcc.Graph(figure=FIG_CLIMA_HORA,    config={"displayModeBar": False}),
            ], style={"display": "grid", "gridTemplateColumns": "3fr 2fr",
                      "gap": "16px", "padding": "16px 32px 32px"}),
        ]),

        # ── PESTAÑA 5: Factores de riesgo ─────────────────────
        dcc.Tab(label="⚠ Factores", value="tab-factores",
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE,
                children=[

            html.Div([
                card("Factor más frecuente",
                     _factor_kpis[_factor_mas_frecuente]["label"]
                     if _factor_mas_frecuente else "—", CYAN),
                card("% accidentes con ese factor",
                     f"{_factor_kpis[_factor_mas_frecuente]['pct']}%"
                     if _factor_mas_frecuente else "—", PURPLE),
                card("Factor más letal",
                     _factor_kpis[_factor_mas_letal]["label"]
                     if _factor_mas_letal else "—", CYAN),
                card("Mortalidad con ese factor",
                     f"{_factor_kpis[_factor_mas_letal]['mortalidad']}%"
                     if _factor_mas_letal else "—", PINK),
            ], style={**_KPI_ROW, "padding": "16px 32px 8px"}),

            html.Div([
                dcc.Graph(figure=FIG_RADAR, config={"displayModeBar": False}),
                dcc.Graph(figure=FIG_COMBO, config={"displayModeBar": False}),
            ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                      "gap": "16px", "padding": "8px 32px 0"}),

            html.Div([
                dcc.Graph(figure=FIG_CLASE_FACTOR, config={"displayModeBar": False}),
            ], style={"padding": "16px 32px 32px"}),
        ]),

    ]),
])


# ════════════════════════════════════════════════════════════
#  11. CALLBACKS
# ════════════════════════════════════════════════════════════

def _filtrar(anno, mes, dia, hora_range,
             gravedades, clase, vehiculo, condicion, lluvia):
    df = acc_asignado
    if anno:
        df = df[df["Fecha"].dt.year.isin(anno)]
    if mes:
        df = df[df["Fecha"].dt.month.isin(mes)]
    if dia:
        df = df[df["Fecha"].dt.day.isin(dia)]
    if hora_range:
        df = df[(df["hourAcc"] >= hora_range[0]) & (df["hourAcc"] <= hora_range[1])]
    if gravedades:
        df = df[df["Gravedad_Indicador_Tradicional"].isin(gravedades)]
    if clase:
        df = df[df["Clase_Acc"].isin(clase)]
    if vehiculo and "Tipo_de_vehiculo" in df.columns:
        df = df[df["Tipo_de_vehiculo"].isin(vehiculo)]
    if condicion and "Condicion" in df.columns:
        df = df[df["Condicion"].isin(condicion)]
    if lluvia:
        df = df[df["ClaseLluvia"].isin(lluvia)]
    return df


@app.callback(
    Output("mapa-voronoi",  "figure"),
    Output("fig-clima-acc", "figure"),
    Output("contador",      "children"),
    Input("dd-anno",        "value"),
    Input("dd-mes",         "value"),
    Input("dd-dia",         "value"),
    Input("sl-hora",        "value"),
    Input("chk-gravedad",   "value"),
    Input("dd-clase",       "value"),
    Input("dd-vehiculo",    "value"),
    Input("dd-condicion",   "value"),
    Input("chk-lluvia",     "value"),
    Input("dd-color",       "value"),
    Input("chk-voronoi",    "value"),
)
def actualizar(anno, mes, dia, hora_range,
               gravedades, clase, vehiculo, condicion, lluvia,
               colorear_por, voronoi_on):

    df_filt = _filtrar(anno, mes, dia, hora_range,
                       gravedades, clase, vehiculo, condicion, lluvia)
    n = len(df_filt)

    fig_mapa = construir_mapa(df_filt, clima_df, bool(voronoi_on),
                              colorear_por or "ClaseLluvia")

    _resumen = (df_filt
                .groupby(["ClaseLluvia", "Gravedad_Indicador_Tradicional"])
                .size().reset_index(name="Accidentes"))
    _resumen["ClaseLluvia"] = pd.Categorical(
        _resumen["ClaseLluvia"], categories=LLUVIA_ORDEN, ordered=True)
    _resumen = _resumen.sort_values("ClaseLluvia")

    fig_rel = px.bar(
        _resumen, x="ClaseLluvia", y="Accidentes",
        color="Gravedad_Indicador_Tradicional",
        barmode="group",
        color_discrete_map={"ILESO": "#50FA7B", "HERIDO": "#FFB86C", "MUERTO": "#FF5555"},
        title="Accidentes por categoría de lluvia y gravedad",
        template="plotly_dark",
        labels={"ClaseLluvia": "Categoría de lluvia",
                "Gravedad_Indicador_Tradicional": "Gravedad"},
        category_orders={"ClaseLluvia": LLUVIA_ORDEN,
                         "Gravedad_Indicador_Tradicional": ["ILESO", "HERIDO", "MUERTO"]},
    )
    fig_rel.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
        title_font_color=CYAN, height=380,
        legend=dict(bgcolor="rgba(13,17,23,0.88)",
                    font=dict(color=TEXT, size=11),
                    bordercolor=BORDER, borderwidth=1),
    )

    return fig_mapa, fig_rel, f"{n:,} de {len(acc_asignado):,} accidentes"


@app.callback(
    Output("fig-temp-anual-grav",    "figure"),
    Output("fig-temp-hora-dia",      "figure"),
    Output("fig-temp-mes",           "figure"),
    Output("fig-temp-scatter-anios", "figure"),
    Output("fig-temp-correlacion",   "figure"),
    Input("dd-temp-anno",      "value"),
    Input("dd-temp-mes",       "value"),
    Input("dd-temp-dia",       "value"),
    Input("sl-temp-hora",      "value"),
    Input("chk-temp-gravedad", "value"),
    Input("dd-temp-clase",     "value"),
    Input("dd-temp-vehiculo",  "value"),
    Input("dd-temp-condicion", "value"),
)
def actualizar_temporal(anno, mes, dia, hora_range,
                        gravedades, clase, vehiculo, condicion):

    # Normalizar slider: si llega un solo int, convertir a lista
    if hora_range is None:
        hora_range = [0, 23]
    elif isinstance(hora_range, (int, float)):
        hora_range = [int(hora_range), 23]

    df_temp = _filtrar(anno, mes, dia, hora_range,
                       gravedades, clase, vehiculo, condicion, None)

    _layout_vacio = dict(paper_bgcolor=BG, plot_bgcolor=BG2,
                         font_color=TEXT, title_font_color=CYAN)

    if df_temp.empty or df_temp["Fecha"].isna().all():
        _fig_vacio = go.Figure().update_layout(
            title="Sin datos para los filtros seleccionados", **_layout_vacio)
        return _fig_vacio, _fig_vacio, _fig_vacio, _fig_vacio, _fig_vacio

    df_temp = df_temp[df_temp["Fecha"].notna()].copy()
    df_temp["Año"] = df_temp["Fecha"].dt.year.astype(int)

    # ── Gráfico 1: área o barras según cantidad de años ───────
    _anual_grav = (df_temp
        .groupby(["Año", "Gravedad_Indicador_Tradicional"])
        .size().reset_index(name="Accidentes"))

    if not _anual_grav.empty:
        _anual_grav["Gravedad_Indicador_Tradicional"] = pd.Categorical(
            _anual_grav["Gravedad_Indicador_Tradicional"],
            categories=["ILESO", "HERIDO", "MUERTO"], ordered=True)
        _anual_grav = _anual_grav.sort_values(["Año", "Gravedad_Indicador_Tradicional"])

        _n_annos = _anual_grav["Año"].nunique()

        if _n_annos < 2:
            # Con un solo año usamos barras agrupadas para evitar error en px.area
            fig_anual_grav = px.bar(
                _anual_grav,
                x="Año", y="Accidentes",
                color="Gravedad_Indicador_Tradicional",
                barmode="group",
                color_discrete_map={"ILESO": "#50FA7B", "HERIDO": "#FFB86C", "MUERTO": "#FF5555"},
                title="Accidentes por gravedad (año seleccionado)",
                template="plotly_dark",
                labels={"Gravedad_Indicador_Tradicional": "Gravedad", "Año": "Año"},
                category_orders={"Gravedad_Indicador_Tradicional": ["ILESO", "HERIDO", "MUERTO"]},
            )
        else:
            fig_anual_grav = px.area(
                _anual_grav,
                x="Año", y="Accidentes",
                color="Gravedad_Indicador_Tradicional",
                color_discrete_map={"ILESO": "#50FA7B", "HERIDO": "#FFB86C", "MUERTO": "#FF5555"},
                title="Evolución anual de accidentes por gravedad",
                template="plotly_dark",
                labels={"Gravedad_Indicador_Tradicional": "Gravedad", "Año": "Año"},
                category_orders={"Gravedad_Indicador_Tradicional": ["ILESO", "HERIDO", "MUERTO"]},
                markers=True,
            )
            fig_anual_grav.update_traces(opacity=0.75)
    else:
        fig_anual_grav = go.Figure().update_layout(
            title="Sin datos anuales", **_layout_vacio)

    fig_anual_grav.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font_color=TEXT, title_font_color=CYAN, height=420,
        xaxis=dict(dtick=1, gridcolor=BORDER, color=TEXT, tickformat="d"),
        yaxis=dict(gridcolor=BORDER, color=TEXT),
        legend=dict(bgcolor="rgba(13,17,23,0.88)", font=dict(color=TEXT, size=11),
                    bordercolor=BORDER, borderwidth=1),
        hovermode="x unified",
    )

    # ── Gráfico 2: heatmap hora × día ─────────────────────────
    _tmp = pd.DataFrame({
        "hourAcc":   df_temp["hourAcc"],
        "DiaSemana": df_temp["Fecha"].dt.dayofweek.map(_DIA_MAP),
    })
    _chd = _tmp.groupby(["hourAcc", "DiaSemana"]).size().reset_index(name="Accidentes")

    fig_hora_dia = px.density_heatmap(
        _chd, x="hourAcc", y="DiaSemana", z="Accidentes",
        histfunc="sum", color_continuous_scale="Viridis",
        title="Accidentes por hora y día de semana",
        template="plotly_dark",
        labels={"hourAcc": "Hora del día", "DiaSemana": "Día"},
        category_orders={"DiaSemana": _DIA_ORD},
    )
    fig_hora_dia.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
        title_font_color=CYAN, height=380, xaxis=dict(dtick=1),
    )

    # ── Gráfico 3: barras por mes ──────────────────────────────
    _cmes = (pd.DataFrame({"mes_num": df_temp["Fecha"].dt.month})
             .assign(mes_nom=lambda d: d["mes_num"].map(MESES_NOMBRE))
             .groupby(["mes_num", "mes_nom"]).size()
             .reset_index(name="Accidentes")
             .sort_values("mes_num"))

    fig_mes = px.bar(
        _cmes, x="mes_nom", y="Accidentes",
        title="Accidentes por mes", template="plotly_dark",
        color="Accidentes", color_continuous_scale="Blues",
        labels={"mes_nom": "Mes"},
    )
    fig_mes.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TEXT,
        title_font_color=CYAN, height=380, coloraxis_showscale=False,
    )

    # ── Gráfico 4: scatter burbujas por año ───────────────────
    _anual = (df_temp.groupby("Año").size()
              .reset_index(name="Accidentes")
              .sort_values("Año"))

    fig_scatter_anios = go.Figure()
    if not _anual.empty:
        if len(_anual) > 1:
            _x = _anual["Año"].values
            _y = _anual["Accidentes"].values
            _m, _b = np.polyfit(_x, _y, 1)
            fig_scatter_anios.add_trace(go.Scatter(
                x=_x, y=_m * _x + _b,
                mode="lines", name="Tendencia",
                line=dict(color=ORANGE, width=2, dash="dash"),
                hoverinfo="skip",
            ))
        fig_scatter_anios.add_trace(go.Scatter(
            x=_anual["Año"], y=_anual["Accidentes"],
            mode="markers+text", name="Accidentes / año",
            marker=dict(
                size=_anual["Accidentes"] / _anual["Accidentes"].max() * 36 + 10,
                color=_anual["Accidentes"], colorscale="Viridis", showscale=True,
                colorbar=dict(title="Acc.", thickness=12,
                              tickfont=dict(color=TEXT), title_font_color=TEXT),
                line=dict(color=BG, width=1),
            ),
            text=_anual["Accidentes"].apply(lambda v: f"{v:,}"),
            textposition="top center",
            textfont=dict(color=TEXT, size=10),
            hovertemplate="<b>%{x}</b><br>Accidentes: %{y:,}<extra></extra>",
        ))
    fig_scatter_anios.update_layout(
        title="Accidentalidad total por año",
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font_color=TEXT, title_font_color=CYAN, height=400,
        xaxis=dict(title="Año", dtick=1, gridcolor=BORDER, color=TEXT, tickangle=-45),
        yaxis=dict(title="Accidentes", gridcolor=BORDER, color=TEXT),
        legend=dict(bgcolor="rgba(13,17,23,0.88)", font=dict(color=TEXT, size=11),
                    bordercolor=BORDER, borderwidth=1),
        hovermode="x unified",
    )

    # ── Gráfico 5: correlación mensual precipitación ──────────
    _has_precip = "ValorObservado" in df_temp.columns
    df_corr = df_temp.copy()
    df_corr["mes_num"] = df_corr["Fecha"].dt.month
    df_corr["mes_nom"] = df_corr["mes_num"].map(MESES_NOMBRE)

    if _has_precip:
        _grp = (df_corr
                .groupby(["mes_num", "mes_nom"])
                .agg(Accidentes=("mes_num", "size"),
                     Precipitacion=("ValorObservado", "mean"))
                .reset_index().sort_values("mes_num"))
    else:
        _grp = (df_corr.groupby(["mes_num", "mes_nom"]).size()
                .reset_index(name="Accidentes").sort_values("mes_num"))
        _grp["Precipitacion"] = 0.0

    fig_corr = go.Figure()
    if not _grp.empty:
        fig_corr.add_trace(go.Bar(
            x=_grp["mes_nom"], y=_grp["Accidentes"],
            name="Accidentes", marker_color=ORANGE, opacity=0.75, yaxis="y1",
        ))
        if _has_precip:
            fig_corr.add_trace(go.Scatter(
                x=_grp["mes_nom"], y=_grp["Precipitacion"],
                name="Precipitación prom. (mm/h)",
                mode="lines+markers",
                line=dict(color=CYAN, width=2), marker=dict(size=5), yaxis="y2",
            ))
            if len(_grp) > 2:
                _corr_val = _grp[["Accidentes", "Precipitacion"]].corr().iloc[0, 1]
                fig_corr.add_annotation(
                    text=f"r de Pearson = {_corr_val:.3f}",
                    xref="paper", yref="paper", x=0.01, y=0.97,
                    showarrow=False,
                    font=dict(color=CYAN, size=13),
                    bgcolor="rgba(13,17,23,0.8)",
                    bordercolor=BORDER, borderpad=6,
                )
    fig_corr.update_layout(
        title="Correlación mensual: accidentalidad vs. precipitación promedio",
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font_color=TEXT, title_font_color=CYAN, height=400,
        yaxis=dict(title="Accidentes", gridcolor=BORDER, color=ORANGE),
        yaxis2=dict(title="Precipitación prom. (mm/h)", overlaying="y", side="right",
                    color=CYAN, showgrid=False),
        xaxis=dict(gridcolor=BORDER, color=TEXT,
                   categoryorder="array",
                   categoryarray=[MESES_NOMBRE[m] for m in range(1, 13)]),
        legend=dict(bgcolor="rgba(13,17,23,0.88)", font=dict(color=TEXT, size=11),
                    bordercolor=BORDER, borderwidth=1),
        barmode="overlay", hovermode="x unified",
    )

    return fig_anual_grav, fig_hora_dia, fig_mes, fig_scatter_anios, fig_corr


@app.callback(
    Output("grid-kpi", "style"),
    Input("tabs-main", "value"),
)
def forzar_render_grid(tab):
    return {"display": "grid", "gridTemplateColumns": "1fr 1fr",
            "gap": "16px", "padding": "8px 32px 32px"}


# ════════════════════════════════════════════════════════════
#  12. MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀  Servidor Dash →  http://127.0.0.1:8050\n")
    app.run(debug=True, port=8050)