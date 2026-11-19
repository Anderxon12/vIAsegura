"""
=============================================================
  EVALUAR MEDIDA — El "cerebro" que usará el chatbot
  Bogotá D.C.
=============================================================

Entrada: la medida ya estructurada (el JSON que produce el LLM extractor):
    {
      "tipo_medida": "limite_velocidad",
      "direccion": "reducir",
      "valor_actual": 50, "valor_propuesto": 30,
      "ubicacion": {"nivel": "localidad", "nombre": "Kennedy"},
      "acompanamiento": ["fotodeteccion"],
      "velocidad_media_actual": null
    }
    ubicacion.nivel puede ser: localidad · corredor (nombre en corredores.geojson)
                               · puntos (lista [[lon, lat], ...]) · celdas (lista de ids)

Salida: veredicto Positiva / Negativa / Neutra, cambio estimado en siniestros
con víctimas y en muertes (con rango), nivel de confianza, fuentes y
advertencias. El LLM de síntesis SOLO redacta a partir de esta salida.

Cómo decide
    1. Línea base: siniestros con víctimas y muertes de los últimos 24 meses
       en las celdas de la zona (llevado a un año).
    2. Evidencia local: efectos medidos en Bogotá para el mismo tipo de medida
       (out/efectos/efectos_catalogo.csv, filas revisadas).
    3. Evidencia de referencia: Nilsson / OMS (velocidad) o CMF.
    4. Combina 2 y 3 ponderando por la precisión de cada una.
    5. Veredicto: Positiva si el cambio central ≤ −5 % y el rango está casi
       todo bajo 0; Negativa si ≥ +5 % y casi todo sobre 0; si no, Neutra.

Uso
    python src/evaluar_medida.py --ejemplo
    python src/evaluar_medida.py medida.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factores_referencia as fr  # noqa: E402
from efecto_intervencion import (CORREDORES, cargar_panel, celdas_de_linea,  # noqa: E402
                                 celdas_de_localidad, celdas_de_puntos)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EFECTOS = os.path.join(RAIZ, "out", "efectos", "efectos_catalogo.csv")
UMBRAL = 0.05


# ════════════════════════════════════════════════════════════
#  ZONA Y LÍNEA BASE
# ════════════════════════════════════════════════════════════
def tipos_con_evidencia() -> set:
    """Tipos de medida con evidencia cargada: medida en Bogotá (efectos_catalogo.csv) o internacional (cmf.csv)."""
    tipos = {"limite_velocidad"}                     # siempre: modelo de potencia de Nilsson
    for ruta in (EFECTOS, os.path.join(RAIZ, "data", "referencia", "cmf.csv")):
        if os.path.exists(ruta):
            d = pd.read_csv(ruta)
            if "efecto_pct" in d:
                d = d[d["efecto_pct"].notna()]
            if "cmf" in d:
                d = d[d["cmf"].notna()]
            tipos |= set(d["tipo_medida"].dropna())
    return tipos


def nombre_canonico(texto: str) -> str:
    """'avenida boyaca', 'AK 72' → 'Av. Boyacá' (nombres de catalogo_resoluciones.CORREDORES)."""
    import re
    from catalogo_resoluciones import CORREDORES as PATRONES, plano
    t = plano(texto)
    for canon, pats in PATRONES.items():
        if plano(canon) == t or any(re.search(p, t) for p in pats):
            return canon
    return texto


def celdas_de_ubicacion(panel, ub: dict) -> tuple[list[str], str]:
    nivel = (ub or {}).get("nivel")
    if nivel == "localidad":
        return celdas_de_localidad(panel, ub["nombre"]), f"localidad {ub['nombre']}"
    if nivel == "celdas":
        return list(ub["ids"]), f"{len(ub['ids'])} celdas"
    if nivel == "puntos":
        return celdas_de_puntos(panel, ub["coordenadas"], ub.get("radio_m", 300)), f"{len(ub['coordenadas'])} puntos"
    if nivel == "corredor":
        nombre = nombre_canonico(ub["nombre"])
        if os.path.exists(CORREDORES):
            geo = json.load(open(CORREDORES, encoding="utf-8"))
            for f in geo["features"]:
                if f["properties"].get("corredor", "").lower() == nombre.lower():
                    g = f["geometry"]
                    partes = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
                    return sorted({c for p in partes for c in celdas_de_linea(panel, p, ub.get("ancho_m", 150))}), f"corredor {nombre}"
        return [], f"corredor {ub.get('nombre')} (sin geometría en corredores.geojson)"
    return [], "ubicación no reconocida"


def linea_base(panel, celdas, meses=24) -> dict:
    ult = panel["mes"].max()
    d = panel[panel["celda"].isin(celdas) & (panel["mes"] > ult - pd.DateOffset(months=meses))]
    f = 12 / max(d["mes"].nunique(), 1)
    return {"periodo": f"{(ult - pd.DateOffset(months=meses - 1)).strftime('%Y-%m')} a {ult.strftime('%Y-%m')}",
            "sin_victimas_anual": round(float(d["n_sin_victimas"].sum() * f), 1),
            "muertos_anual": round(float(d["n_muertos"].sum() * f), 1),
            "heridos_anual": round(float(d["n_heridos"].sum() * f), 1),
            "celdas": len(celdas)}


# ════════════════════════════════════════════════════════════
#  EVIDENCIA
# ════════════════════════════════════════════════════════════
def _se_desde_rango(c, inf, sup):
    """Error estándar en escala log a partir de un rango tipo IC95."""
    return max((np.log1p(sup) - np.log1p(inf)) / (2 * 1.96), 0.02)


def _num(v):
    try:
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


ORDEN_NIVELES = ["apilado_celdas", "apilado_vias", "grupo", "corredor"]
TITULOS = {"limite_velocidad": "límite de 50 km/h en corredores (2018–2019)",
           "zona_escolar": "30 km/h en zonas escolares (Decreto 073 de 2021)",
           "carril_preferencial": "carriles preferenciales SITP (2014–2017)",
           "fotodeteccion": "cámaras salvavidas (2019–2026)",
           "reductor_velocidad": "resaltos parabólicos (2024–2026)"}


def evidencia_local(m: dict, objetivo: str = "n_sin_victimas") -> tuple[list[dict], list[str]]:
    """Efectos medidos en Bogotá (out/efectos/efectos_catalogo.csv) para el mismo tipo de medida.

    Se usan las estimaciones por GRUPO cuando existen (más precisas) y, si no,
    las de cada corredor. Para límites de velocidad, el efecto medido se
    reescala al cambio de límite que se pregunta, suponiendo que el efecto en
    escala logarítmica es proporcional a los km/h de cambio (supuesto)."""
    tipo, direccion = m.get("tipo_medida"), m.get("direccion")
    if not os.path.exists(EFECTOS):
        return [], []
    e = pd.read_csv(EFECTOS)
    if "efecto_pct" not in e:
        return [], []
    e = e[(e["tipo_medida"] == tipo) & e["efecto_pct"].notna()]
    if "error" in e:
        e = e[e["error"].isna()]
    obj = e["objetivo"].fillna("n_sin_victimas") if "objetivo" in e else pd.Series("n_sin_victimas", index=e.index)
    e = e[obj == objetivo]
    # Una sola fuente de evidencia local, la mejor disponible: las estimaciones
    # apiladas ya contienen a las de cada grupo y corredor (usarlas juntas
    # contaría los mismos datos varias veces).
    if "nivel_estimacion" in e:
        for nivel in ORDEN_NIVELES:
            sub = e[e["nivel_estimacion"] == nivel]
            if len(sub):
                e = sub
                break
    escalar = tipo == "limite_velocidad" and _num(m.get("valor_actual")) and _num(m.get("valor_propuesto"))
    if not escalar and direccion and "direccion" in e:
        e = e[e["direccion"].isna() | (e["direccion"] == direccion)]
    out, notas = [], []
    for _, r in e.iterrows():
        ic = json.loads(r["ic95_pct"]) if isinstance(r["ic95_pct"], str) else r["ic95_pct"]
        cambio, inf, sup = r["efecto_pct"] / 100, ic[0] / 100, ic[1] / 100
        if str(r.get("nivel_estimacion", "")).startswith("apilado"):
            grupos = r.get("grupos")
            n = len(json.loads(grupos)) if isinstance(grupos, str) and grupos.startswith("[") else "varias"
            etiqueta = (f"Bogotá · {TITULOS.get(tipo, tipo)} · {n} fechas de implementación evaluadas juntas "
                        f"(antes/después con controles, {'celdas de 500 m' if 'celdas' in r['nivel_estimacion'] else 'sobre la avenida'})")
        else:
            lugar = next((str(x) for x in (r.get("corredor"), r.get("localidad")) if isinstance(x, str) and x.strip()), None)
            etiqueta = (f"Bogotá · {lugar} · desde {r.get('inicio')}" if lugar else
                        f"Bogotá · {TITULOS.get(r.get('tipo_medida'), r.get('tipo_medida'))} (antes/después con controles)")
        if escalar:
            d_med = _num(r.get("valor_nuevo")) - _num(r.get("valor_anterior")) if _num(r.get("valor_nuevo")) and _num(r.get("valor_anterior")) else None
            d_pre = float(m["valor_propuesto"]) - float(m["valor_actual"])
            if not d_med:
                continue
            k = d_pre / d_med
            cambio, inf, sup = np.expm1(k * np.log1p(cambio)), np.expm1(k * np.log1p(inf)), np.expm1(k * np.log1p(sup))
            inf, sup = min(inf, sup), max(inf, sup)
            if abs(k - 1) > 1e-9:
                etiqueta += f" (medido para {r.get('valor_anterior')}→{r.get('valor_nuevo')} km/h, reescalado ×{k:.2f})"
                notas.append("El efecto medido en Bogotá se reescaló al cambio de límite preguntado, suponiendo "
                             "un efecto proporcional a los km/h de cambio.")
        out.append({"fuente": etiqueta, "cambio": float(cambio), "rango": [float(inf), float(sup)], "tipo": "local"})
    return out, sorted(set(notas))


def evidencia_referencia(m: dict, proporciones: dict | None = None) -> tuple[list[dict], list[str], list[dict]]:
    """Evidencia internacional: (para siniestros con víctimas, supuestos, para siniestros fatales)."""
    tipo = m.get("tipo_medida")
    if tipo == "limite_velocidad" and m.get("valor_actual") and m.get("valor_propuesto"):
        con_control = bool(set(m.get("acompanamiento") or []) & {"fotodeteccion", "control_policial"})
        r = fr.efecto_limite(float(m["valor_actual"]), float(m["valor_propuesto"]),
                             m.get("velocidad_media_actual"), con_control)
        ev = {"fuente": r["fuente"], "cambio": r["cambio_sin_victimas"], "rango": r["rango"], "tipo": "referencia"}
        fatal = {"fuente": r["fuente"], "cambio": r["cambio_fatales"], "rango": r["rango_fatales"], "tipo": "referencia"}
        return [ev], r["supuestos"], [fatal]
    evs, notas = fr.efecto_cmf(tipo, proporciones, "sin_victimas")
    evs_f, _ = fr.efecto_cmf(tipo, proporciones, "fatales")
    return evs, notas, evs_f


def proporciones_zona(panel, celdas, meses=24) -> dict:
    """Qué parte de los siniestros con víctimas de la zona son atropellos, con ciclistas, motos o alcohol."""
    ult = panel["mes"].max()
    d = panel[panel["celda"].isin(celdas) & (panel["mes"] > ult - pd.DateOffset(months=meses))]
    total = max(float(d["n_sin_victimas"].sum()), 1.0)
    col = {"peaton": "n_vict_peaton", "ciclista": "n_vict_ciclista", "moto": "n_vict_moto", "embriaguez": "n_vict_embriaguez"}
    return {k: float(d[c].sum()) / total for k, c in col.items() if c in d}


def combinar(evs: list[dict]) -> dict | None:
    """Promedio ponderado por precisión (inverso de la varianza) en escala log."""
    if not evs:
        return None
    b = np.array([np.log1p(e["cambio"]) for e in evs])
    se = np.array([_se_desde_rango(e["cambio"], *e["rango"]) for e in evs])
    w = 1 / se ** 2
    est, se_c = float(np.sum(w * b) / w.sum()), float(np.sqrt(1 / w.sum()))
    # heterogeneidad: si los estudios no concuerdan, se ensancha el rango
    if len(evs) > 1:
        q = float(np.sum(w * (b - est) ** 2))
        tau2 = max(0.0, (q - (len(evs) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        se_c = float(np.sqrt(se_c ** 2 + tau2))
    return {"cambio": np.expm1(est), "rango": [np.expm1(est - 1.96 * se_c), np.expm1(est + 1.96 * se_c)]}


def _signo(c: float | None, rango: list[float] | None) -> int:
    """+1 reduce claramente, −1 aumenta claramente, 0 no concluyente."""
    if c is None or rango is None:
        return 0
    inf, sup = rango
    if c <= -UMBRAL and sup < UMBRAL / 2:
        return 1
    if c >= UMBRAL and inf > -UMBRAL / 2:
        return -1
    return 0


def veredicto(c: float, rango: list[float], c_muertes: float | None = None,
              rango_muertes: list[float] | None = None) -> tuple[str, str]:
    """Positiva si reduce claramente los siniestros con víctimas o las muertes, sin
    aumentar claramente lo otro; Negativa al revés; Neutra en otro caso."""
    s_sin, s_mue = _signo(c, rango), _signo(c_muertes, rango_muertes)
    if s_sin == 1 and s_mue >= 0:
        return "Positiva", "reduce los siniestros con víctimas"
    if s_mue == 1 and s_sin >= 0:
        return "Positiva", "reduce las muertes, aunque el cambio en el número de siniestros no es concluyente" \
            if s_sin == 0 else "reduce las muertes"
    if s_sin == -1 and s_mue <= 0:
        return "Negativa", "aumenta los siniestros con víctimas"
    if s_mue == -1 and s_sin <= 0:
        return "Negativa", "aumenta las muertes"
    if s_sin and s_mue:
        return "Neutra", "tiene efectos en sentidos opuestos sobre siniestros y muertes"
    return "Neutra", "no muestra un cambio claro ni en siniestros ni en muertes"


# ════════════════════════════════════════════════════════════
#  PRINCIPAL
# ════════════════════════════════════════════════════════════
def evaluar(m: dict, panel: pd.DataFrame | None = None) -> dict:
    panel = cargar_panel() if panel is None else panel
    celdas, zona = celdas_de_ubicacion(panel, m.get("ubicacion"))
    res = {"medida": m, "zona": zona, "advertencias": []}
    if not celdas:
        res.update(veredicto="Sin evaluar", confianza="baja",
                   advertencias=[f"No se pudo ubicar la zona ({zona}). Pida al usuario una localidad, corredor o dirección."])
        return res
    base = linea_base(panel, celdas)
    res["linea_base"] = base

    locales, notas_locales = evidencia_local(m)
    if m.get("entorno_escolar") and m.get("tipo_medida") != "zona_escolar":
        esc, notas_esc = evidencia_local({**m, "tipo_medida": "zona_escolar"})   # 30 km/h escolar medido en Bogotá
        locales, notas_locales = locales + esc, notas_locales + notas_esc
    prop = proporciones_zona(panel, celdas)
    res["proporciones"] = {k: round(v, 3) for k, v in prop.items()}
    refs, supuestos, refs_fatal = evidencia_referencia(m, prop)
    supuestos = supuestos + notas_locales
    res["evidencia"] = locales + refs
    res["supuestos"] = supuestos
    comb = combinar(locales + refs)
    if comb is None:
        res.update(veredicto="Neutra", confianza="baja", cambio_pct=None,
                   advertencias=["No hay evidencia local ni de referencia para este tipo de medida. "
                                 "El agente debe decirlo y no inventar un porcentaje."])
        return res

    c, rg = comb["cambio"], comb["rango"]
    res["cambio_sin_victimas_pct"] = round(100 * c, 1)
    res["rango_pct"] = [round(100 * rg[0], 1), round(100 * rg[1], 1)]
    res["siniestros_victimas_evitados_anual"] = round(-c * base["sin_victimas_anual"], 1)
    locales_fatal, _ = evidencia_local(m, "n_sin_fatales")
    evs_fatal = locales_fatal + refs_fatal
    comb_f = combinar(evs_fatal)
    if comb_f:
        res["cambio_muertes_pct"] = round(100 * comb_f["cambio"], 1)
        res["rango_muertes_pct"] = [round(100 * comb_f["rango"][0], 1), round(100 * comb_f["rango"][1], 1)]
        res["muertes_evitadas_anual"] = round(-comb_f["cambio"] * base["muertos_anual"], 1)
        res["evidencia_muertes"] = evs_fatal
    res["veredicto"], res["motivo_veredicto"] = veredicto(
        c, rg, comb_f["cambio"] if comb_f else None, comb_f["rango"] if comb_f else None)

    ancho = rg[1] - rg[0]
    n_local = len(locales)
    # alta: varios casos medidos en Bogotá y rango estrecho · media: algo de evidencia y rango razonable
    res["confianza"] = ("alta" if n_local >= 2 and ancho < 0.20 else
                        "media" if (n_local >= 1 or (refs and not supuestos)) and ancho < 0.40 else "baja")
    if n_local == 0:
        res["advertencias"].append("No hay casos medidos en Bogotá para esta medida: el estimado usa solo evidencia internacional.")
    if base["sin_victimas_anual"] < 20:
        res["advertencias"].append("La zona tiene pocos siniestros al año: el número de siniestros evitados es muy incierto.")
    res["advertencias"] += [f"Supuesto: {s}" for s in supuestos]
    return res


EJEMPLO = {"tipo_medida": "limite_velocidad", "direccion": "reducir", "valor_actual": 50, "valor_propuesto": 30,
           "ubicacion": {"nivel": "localidad", "nombre": "Kennedy"}, "acompanamiento": [],
           "velocidad_media_actual": None}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--ejemplo":
        medida = json.load(open(sys.argv[1], encoding="utf-8"))
    else:
        medida = EJEMPLO
    print(json.dumps(evaluar(medida), ensure_ascii=False, indent=1, default=str))
