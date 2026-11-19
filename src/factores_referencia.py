"""
=============================================================
  FACTORES DE REFERENCIA — Evidencia internacional
  (se usa cuando Bogotá no tiene un caso parecido)
=============================================================

1. Modelo de potencia de Nilsson (2004)
       siniestros_nuevos / siniestros_actuales = (v_nueva / v_actual) ^ k
       k = 2 siniestros con heridos · 3 con heridos graves o muertos · 4 fatales
   v = velocidad MEDIA del tráfico, no el límite.

2. OMS, Save LIVES (2017): +1 km/h de velocidad media ≈ +3 % siniestros
   con heridos y +4 a 5 % siniestros fatales. Se usa como contraste.

3. Del límite a la velocidad media
   Bajar el límite no baja la velocidad media en la misma cantidad. El
   parámetro TRASLADO_LIMITE dice qué fracción del cambio de límite llega a
   la velocidad media. Es un SUPUESTO: debe calibrarse con los datos de
   velocidad de la SDM (velocidad_via) antes y después de 2018–2019, y se
   refuerza si hay control (cámaras o agentes).

4. Factores de modificación (CMF) para otras medidas
   data/referencia/cmf.csv se llena a mano desde el CMF Clearinghouse
   (https://www.cmfclearinghouse.org) u otra revisión sistemática, una fila
   por estudio, con su cita. No se incluyen valores inventados: si la tabla
   está vacía para un tipo de medida, el agente dice que no hay evidencia.
"""

from __future__ import annotations

import os

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_CMF = os.path.join(RAIZ, "data", "referencia", "cmf.csv")
COLUMNAS_CMF = ["tipo_medida", "descripcion", "cmf", "cmf_ic_inf", "cmf_ic_sup", "severidad", "aplica_a",
                "tipo_via", "pais", "estrellas_calidad", "fuente", "url", "nota"]
# Si un estudio no publica intervalo, se asume una incertidumbre de ±50 % del efecto
# (en escala logarítmica). Es un supuesto explícito, no un dato del estudio.
INCERTIDUMBRE_SIN_IC = 0.5
NOMBRES_GRUPO = {"peaton": "atropellos", "ciclista": "siniestros con ciclistas", "embriaguez": "siniestros con alcohol",
                 "moto": "siniestros con motociclistas"}

EXPONENTES_NILSSON = {"heridos": 2.0, "graves_o_muertos": 3.0, "fatales": 4.0}
OMS_POR_KMH = {"heridos": 0.03, "fatales": 0.045}
TRASLADO_LIMITE = {"sin_control": 0.25, "con_control": 0.5}   # SUPUESTO — calibrar


def velocidad_media_nueva(v_media_actual: float, limite_actual: float, limite_nuevo: float,
                          con_control: bool = False) -> float:
    """Velocidad media esperada tras cambiar el límite (supuesto de traslado parcial)."""
    f = TRASLADO_LIMITE["con_control" if con_control else "sin_control"]
    return v_media_actual + f * (limite_nuevo - limite_actual)


def efecto_velocidad(v_actual: float, v_nueva: float) -> dict:
    """Cambio relativo esperado (proporción, −0.2 = −20 %) por severidad."""
    r = v_nueva / v_actual
    nilsson = {k: r ** e - 1 for k, e in EXPONENTES_NILSSON.items()}
    dv = v_nueva - v_actual
    oms = {k: (1 + t) ** dv - 1 for k, t in OMS_POR_KMH.items()}
    return {"nilsson": nilsson, "oms": oms, "v_actual": v_actual, "v_nueva": v_nueva}


def efecto_limite(limite_actual: float, limite_nuevo: float, v_media_actual: float | None = None,
                  con_control: bool = False) -> dict:
    """Efecto esperado de cambiar el límite de velocidad, con un rango de supuestos.

    Si no se conoce la velocidad media actual se supone igual al 90 % del
    límite actual y se marca como supuesto."""
    supuestos = []
    if v_media_actual is None:
        v_media_actual = 0.9 * limite_actual
        supuestos.append("velocidad media actual supuesta = 90 % del límite (sin dato medido)")
    escenarios = {}
    for nombre, traslado in (("bajo", 0.15), ("central", TRASLADO_LIMITE["con_control" if con_control else "sin_control"]),
                             ("alto", 0.6 if con_control else 0.4)):
        v1 = v_media_actual + traslado * (limite_nuevo - limite_actual)
        escenarios[nombre] = efecto_velocidad(v_media_actual, v1)
    supuestos.append(f"traslado del límite a la velocidad media: {TRASLADO_LIMITE} (supuesto, calibrar con datos SDM)")
    central = escenarios["central"]["nilsson"]["heridos"]
    # rango: extremos entre escenarios y entre Nilsson / OMS
    valores = [e["nilsson"]["heridos"] for e in escenarios.values()] + [e["oms"]["heridos"] for e in escenarios.values()]
    return {"cambio_sin_victimas": central, "rango": [min(valores), max(valores)],
            "cambio_fatales": escenarios["central"]["nilsson"]["fatales"],
            "rango_fatales": sorted([escenarios["bajo"]["nilsson"]["fatales"], escenarios["alto"]["nilsson"]["fatales"]]),
            "escenarios": escenarios, "supuestos": supuestos, "fuente": "Nilsson (2004); OMS Save LIVES (2017)"}


def cargar_cmf() -> pd.DataFrame:
    if not os.path.exists(RUTA_CMF):
        os.makedirs(os.path.dirname(RUTA_CMF), exist_ok=True)
        pd.DataFrame(columns=COLUMNAS_CMF).to_csv(RUTA_CMF, index=False, encoding="utf-8-sig")
    return pd.read_csv(RUTA_CMF)


def efecto_cmf(tipo_medida: str, proporciones: dict | None = None, objetivo: str = "sin_victimas") -> tuple[list[dict], list[str]]:
    """Una evidencia por estudio de data/referencia/cmf.csv para ese tipo de medida.

    objetivo: "sin_victimas" (siniestros con heridos o muertos) o "fatales".
    proporciones: qué parte de los siniestros con víctimas de la zona son atropellos,
    con ciclistas o con alcohol. Un factor que solo actúa sobre atropellos cambia el
    total en (proporción de atropellos) × (factor − 1)."""
    import numpy as np
    t = cargar_cmf()
    if t.empty or "tipo_medida" not in t:
        return [], []
    t = t[t["tipo_medida"] == tipo_medida].dropna(subset=["cmf"])
    sev = t["severidad"].fillna("todas") if "severidad" in t else pd.Series("todas", index=t.index)
    validas = ("fatal_y_heridos", "todas", "fatal") if objetivo == "fatales" else ("fatal_y_heridos", "todas", "heridos", "fatal")
    t = t[sev.isin(validas)]
    evs, notas = [], set()
    for _, r in t.iterrows():
        cmf = float(r["cmf"])
        inf, sup = r.get("cmf_ic_inf"), r.get("cmf_ic_sup")
        if pd.isna(inf) or pd.isna(sup):
            b = np.log(cmf)
            inf, sup = float(np.exp(b * (1 + INCERTIDUMBRE_SIN_IC))), float(np.exp(b * (1 - INCERTIDUMBRE_SIN_IC)))
            inf, sup = min(inf, sup), max(inf, sup)
            notas.add("Algunos estudios no publican intervalo: se asumió una incertidumbre de ±50 % del efecto.")
        aplica = str(r.get("aplica_a") or "todos")
        fraccion = 1.0
        if aplica != "todos":
            fraccion = float((proporciones or {}).get(aplica, 0.0))
            notas.add(f"El efecto de los estudios sobre {NOMBRES_GRUPO.get(aplica, aplica)} se aplicó solo a esa parte "
                      f"de los siniestros de la zona ({100 * fraccion:.0f} %).")
        if r.get("severidad") == "fatal" and objetivo != "fatales":
            notas.add("Un estudio mide muertes; se usó también como aproximación para siniestros con heridos.")
        cambio = fraccion * (cmf - 1)
        rango = [fraccion * (float(inf) - 1), fraccion * (float(sup) - 1)]
        evs.append({"fuente": f"{r['descripcion']}: {r['fuente']}", "cambio": cambio, "rango": rango,
                    "tipo": "referencia", "nota": r.get("nota") if isinstance(r.get("nota"), str) else ""})
    return evs, sorted(notas)


if __name__ == "__main__":
    import json
    print(json.dumps(efecto_limite(50, 30), indent=1, default=lambda x: round(x, 3)))
