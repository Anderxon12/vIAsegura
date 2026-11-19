"""
=============================================================
  VÍASEGURA — Puente con los modelos de vIAsegura_ML
=============================================================

Carga el paquete entrenado por el proyecto de ML (Grupo3_Modelo_Final.pkl,
generado por vIAsegura_ML/entrega/Grupo3_ModelamientoEvaluacion.ipynb) y lo
expone a la app web, con la misma lógica de la demo (Grupo3_Demo.ipynb):

    1. Gravedad        Random Forest calibrado: probabilidad de que un siniestro
                       con víctimas sea fatal, nivel de riesgo y factores que pesan.
    2. Riesgo por tramo LightGBM Poisson: siniestros con víctimas esperados en
                       2026 por segmento vial (CIV) y simulación de dispositivos.
    3. Perfiles        K-Means (12 perfiles): a qué tipo de siniestro se parece.
    +  Normas          Motor de similitud temática: norma y artículo aplicables.

No reentrena nada. Dónde busca el modelo (el primero que exista):
    1. variable de entorno VIASEGURA_MODELO_ML
    2. <app>/modelos_ml/Grupo3_Modelo_Final.pkl          (copia para el servidor)
    3. <proyecto>/vIAsegura_ML/entrega/Grupo3_Modelo_Final.pkl   (proyecto fusionado)

Versiones con que se guardó el modelo: scikit-learn 1.7.2 y lightgbm 4.6.0.
"""

from __future__ import annotations

import json
import os
import threading
import warnings

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # vIAsegura_IA
PROYECTO = os.path.dirname(RAIZ)                                            # vIAsegura_proyecto
NOMBRE_PKL = "Grupo3_Modelo_Final.pkl"
CANDIDATOS = [os.path.join(RAIZ, "modelos_ml", NOMBRE_PKL),
              os.path.join(PROYECTO, "vIAsegura_ML", "entrega", NOMBRE_PKL)]

CAUSAS_VALIDAS = ["velocidad", "embriaguez", "semaforo", "senales", "prelacion", "distancia", "adelantar_invadir",
                  "moto_zigzag", "peaton", "estado_via", "falla_vehiculo", "visibilidad_clima",
                  "distraccion_impericia", "transporte_pasajeros"]
NOMBRES_CAUSA = {
    "velocidad": "exceso de velocidad", "embriaguez": "embriaguez o drogas", "semaforo": "pasarse el semáforo",
    "senales": "desobedecer señales", "prelacion": "no respetar prelación", "distancia": "no mantener distancia",
    "adelantar_invadir": "adelantar o invadir carril", "moto_zigzag": "moto en zigzag",
    "peaton": "conducta del peatón", "estado_via": "mal estado de la vía", "falla_vehiculo": "falla del vehículo",
    "visibilidad_clima": "visibilidad o clima", "distraccion_impericia": "distracción o impericia",
    "transporte_pasajeros": "maniobra de transporte de pasajeros",
}
CLASES = ["CHOQUE", "ATROPELLO", "VOLCAMIENTO", "CAIDA DE OCUPANTE", "OTRO"]

# Valor "neutro" de cada factor: con qué se compara para medir cuánto aporta (igual que la demo)
NEUTROS = {"hora": 14, "clase_siniestro": "CHOQUE", "mayores_60": 0, "menores_18": 0, "peatones": 0, "motociclistas": 0,
           "ciclistas": 0, "motos": 0, "carga": 0, "buses": 0, "via_arterial": False, "iluminacion": "CON",
           "condicion": "SECA", "estado": "BUENO", "fuga": False, "interseccion": False, "velocidad_reglamentaria": 30,
           "edad_conductor_min": 40, "camara_activa_500m": False, "resalto": False, "semaforo_a_30m": False}
NOMBRES_FACTOR = {
    "hora": "hora", "clase_siniestro": "clase", "mayores_60": "personas mayores de 60", "menores_18": "menores de 18",
    "peatones": "peatones", "motociclistas": "motociclistas", "ciclistas": "ciclistas", "motos": "motos",
    "carga": "vehículos de carga", "buses": "buses", "via_arterial": "vía arterial", "iluminacion": "iluminación",
    "condicion": "condición de la vía", "estado": "estado de la vía", "fuga": "fuga", "interseccion": "intersección",
    "velocidad_reglamentaria": "velocidad reglamentaria", "edad_conductor_min": "edad del conductor más joven",
    "camara_activa_500m": "cámara activa a 500 m", "resalto": "resalto", "semaforo_a_30m": "semáforo a 30 m",
}

# Ejemplos de la demo del proyecto de ML (Grupo3_Demo.ipynb, sección 4)
EJEMPLOS = {
    "Atropello nocturno de una persona mayor en vía arterial": {
        "fecha": "2026-11-14", "hora": 22, "clase_siniestro": "ATROPELLO", "localidad": "KENNEDY",
        "lat": 4.628, "lon": -74.152, "peatones": 1, "conductores": 1, "mayores_60": 1, "edad_conductor_min": 34,
        "autos": 1, "causas": ["velocidad"], "via_arterial": True, "interseccion": False, "iluminacion": "SIN",
        "condicion": "SECA", "estado": "BUENO", "velocidad_reglamentaria": 60},
    "Choque entre dos carros en una intersección, de día": {
        "fecha": "2026-11-12", "hora": 15, "clase_siniestro": "CHOQUE", "localidad": "CHAPINERO",
        "lat": 4.651, "lon": -74.060, "conductores": 2, "pasajeros": 1, "edad_conductor_min": 41, "autos": 2,
        "causas": ["distancia"], "via_arterial": False, "interseccion": True, "iluminacion": "CON", "condicion": "SECA",
        "estado": "BUENO", "velocidad_reglamentaria": 30, "semaforo_a_30m": True},
    "Motociclista en la madrugada, con embriaguez y vía húmeda": {
        "fecha": "2026-11-15", "hora": 3, "clase_siniestro": "CHOQUE", "localidad": "ENGATIVA",
        "lat": 4.700, "lon": -74.110, "motociclistas": 1, "conductores": 1, "edad_conductor_min": 22, "motos": 1,
        "causas": ["embriaguez", "velocidad"], "via_arterial": True, "iluminacion": "CON", "condicion": "HUMEDA",
        "estado": "BUENO", "velocidad_reglamentaria": 60, "camara_activa_500m": False},
}

# Cambios que se pueden simular en un tramo (columnas del modelo de riesgo por tramo)
ESCENARIOS = {"camara": "con cámara salvavidas", "resalto": "con resalto parabólico",
              "semaforo": "con semáforo", "vel30": "con límite de 30 km/h"}


def ruta_modelo() -> str | None:
    env = os.environ.get("VIASEGURA_MODELO_ML", "").strip()
    for r in ([env] if env else []) + CANDIDATOS:
        if r and os.path.exists(r):
            return r
    return None


def localidad_modelo(nombre: str | None) -> str:
    """«Ciudad Bolívar» → «CIUDAD BOLIVAR» (mayúsculas sin tildes, pero con Ñ: así viene en la base de la SDM)."""
    s = str(nombre or "SIN INFO").strip().upper()
    for a, b in zip("ÁÉÍÓÚÜ", "AEIOUU"):
        s = s.replace(a, b)
    return s or "SIN INFO"


def _f(x, nd=4):
    """Número de numpy → float de Python redondeado (o None) para JSON."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else round(x, nd)


def _reparar_bsplines(obj) -> int:
    """Adapta los BSpline guardados con scipy <= 1.16 a scipy 1.17.

    scipy 1.17 guarda los nudos y coeficientes en `_t` / `_c` y lee `t` / `c` por medio de
    `self._asarray`. Un BSpline viejo sale del .pkl con `t` / `c` en su __dict__ y sin `_asarray`,
    y SplineTransformer falla con «'BSpline' object has no attribute '_asarray'». Con scipy
    <= 1.16 esto no cambia nada. Devuelve cuántos objetos reparó.
    """
    try:
        from scipy.interpolate import BSpline
    except ImportError:
        return 0
    if not isinstance(BSpline.__dict__.get("t"), property):          # scipy <= 1.16: nada que hacer
        return 0
    vistos, pila, n = set(), [obj], 0
    while pila:
        o = pila.pop()
        if id(o) in vistos or isinstance(o, (str, bytes, int, float, bool, type(None), np.ndarray,
                                                 pd.DataFrame, pd.Series, pd.Index)):
            continue
        vistos.add(id(o))
        if isinstance(o, BSpline):
            d = o.__dict__
            if "_asarray" not in d:
                if "t" in d:
                    d["_t"] = np.ascontiguousarray(d.pop("t"), dtype=np.float64)
                if "c" in d:
                    d["_c"] = np.asarray(d.pop("c"))
                d["_asarray"] = np.asarray
                n += 1
            continue
        if isinstance(o, dict):
            pila.extend(o.values())
        elif isinstance(o, (list, tuple, set)):
            pila.extend(o)
        elif hasattr(o, "__dict__") and not isinstance(o, type):
            pila.extend(vars(o).values())
    return n


class ModelosML:
    """Envuelve el paquete del modelo. Se carga en segundo plano para no demorar el arranque de la app."""

    def __init__(self):
        self.paquete, self.error, self.ruta, self.aviso = None, None, ruta_modelo(), ""
        self.listo = threading.Event()
        self._malla, self._lock_malla = None, threading.Lock()
        threading.Thread(target=self._cargar, daemon=True).start()

    # ─────────────── carga ───────────────
    def _cargar(self):
        try:
            if not self.ruta:
                raise FileNotFoundError(f"no encuentro {NOMBRE_PKL} (búsquedas: {', '.join(CANDIDATOS)}). "
                                        "Genérelo con vIAsegura_ML/entrega/Grupo3_ModelamientoEvaluacion.ipynb "
                                        "o indique la ruta en la variable VIASEGURA_MODELO_ML.")
            import joblib
            from importlib.metadata import version
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter("always")
                p = joblib.load(self.ruta)
            _reparar_bsplines(p)
            esperadas = p["metadatos"].get("versiones", {})
            distintas = {k: version(k) for k in ("scikit-learn", "lightgbm")
                         if esperadas.get(k) and version(k) != esperadas[k]}
            if distintas:
                self.aviso = ("El modelo se guardó con " + ", ".join(f"{k} {esperadas[k]}" for k in distintas)
                              + " y está instalado " + ", ".join(f"{k} {v}" for k, v in distintas.items())
                              + ": instale requirements_app.txt para resultados idénticos a la demo.")
            elif avisos:
                self.aviso = str(avisos[0].message)[:200]
            self.G, self.T, self.P, self.N, self.META = (p["gravedad"], p["riesgo_tramos"], p["perfiles"],
                                                          p["normas"], p["metadatos"])
            self.pred = self.T["prediccion_2026"].reset_index(drop=True)
            self.paquete = p
            print(f"  modelos ML: {self.G['algoritmo']} · {self.T['algoritmo']} · {self.P['algoritmo']} "
                  f"({os.path.relpath(self.ruta, PROYECTO)})" + (f" · ⚠ {self.aviso}" if self.aviso else ""))
        except ImportError as e:
            self.error = (f"falta la librería {e.name} para los modelos de ML. "
                          "Instale: pip install -r requirements_app.txt")
        except Exception as e:
            import traceback
            traceback.print_exc()                     # queda completo en el log (p. ej. en Render)
            self.error = f"{type(e).__name__}: {e}"
            if "unpack" in str(e) or "setstate" in traceback.format_exc():
                self.error += (" (versión de librería incompatible con el .pkl: instale requirements_app.txt; "
                               "scipy debe ser < 1.18)")
        finally:
            if self.error:
                print(f"  modelos ML: no disponibles — {self.error}")
            self.listo.set()

    @property
    def disponible(self) -> bool:
        return self.paquete is not None

    # ─────────────── 1. gravedad (clasificación) ───────────────
    def preparar(self, s: dict) -> pd.DataFrame:
        """Descripción sencilla de un siniestro → la fila de variables que usa el modelo (igual que la demo)."""
        G, META = self.G, self.META
        v = lambda k, d=np.nan: s.get(k, d) if s.get(k) not in (None, "") else d
        fecha = pd.Timestamp(v("fecha", pd.Timestamp.today().normalize()))
        hora = v("hora")
        hora = float(hora) if not pd.isna(hora) else np.nan
        personas = {k: float(v(k, 0)) for k in ("peatones", "ciclistas", "motociclistas", "pasajeros", "conductores",
                                                 "menores_18", "mayores_60")}
        vehiculos = {k: float(v(k, 0)) for k in ("autos", "motos", "bicicletas", "buses", "carga", "otros")}
        n_veh = sum(vehiculos.values())
        causas = [c for c in v("causas", []) if c in CAUSAS_VALIDAS]
        arterial = bool(v("via_arterial", False))
        vel = v("velocidad_reglamentaria")
        vel = float(vel) if not pd.isna(vel) else np.nan
        clase = str(v("clase_siniestro", "CHOQUE")).upper()
        edad = v("edad_conductor_min")
        fila = {
            "mes": fecha.month, "dia_semana": fecha.dayofweek, "fin_de_semana": int(fecha.dayofweek >= 5), "hora": hora,
            "nocturno": int(hora >= 18 or hora <= 5) if not pd.isna(hora) else np.nan,
            "clase_siniestro": clase, "localidad": localidad_modelo(v("localidad", "SIN INFO")),
            "lat": v("lat"), "lon": v("lon"),
            "act_n_peatones": personas["peatones"], "act_n_ciclistas": personas["ciclistas"],
            "act_n_motociclistas": personas["motociclistas"], "act_n_pasajeros": personas["pasajeros"],
            "act_n_conductores": personas["conductores"], "act_n_menores18": personas["menores_18"],
            "act_n_mayores60": personas["mayores_60"],
            "act_n_total": personas["peatones"] + personas["ciclistas"] + personas["motociclistas"]
                           + personas["pasajeros"] + personas["conductores"],
            "act_edad_conductor_min": edad,
            "act_conductor_joven": int(float(v("edad_conductor_min", 99)) < 25),
            "tema_motociclistas": int(vehiculos["motos"] > 0 or personas["motociclistas"] > 0),
            "tema_peatones": int(personas["peatones"] > 0 or clase == "ATROPELLO"),
            "tema_ciclistas": int(vehiculos["bicicletas"] > 0 or personas["ciclistas"] > 0),
            "tema_transporte_publico": int(vehiculos["buses"] > 0), "tema_vehiculos_carga": int(vehiculos["carga"] > 0),
            "tema_menores": int(personas["menores_18"] > 0),
            "veh_n_liviano": vehiculos["autos"], "veh_n_moto": vehiculos["motos"],
            "veh_n_bicicleta": vehiculos["bicicletas"], "veh_n_bus": vehiculos["buses"],
            "veh_n_carga": vehiculos["carga"], "veh_n_otro": vehiculos["otros"],
            "veh_n_total": n_veh, "veh_n_servicio_publico": float(v("servicio_publico", 0)), "veh_n_oficial": 0.0,
            "veh_hay_fuga": int(bool(v("fuga", False))), "veh_solo_un_vehiculo": int(n_veh == 1),
            **{f"causa_{c}": int(c in causas) for c in CAUSAS_VALIDAS},
            "causa_n_hipotesis": len(causas), "causa_solo_otra": 0,
            "via_iluminacion": str(v("iluminacion", "SIN INFO")).upper(),
            "via_sin_iluminacion": int(str(v("iluminacion", "")).upper() == "SIN"),
            "via_condicion": str(v("condicion", "SIN INFO")).upper(),
            "via_humeda": int(str(v("condicion", "")).upper() in ("HUMEDA", "LODO", "ACEITE", "MATERIAL SUELTO")),
            "via_estado": str(v("estado", "SIN INFO")).upper(),
            "via_mal_estado": int(str(v("estado", "BUENO")).upper() not in ("BUENO", "SIN INFO")),
            "via_n_vias": 2 if v("interseccion", False) else 1, "via_interseccion": int(bool(v("interseccion", False))),
            "via_semaforo_informe": int(bool(v("semaforo_a_30m", False))),
            "malla_es_arterial": int(arterial), "malla_tipo_via": "AK" if arterial else "CL",
            "malla_vel_reglamentaria": vel,
            "vel_limite_vigente": (50 if vel == 60 and fecha >= pd.Timestamp("2020-05-12") else vel),
            "regimen_50kmh": int(fecha >= pd.Timestamp("2020-05-12")),
            "sem_a_30m": int(bool(v("semaforo_a_30m", False))),
            "cam_activa_500m": int(bool(v("camara_activa_500m", False))),
            "resalto_activo": int(bool(v("resalto", False))),
            # contexto (no son variables del modelo; los usan perfiles y normas)
            "fecha": fecha, "con_victimas": 1, "es_fatal": 0,
        }
        df = pd.DataFrame([fila])
        for c in G["variables"]:                       # lo que no se informó → sin información
            if c not in df:
                df[c] = "SIN INFO" if c in META["variables_categoricas"] else np.nan
        for c in G["variables"]:
            if c in META["variables_categoricas"]:
                df[c] = df[c].astype(object)
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        return df

    def predecir_gravedad(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidad calibrada de que el siniestro con víctimas sea fatal."""
        G = self.G
        crudo = G["modelo"].predict_proba(df[G["variables"]])[:, 1]
        if G["metodo_calibracion"] == "platt":
            c = np.clip(crudo, 1e-4, 1 - 1e-4)
            return G["calibrador"].predict_proba(np.log(c / (1 - c)).reshape(-1, 1))[:, 1]
        return G["calibrador"].predict(crudo)

    def nivel_riesgo(self, p: float) -> str:
        G = self.G
        if p >= 2 * G["umbral"]:
            return "MUY ALTO"
        if p >= G["umbral"]:
            return "ALTO"
        if p >= G["tasa_base"] / 2:
            return "MEDIO"
        return "BAJO"

    def explicar(self, s: dict, p0: float) -> list[dict]:
        """Aporte de cada factor informado frente a un valor neutro (positivo = sube el riesgo).
        Todas las variantes se predicen en un solo lote."""
        variantes, etiquetas = [], []
        for k, val in s.items():
            if k == "causas":
                for c in val or []:
                    variantes.append({**s, "causas": [x for x in val if x != c]})
                    etiquetas.append(f"causa: {NOMBRES_CAUSA.get(c, c)}")
            elif k in NEUTROS and val not in (None, "") and val != NEUTROS[k]:
                variantes.append({**s, k: NEUTROS[k]})
                etiquetas.append(f"{NOMBRES_FACTOR.get(k, k)} = {self._legible(val)}")
        if not variantes:
            return []
        df = pd.concat([self.preparar(x) for x in variantes], ignore_index=True)
        p = self.predecir_gravedad(df)
        aportes = [{"factor": e, "pp": _f(100 * (p0 - pi), 2)} for e, pi in zip(etiquetas, p) if abs(p0 - pi) >= 0.001]
        return sorted(aportes, key=lambda a: -abs(a["pp"]))[:10]

    @staticmethod
    def _legible(val) -> str:
        if val is True:
            return "sí"
        if val is False:
            return "no"
        return str(val).lower() if isinstance(val, str) else f"{val:g}" if isinstance(val, (int, float)) else str(val)

    # ─────────────── 3. perfiles (no supervisado) ───────────────
    def asignar_perfil(self, df: pd.DataFrame) -> int:
        P = self.P
        rasgos = {}
        for nombre, cond in P["definicion_rasgos"].items():
            conds = cond if isinstance(cond, list) else [cond]
            rasgos[nombre] = int(any(pd.to_numeric(df[c], errors="coerce").fillna(0).iloc[0] == val
                                     for c, _, val in conds))
        B = pd.DataFrame([rasgos])[P["rasgos"]]
        Z = (B - P["prevalencia"]) * P["pesos"]
        return int(P["modelo"].predict(Z.values)[0])

    def perfiles(self) -> list[dict]:
        d = self.P["descripcion_perfiles"]
        return [{"perfil": int(r["perfil"]), "pct_siniestros": _f(r["% siniestros"]),
                 "tasa_muerte": _f(r["tasa de muerte"]), "veces": _f(r["veces la tasa general"], 2),
                 "pct_muertes": _f(r["% de todas las muertes"]), "rasgos": str(r["rasgos que lo definen (lift)"])}
                for _, r in d.iterrows()]

    # ─────────────── normas (motor de similitud) ───────────────
    @staticmethod
    def _cumple(fila: pd.Series, condiciones) -> bool:
        for col, op, val in condiciones:
            x = fila.get(col, np.nan)
            if isinstance(val, str):
                if str(x) != val:
                    return False
                continue
            x = 0.0 if pd.isna(x) else float(x)
            if (op == "eq" and x != val) or (op == "ne" and x == val) or (op == "gt" and not x > val):
                return False
        return True

    def vector_temas(self, df: pd.DataFrame) -> np.ndarray:
        N = self.N
        fila, a = df.iloc[0], np.zeros(len(N["temas"]))
        for tema, peso, condiciones in N["reglas"]:
            if self._cumple(fila, condiciones):
                j = N["temas"].index(tema)
                a[j] = max(a[j], peso)
        return a

    def asignar_norma(self, df: pd.DataFrame, top: int = 3) -> tuple[list[dict], list[str], str | None]:
        N = self.N
        fila, a = df.iloc[0], self.vector_temas(df)
        temas = [t for t, x in zip(N["temas"], a) if x >= 0.5]
        if a.sum() == 0:
            return [], temas, None
        w = np.sqrt(N["pesos_temas"])
        A, M = a * w, N["matriz_articulos"] * w
        cos = (M @ A) / (np.linalg.norm(M, axis=1) * np.linalg.norm(A) + 1e-12)
        art = N["articulos"].copy()
        art["similitud"] = cos * ((1 - N["peso_relevancia"]) + N["peso_relevancia"] * art["relevancia"])
        vigente = (art["inicio"] <= fila["fecha"]) & (fila["fecha"] < art["fin"])
        art = art[vigente].sort_values("similitud", ascending=False).drop_duplicates("id_norma").head(top)
        conf = "ALTA" if any(fila.get(c, 0) == 1 for c in N["causas_confianza"]) else "BAJA"
        normas = [{"id_norma": r["id_norma"], "segmento": r["segmento"], "similitud": _f(r["similitud"], 3),
                   "tema_principal": r["tema_principal"], "resumen": r["resumen"],
                   "texto": str(r["texto"])[:600]} for _, r in art.iterrows()]
        return normas, temas, conf

    # ─────────────── ficha completa de un siniestro ───────────────
    def analizar(self, s: dict) -> dict:
        s = {k: v for k, v in s.items() if v not in (None, "")}
        df = self.preparar(s)
        p = float(self.predecir_gravedad(df)[0])
        perfil = self.asignar_perfil(df)
        normas, temas, conf = self.asignar_norma(df)
        prec = self.N["precision_validada"]
        G = self.G
        return {
            "probabilidad": _f(p), "nivel": self.nivel_riesgo(p), "veces_tasa": _f(p / G["tasa_base"], 2),
            "tasa_base": _f(G["tasa_base"]), "umbral": _f(G["umbral"]),
            "factores": self.explicar(s, p),
            "perfil": next(x for x in self.perfiles() if x["perfil"] == perfil),
            "temas": temas, "normas": normas, "confianza_norma": conf,
            "precision_norma": _f(prec["confianza_alta" if conf == "ALTA" else "confianza_baja"], 2) if conf else None,
            "localidad": localidad_modelo(s.get("localidad")),
        }

    # ─────────────── 2. riesgo por tramo (regresión) ───────────────
    def tramos(self) -> list[dict]:
        cols = ["civ", "lat", "lon", "pred_2026", "tipo_via", "vel_reglamentaria", "vel_limite", "largo_m",
                "y_lag1", "y_prom3", "camara", "resalto", "semaforo"]
        d = self.pred[cols]
        return [{"civ": int(r.civ), "lat": _f(r.lat, 6), "lon": _f(r.lon, 6), "pred": _f(r.pred_2026, 3),
                 "tipo_via": str(r.tipo_via), "vel_reglamentaria": _f(r.vel_reglamentaria, 0),
                 "vel_limite": _f(r.vel_limite, 0), "largo_m": _f(r.largo_m, 0), "y_2025": _f(r.y_lag1, 0),
                 "y_prom3": _f(r.y_prom3, 2), "camara": int(r.camara or 0), "resalto": int(r.resalto or 0),
                 "semaforo": int(r.semaforo or 0), "rango": i + 1}
                for i, r in enumerate(d.itertuples(index=False))]

    def resumen_tramos(self) -> dict:
        T, pred = self.T, self.pred
        return {"n_tramos": int(T["n_tramos"]), "n_paquete": int(len(pred)),
                "total_2026": _f(T["total_esperado_2026"], 0),
                "top100_pct": _f(pred["pred_2026"].head(100).sum() / T["total_esperado_2026"], 3)}

    def escenario(self, civ: int, cambios: list[str]) -> dict:
        """Predicción 2026 de un tramo con y sin los dispositivos indicados (asociación del modelo, no causalidad)."""
        T = self.T
        fila = self.pred[self.pred["civ"] == civ]
        if fila.empty:
            raise ValueError("Ese CIV no está entre los tramos del paquete del modelo.")
        x = fila[T["variables"]].astype(float)
        actual = float(T["modelo"].predict(x)[0])
        nueva = x.copy()
        aplicados = []
        for c in cambios:
            if c == "camara":
                nueva["camara"] = 1
            elif c == "resalto":
                nueva["resalto"] = 1
            elif c == "semaforo":
                nueva["semaforo"] = 1
                nueva["sem_n"] = max(1.0, float(nueva["sem_n"].iloc[0] or 0))
            elif c == "vel30":
                nueva["vel_limite"] = 30
            else:
                continue
            aplicados.append(ESCENARIOS[c])
        simulado = float(T["modelo"].predict(nueva)[0])
        return {"civ": int(civ), "actual": _f(actual, 3), "simulado": _f(simulado, 3),
                "cambio_pct": _f(100 * (simulado / actual - 1), 1) if actual > 0 else None,
                "cambios": aplicados}

    # ─────────────── tramos a lo largo de una vía dibujada ───────────────
    # data/ml/ (lo genera src/preparar_tramos_ml.py): variables del modelo para TODA la malla vial
    def _cargar_malla(self):
        if getattr(self, "_malla", None) is not None:
            return
        with self._lock_malla:
            if getattr(self, "_malla", None) is not None:
                return
            d = os.path.join(RAIZ, "data", "ml")
            ruta_t, ruta_p = os.path.join(d, "tramos_2026.parquet"), os.path.join(d, "tramos_puntos.npz")
            if not (os.path.exists(ruta_t) and os.path.exists(ruta_p)):
                raise FileNotFoundError("Falta la tabla de tramos de toda la malla vial. Ejecute una vez: "
                                        "python src/preparar_tramos_ml.py")
            t = pd.read_parquet(ruta_t)
            z = np.load(ruta_p)
            if not np.array_equal(z["civ"], t["civ"].to_numpy()):
                raise ValueError("tramos_puntos.npz no corresponde a tramos_2026.parquet: vuelva a ejecutar "
                                 "python src/preparar_tramos_ml.py")
            t["pred_2026"] = self.T["modelo"].predict(t[self.T["variables"]])
            self._puntos = z["puntos"].astype(float)                # (n, 11, 2) lon/lat
            self._malla = t

    def _malla_cerca(self, lineas: list, margen: float) -> tuple[pd.DataFrame, np.ndarray]:
        """Tramos que pueden quedar cerca de la línea: de Supabase (solo los de la zona) o de data/ml/."""
        import bd
        if bd.activa():
            try:
                return self._malla_supabase(lineas, margen)
            except Exception as e:
                print(f"  ⚠ tramos desde Supabase: {type(e).__name__}: {e}")
                if not os.path.exists(os.path.join(RAIZ, "data", "ml", "tramos_2026.parquet")):
                    raise ValueError(f"No pude leer los tramos de Supabase ({type(e).__name__}). "
                                     "¿Ya corrió python src/subir_a_supabase.py?") from e
        self._cargar_malla()
        return self._malla, self._puntos

    def _malla_supabase(self, lineas: list, margen: float) -> tuple[pd.DataFrame, np.ndarray]:
        """Pide a Supabase solo los tramos cuyas cajas tocan la línea, en trozos de ~300 m (índice GiST),
        y predice 2026 solo para ellos: no se carga la malla entera en memoria."""
        import bd
        ML, MA, PASO = 107_000.0, 110_600.0, 300.0
        cajas = []
        for l in lineas:
            l = np.asarray(l, float)
            pts = [l[0]]
            for a, b in zip(l[:-1], l[1:]):                          # ningún segmento de más de 300 m
                n = max(1, int(np.ceil(np.hypot((b[0] - a[0]) * ML, (b[1] - a[1]) * MA) / PASO)))
                pts.extend(a + (b - a) * k / n for k in range(1, n + 1))
            pts = np.array(pts)
            largo = np.concatenate([[0], np.cumsum(np.hypot(np.diff(pts[:, 0]) * ML, np.diff(pts[:, 1]) * MA))])
            trozo = np.floor(largo / PASO).astype(int)
            for k in np.unique(trozo):
                idx = np.flatnonzero(trozo == k)
                q = pts[max(idx[0] - 1, 0): idx[-1] + 1]           # incluye el punto anterior: sin huecos
                cajas.append(f"(({q[:, 0].min() - margen},{q[:, 1].min() - margen}),"
                             f"({q[:, 0].max() + margen},{q[:, 1].max() + margen}))")
        filas = bd.consultar(
            f"select distinct on (t.civ) t.* from unnest(%s::text[]) as c(caja) "
            f"join {bd.ESQUEMA}.tramos t on t.caja && c.caja::box order by t.civ", (cajas,))
        if not filas:
            cols = ["civ", "nombre_via", "localidad", "tipo_via", "geom", "pred_2026", "largo_m", "vel_limite",
                    "camara", "resalto", "semaforo", "y_lag1", "y_lag2", "y_lag3"]
            cols += [c for c in self.T["variables"] if c not in cols]
            return pd.DataFrame(columns=cols, dtype=float), np.zeros((0, 11, 2))
        P = np.array([f.pop("puntos") for f in filas], float).reshape(len(filas), -1, 2)
        t = pd.DataFrame(filas).drop(columns=["caja"], errors="ignore")
        t["pred_2026"] = self.T["modelo"].predict(t[self.T["variables"]].astype(float))
        return t, P

    @staticmethod
    def _distancia(px: np.ndarray, py: np.ndarray, lineas: list, hasta: float) -> np.ndarray:
        """Distancia (m) de cada punto a la línea más cercana; inf si pasa de `hasta`."""
        ML, MA = 107_000.0, 110_600.0                                # metros por grado en Bogotá
        # La línea se densifica en puntos cada ~2 m y se busca el más cercano con un árbol KD:
        # error máximo ~1 m, y una avenida entera (miles de vértices) se resuelve en fracciones de segundo.
        from scipy.spatial import cKDTree
        d = np.full(len(px), np.inf)
        trozos = []
        for l in lineas:
            l = np.asarray(l, float) * [ML, MA]
            if len(l) < 2:
                continue
            for a, b in zip(l[:-1], l[1:]):
                n = max(2, int(np.ceil(np.hypot(*(b - a)) / 2.0)) + 1)
                trozos.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
        if not trozos or not len(px):
            return d
        dist, _ = cKDTree(np.vstack(trozos)).query(np.column_stack([px * ML, py * MA]), distance_upper_bound=hasta + 1)
        return dist

    def _aplicar(self, x: pd.DataFrame, cambios: list[str]) -> tuple[pd.DataFrame, list[str]]:
        nueva, aplicados = x.copy(), []
        for c in cambios:
            if c == "camara":
                nueva["camara"] = 1
            elif c == "resalto":
                nueva["resalto"] = 1
            elif c == "semaforo":
                nueva["semaforo"] = 1
                nueva["sem_n"] = np.maximum(1.0, nueva["sem_n"].fillna(0))
            elif c == "vel30":
                nueva["vel_limite"] = 30
            else:
                continue
            aplicados.append(ESCENARIOS[c])
        return nueva, aplicados

    def tramos_de_linea(self, lineas: list, ancho_m: float = 20, cambios: list[str] | None = None,
                        minimo: float = 0.6, maximo: int = 600) -> dict:
        """Tramos (CIV) que recorre la línea: al menos `minimo` de su largo a menos de `ancho_m` metros.
        Predicción 2026 de cada uno y, si hay `cambios`, la misma predicción con esa infraestructura."""
        lineas = [l for l in lineas if len(l) >= 2]
        if not lineas:
            raise ValueError("Dibuje una línea con al menos dos puntos.")
        ancho = float(min(max(ancho_m or 20, 5), 100))
        todos = np.vstack([np.asarray(l, float) for l in lineas])
        margen = (ancho + 50) / 100_000
        t, P = self._malla_cerca(lineas, margen)
        cand = np.flatnonzero((P[:, :, 0].max(1) >= todos[:, 0].min() - margen) & (P[:, :, 0].min(1) <= todos[:, 0].max() + margen) &
                              (P[:, :, 1].max(1) >= todos[:, 1].min() - margen) & (P[:, :, 1].min(1) <= todos[:, 1].max() + margen))
        if len(cand):
            pts = P[cand].reshape(-1, 2)
            dentro = (self._distancia(pts[:, 0], pts[:, 1], lineas, ancho) <= ancho).reshape(len(cand), -1)
            cand = cand[dentro.mean(1) >= minimo]
        sel = t.iloc[cand].sort_values("pred_2026", ascending=False)
        recortado = len(sel) > maximo
        sel = sel.head(maximo)
        x = sel[self.T["variables"]].astype(float)
        base = sel["pred_2026"].to_numpy() if len(sel) else np.array([])
        cambios = list(cambios or [])
        nueva, aplicados = self._aplicar(x, cambios)
        sim = self.T["modelo"].predict(nueva) if (len(sel) and aplicados) else base
        tramos = []
        for (_, r), p0, p1 in zip(sel.iterrows(), base, sim):
            tramos.append({"civ": int(r["civ"]), "via": r["nombre_via"] or "", "tipo_via": r["tipo_via"],
                           "localidad": r["localidad"] or "", "largo_m": _f(r["largo_m"], 0),
                           "vel_limite": _f(r["vel_limite"], 0), "camara": int(r["camara"]), "resalto": int(r["resalto"]),
                           "semaforo": int(r["semaforo"]), "y_2023": _f(r["y_lag3"], 0), "y_2024": _f(r["y_lag2"], 0),
                           "y_2025": _f(r["y_lag1"], 0), "pred": _f(p0, 3), "sim": _f(p1, 3),
                           "geom": json.loads(r["geom"])})
        s = lambda c: _f(sel[c].sum(), 0) if len(sel) else 0
        tot0, tot1 = float(base.sum()), float(np.sum(sim))
        return {"n_tramos": len(tramos), "recortado": recortado, "ancho_m": ancho,
                "largo_km": _f(sel["largo_m"].sum() / 1000, 2) if len(sel) else 0,
                "historia": {"2023": s("y_lag3"), "2024": s("y_lag2"), "2025": s("y_lag1")},
                "esperado_2026": _f(tot0, 2), "con_cambios_2026": _f(tot1, 2) if aplicados else None,
                "cambio_abs": _f(tot1 - tot0, 2) if aplicados else None,
                "cambio_pct": _f(100 * (tot1 / tot0 - 1), 1) if (aplicados and tot0 > 0) else None,
                "cambios": aplicados,
                "ya_tienen": {k: int(sel[k].sum()) for k in ("camara", "resalto", "semaforo")} if len(sel) else {},
                "con_30": int((sel["vel_limite"] <= 30).sum()) if len(sel) else 0,
                "tramos": tramos}

    # ─────────────── estado para el front ───────────────
    def estado(self) -> dict:
        G, T, M = self.G, self.T, self.META
        mg, mt = G["metricas_prueba"], T["metricas_prueba"]
        return {
            "disponible": True, "aviso": self.aviso, "creado": M.get("creado"), "division": M.get("division_temporal"),
            "versiones": M.get("versiones"),
            "gravedad": {"algoritmo": G["algoritmo"], "umbral": _f(G["umbral"]), "tasa_base": _f(G["tasa_base"]),
                         "roc_auc": _f(mg.get("ROC-AUC"), 3), "recall": _f(mg.get("Recall"), 3),
                         "captura_top10": _f(mg.get("Captura top 10%"), 3)},
            "tramos": {"algoritmo": T["algoritmo"], **self.resumen_tramos(),
                       "mae": _f(mt.get("MAE"), 3), "captura_top5": _f(mt.get("Captura top 5%"), 3)},
            "perfiles": {"algoritmo": self.P["algoritmo"], "lista": self.perfiles()},
            "normas": {"n_normas": int(self.N["articulos"]["id_norma"].nunique()),
                       "n_articulos": int(len(self.N["articulos"])),
                       "precision": {k: _f(v, 2) for k, v in self.N["precision_validada"].items()}},
            "causas": [{"id": c, "nombre": NOMBRES_CAUSA[c]} for c in CAUSAS_VALIDAS],
            "clases": CLASES, "ejemplos": EJEMPLOS, "escenarios": ESCENARIOS,
        }
