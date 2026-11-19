"""
Conexión a Supabase (Postgres) — opcional.

Si existe la variable DATABASE_URL (en .env o en Render), la app lee de Supabase las
tablas grandes en vez de cargarlas en memoria:

    viasegura.tramos    los ~113 mil tramos de la malla vial (pestaña «Modelos ML»)
    viasegura.barrios   los polígonos de los sectores catastrales (barrios)

Sin DATABASE_URL todo sigue funcionando como antes, con los archivos locales.

DATABASE_URL: en Supabase → Connect → «Transaction pooler» (puerto 6543), con su contraseña:
    postgresql://postgres.<ref>:<contraseña>@aws-0-<region>.pooler.supabase.com:6543/postgres
Use el pooler y no la conexión directa: la directa es solo IPv6 y Render no la alcanza.

Las tablas se llenan una vez con:  python src/subir_a_supabase.py
"""

from __future__ import annotations

import atexit
import os
import threading

ESQUEMA = "viasegura"
_pool = None
_lock = threading.Lock()


def url() -> str:
    try:
        from entorno import cargar_entorno
        cargar_entorno()
    except Exception:
        pass
    v = os.environ.get("DATABASE_URL", "").strip()
    return "" if (not v or "REEMPLAZAR" in v.upper()) else v


def activa() -> bool:
    return bool(url())


def _obtener_pool():
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                from psycopg_pool import ConnectionPool
                # prepare_threshold=None: el «transaction pooler» de Supabase no admite sentencias preparadas
                _pool = ConnectionPool(url(), min_size=1, max_size=3, timeout=30, open=True,
                                       kwargs={"prepare_threshold": None, "connect_timeout": 10})
                atexit.register(_pool.close)
    return _pool


def consultar(sql: str, params=None) -> list[dict]:
    """Ejecuta una consulta y devuelve las filas como diccionarios."""
    from psycopg.rows import dict_row
    with _obtener_pool().connection() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()
