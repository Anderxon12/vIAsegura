"""
=============================================================
  VÍASEGURA IA — Subir las tablas grandes a Supabase
=============================================================

Crea el esquema «viasegura» en su proyecto de Supabase y carga:

    viasegura.tramos    data/ml/tramos_2026.parquet + tramos_puntos.npz  (~113 mil tramos)
    viasegura.barrios   data/raw/limites/*Sector_catastral*.csv          (~1 200 polígonos)

Antes, en .env (raíz del proyecto), agregue la cadena de conexión del «Transaction pooler»
(Supabase → Connect → Transaction pooler, puerto 6543):

    DATABASE_URL=postgresql://postgres.<ref>:<contraseña>@aws-0-<region>.pooler.supabase.com:6543/postgres

Uso (en su PC; se puede repetir: reemplaza los datos):
    python src/subir_a_supabase.py

Las tablas quedan con RLS activado y sin políticas: la API pública de Supabase (anon) no las
puede leer; solo la app, que entra con DATABASE_URL.
"""

from __future__ import annotations

import io
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bd  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAMOS = os.path.join(RAIZ, "data", "ml", "tramos_2026.parquet")
PUNTOS = os.path.join(RAIZ, "data", "ml", "tramos_puntos.npz")
E = bd.ESQUEMA


def _tipo_sql(serie: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(serie):
        return "boolean"
    if pd.api.types.is_integer_dtype(serie):
        return "bigint"
    if pd.api.types.is_float_dtype(serie):
        return "double precision"
    return "text"


def _valor(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def _conectar():
    import psycopg
    return psycopg.connect(bd.url(), prepare_threshold=None, connect_timeout=15, autocommit=False)


def subir_tramos(con) -> None:
    if not (os.path.exists(TRAMOS) and os.path.exists(PUNTOS)):
        print(f"  ⚠ faltan {os.path.relpath(TRAMOS, RAIZ)} / tramos_puntos.npz: ejecute antes "
              "python src/preparar_tramos_ml.py (o descomprima despliegue/datos_app.zip). Se omiten los tramos.")
        return
    t = pd.read_parquet(TRAMOS)
    z = np.load(PUNTOS)
    if not np.array_equal(z["civ"], t["civ"].to_numpy()):
        sys.exit("tramos_puntos.npz no corresponde a tramos_2026.parquet: vuelva a ejecutar preparar_tramos_ml.py")
    P = z["puntos"].astype(float)                                   # (n, 11, 2) lon/lat
    cols = list(t.columns)
    defs = ", ".join(f'"{c}" {"bigint primary key" if c == "civ" else _tipo_sql(t[c])}' for c in cols)
    with con.cursor() as cur:
        cur.execute(f"drop table if exists {E}.tramos")
        cur.execute(f"create table {E}.tramos ({defs}, puntos double precision[] not null, caja box not null)")
        print(f"  tramos: cargando {len(t):,} filas…")
        nombres = ", ".join(f'"{c}"' for c in cols) + ", puntos, caja"
        with cur.copy(f"copy {E}.tramos ({nombres}) from stdin") as cp:
            for fila, p in zip(t.itertuples(index=False), P):
                caja = f"(({p[:, 0].min()},{p[:, 1].min()}),({p[:, 0].max()},{p[:, 1].max()}))"
                cp.write_row([_valor(v) for v in fila] + [p.ravel().round(7).tolist(), caja])
        cur.execute(f"create index tramos_caja_gist on {E}.tramos using gist (caja)")
        cur.execute(f"alter table {E}.tramos enable row level security")
        cur.execute(f"analyze {E}.tramos")
    con.commit()
    print(f"  tramos: {len(t):,} listos")


def subir_barrios(con) -> None:
    from app_web import Datos                                      # misma lectura y nombres que usa la app
    capa = Datos._capa(None, "Sector_catastral")
    if not capa:
        print("  ⚠ no encontré data/raw/limites/*Sector_catastral*.csv: se omiten los barrios.")
        return
    with con.cursor() as cur:
        cur.execute(f"drop table if exists {E}.barrios")
        cur.execute(f"create table {E}.barrios (nombre text primary key, geometria_json text not null)")
        with cur.copy(f"copy {E}.barrios (nombre, geometria_json) from stdin") as cp:
            for nombre, texto in dict.items(capa):                 # texto ArcGIS sin convertir
                cp.write_row([nombre, texto])
        cur.execute(f"alter table {E}.barrios enable row level security")
    con.commit()
    print(f"  barrios: {len(capa):,} listos")


def main() -> None:
    if not bd.activa():
        sys.exit("Falta DATABASE_URL en .env (vea las instrucciones al inicio de este archivo).")
    t0 = time.time()
    print("Conectando a Supabase…")
    with _conectar() as con:
        with con.cursor() as cur:
            cur.execute(f"create schema if not exists {E}")
            cur.execute(f"revoke all on schema {E} from anon, authenticated")
        con.commit()
        subir_tramos(con)
        subir_barrios(con)
        with con.cursor() as cur:
            cur.execute(f"select pg_size_pretty(sum(pg_total_relation_size(c.oid))) from pg_class c "
                        f"join pg_namespace n on n.oid = c.relnamespace where n.nspname = %s", (E,))
            print(f"\nListo en {time.time() - t0:,.0f} s. Espacio usado en Supabase: {cur.fetchone()[0]}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
