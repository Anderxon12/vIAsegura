"""
Carga las claves y la configuración desde .env.

Busca el archivo en la raíz del proyecto y en src/ (los dos son válidos). Si
una variable aparece en ambos, gana el valor real sobre la plantilla
(«...REEMPLAZAR...»). Las variables ya definidas en Windows se respetan,
salvo que sean plantilla o estén vacías.

    python src/entorno.py      muestra de dónde salió cada clave y prueba Claude y Hugging Face
"""

from __future__ import annotations

import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVOS = [os.path.join(RAIZ, ".env"), os.path.join(RAIZ, "src", ".env")]
ORIGEN: dict[str, str] = {}          # variable → archivo de donde se tomó
_cargado = False


def _es_plantilla(v: str | None) -> bool:
    return not v or "REEMPLAZAR" in v.upper()


def _leer(ruta: str) -> dict[str, str]:
    try:
        from dotenv import dotenv_values
        return {k: v for k, v in dotenv_values(ruta, encoding="utf-8-sig").items() if v is not None}
    except ImportError:                       # sin python-dotenv: lector sencillo
        valores = {}
        with open(ruta, encoding="utf-8-sig") as f:
            for linea in f:
                linea = linea.strip()
                if linea and not linea.startswith("#") and "=" in linea:
                    k, v = linea.split("=", 1)
                    v = v.split(" #")[0].strip().strip('"').strip("'")
                    valores[k.strip().removeprefix("export ").strip()] = v
        return valores


def cargar_entorno() -> dict[str, str]:
    global _cargado
    if _cargado:
        return ORIGEN
    for ruta in ARCHIVOS:
        if not os.path.exists(ruta):
            continue
        for k, v in _leer(ruta).items():
            actual = os.environ.get(k)
            if actual is None or (_es_plantilla(actual) and not _es_plantilla(v)):
                os.environ[k] = v
                ORIGEN[k] = os.path.relpath(ruta, RAIZ)
    _cargado = True
    return ORIGEN


def oculta(v: str | None) -> str:
    if _es_plantilla(v):
        return "(sin configurar)"
    return f"{v[:7]}…{v[-4:]} ({len(v)} caracteres)"


def resumen_claves() -> list[str]:
    cargar_entorno()
    salida = []
    for k, nombre in (("ANTHROPIC_API_KEY", "Claude"), ("HF_TOKEN", "Hugging Face")):
        v = os.environ.get(k)
        de = ORIGEN.get(k, "variable de Windows" if v and not _es_plantilla(v) else "")
        salida.append(f"{nombre}: {oculta(v)}" + (f" · desde {de}" if de and not _es_plantilla(v) else ""))
    return salida


def _probar() -> None:
    cargar_entorno()
    print("Archivos .env encontrados:", ", ".join(os.path.relpath(a, RAIZ) for a in ARCHIVOS if os.path.exists(a)) or "ninguno")
    for linea in resumen_claves():
        print("  " + linea)

    k = os.environ.get("ANTHROPIC_API_KEY")
    if not _es_plantilla(k):
        try:
            import anthropic
            modelo = os.environ.get("ANTHROPIC_MODELO", "claude-sonnet-5-5")
            r = anthropic.Anthropic(api_key=k).messages.create(
                model=modelo, max_tokens=20, messages=[{"role": "user", "content": "Responde solo: OK"}])
            print(f"✅ Claude responde ({modelo}): {''.join(b.text for b in r.content if b.type == 'text').strip()}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Claude falló: {type(e).__name__}: {str(e)[:300]}")

    t = os.environ.get("HF_TOKEN")
    if not _es_plantilla(t):
        try:
            from huggingface_hub import whoami
            print(f"✅ Hugging Face: token válido (usuario {whoami(token=t)['name']})")
        except Exception as e:
            print(f"❌ Hugging Face falló: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    _probar()
