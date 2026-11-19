"""
Línea de comandos del RAG.

    python src/rag/cli.py indexar                 crea o actualiza el índice (out/rag/)
    python src/rag/cli.py buscar "zonas escolares" solo recuperación: fragmentos y similitud
    python src/rag/cli.py preguntar "¿Qué metas tiene el Plan Distrital de Seguridad Vial?"
    python src/rag/cli.py chat                     conversación (salir con «salir»)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # carpeta src/
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rag.pipeline import RAG  # noqa: E402


def mostrar(r) -> None:
    print("\n" + r.texto + "\n")
    if r.fuentes:
        print("Fuentes:")
        for f in r.fuentes:
            linea = f"  [{f['n']}] {f['titulo']} — {f['seccion']} (similitud {f['similitud']:.2f})"
            if f["estado"]:
                linea += f" · {f['estado']}"
            print(linea + (f"\n      {f['url']}" if f["url"] else ""))


def main() -> None:
    p = argparse.ArgumentParser(description="RAG de normas viales de Bogotá")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("indexar")
    b = sub.add_parser("buscar"); b.add_argument("texto"); b.add_argument("-k", type=int, default=5)
    q = sub.add_parser("preguntar"); q.add_argument("texto")
    sub.add_parser("chat")
    a = p.parse_args()

    rag = RAG()
    if a.cmd == "indexar":
        print(f"Indexando {rag.config.carpeta_documentos} con {rag.config.modelo_embeddings}…")
        r = rag.indexar()
        print(f"✅ {len(r['documentos'])} documentos → {r['fragmentos']} fragmentos (vectores de {r['dimension']} dimensiones)")
        for t in r["documentos"]:
            print(f"   · {t}")
        if r["omitidos"]:
            print(f"⚠️  {len(r['omitidos'])} archivos omitidos:")
            for o in r["omitidos"]:
                print(f"   · {o}")
        print(f"Índice guardado en {r['carpeta']}")
    elif a.cmd in ("buscar", "preguntar", "chat"):
        rag.asegurar_modelo()
    if a.cmd == "buscar":
        for n, (f, s) in enumerate(rag.almacen.buscar(rag.embedder.consulta(a.texto), a.k), 1):
            print(f"\n[{n}] similitud {s:.3f} · {f.titulo} — {f.seccion}\n    {f.texto[:300]}…")
    elif a.cmd == "preguntar":
        mostrar(rag.preguntar(a.texto))
    elif a.cmd == "chat":
        modo = f"Claude ({rag.config.modelo_llm})" if rag.config.hay_llm else "sin LLM (extractivo)"
        print(f"RAG de normas viales · modo {modo}. Escriba «salir» para terminar.")
        while True:
            try:
                t = input("\nPregunta> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if t.lower() in ("salir", "exit", "q"):
                break
            if t:
                mostrar(rag.preguntar(t))


if __name__ == "__main__":
    main()
