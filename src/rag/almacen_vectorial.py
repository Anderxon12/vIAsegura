"""
4. ÍNDICE VECTORIAL (FAISS)
Guarda los vectores en un índice FAISS de producto interno (IndexFlatIP): con
vectores normalizados equivale a similitud coseno y la búsqueda es exacta,
suficiente para miles de fragmentos. Junto al índice se guardan los
fragmentos (texto + metadatos) y qué modelo de embeddings se usó, porque
buscar con otro modelo daría resultados sin sentido.

    out/rag/indice.faiss      vectores
    out/rag/fragmentos.jsonl  un fragmento por línea, en el mismo orden
    out/rag/info.json         modelo, dimensión, fecha, documentos
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import faiss
import numpy as np

from .fragmentos import Fragmento


class AlmacenFAISS:
    def __init__(self, indice: faiss.Index, fragmentos: list[Fragmento], info: dict):
        self.indice, self.fragmentos, self.info = indice, fragmentos, info

    @classmethod
    def construir(cls, vectores: np.ndarray, fragmentos: list[Fragmento], modelo: str) -> "AlmacenFAISS":
        indice = faiss.IndexFlatIP(vectores.shape[1])
        indice.add(vectores)
        info = {"modelo_embeddings": modelo, "dimension": int(vectores.shape[1]), "n_fragmentos": len(fragmentos),
                "documentos": sorted({f.titulo for f in fragmentos}), "creado": datetime.now().isoformat(timespec="seconds")}
        return cls(indice, fragmentos, info)

    def guardar(self, carpeta: str) -> None:
        os.makedirs(carpeta, exist_ok=True)
        faiss.write_index(self.indice, os.path.join(carpeta, "indice.faiss"))
        with open(os.path.join(carpeta, "fragmentos.jsonl"), "w", encoding="utf-8") as f:
            for fr in self.fragmentos:
                f.write(json.dumps(fr.a_dict(), ensure_ascii=False) + "\n")
        with open(os.path.join(carpeta, "info.json"), "w", encoding="utf-8") as f:
            json.dump(self.info, f, ensure_ascii=False, indent=2)

    @classmethod
    def cargar(cls, carpeta: str) -> "AlmacenFAISS":
        ruta = os.path.join(carpeta, "indice.faiss")
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No hay índice en {carpeta}. Ejecute primero: python src/rag/cli.py indexar")
        with open(os.path.join(carpeta, "fragmentos.jsonl"), encoding="utf-8") as f:
            fragmentos = [Fragmento(**json.loads(l)) for l in f if l.strip()]
        with open(os.path.join(carpeta, "info.json"), encoding="utf-8") as f:
            info = json.load(f)
        return cls(faiss.read_index(ruta), fragmentos, info)

    @classmethod
    def vacio(cls, dimension: int, modelo: str) -> "AlmacenFAISS":
        return cls.construir(np.zeros((0, dimension), "float32"), [], modelo)

    def _actualizar_info(self) -> None:
        self.info["n_fragmentos"] = len(self.fragmentos)
        self.info["documentos"] = sorted({f.titulo for f in self.fragmentos})
        self.info["actualizado"] = datetime.now().isoformat(timespec="seconds")

    def con_documento(self, vectores: np.ndarray, fragmentos: list[Fragmento], doc_id: str,
                      archivo: str | None = None) -> "AlmacenFAISS":
        """Copia del índice sin los fragmentos viejos del documento y con los nuevos (para reindexarlo)."""
        base = self.sin_documento(doc_id, archivo)
        base.indice.add(vectores)
        base.fragmentos += fragmentos
        base._actualizar_info()
        return base

    def sin_documento(self, doc_id: str, archivo: str | None = None) -> "AlmacenFAISS":
        """Copia del índice sin los fragmentos del documento. IndexFlat guarda los vectores, así que se reconstruye.
        Se reconoce por su id o por su archivo, también con el formato viejo (solo el nombre, sin carpeta)."""
        nombres = {doc_id, doc_id.split("/")[-1]}
        archivos = {archivo, archivo.split("/")[-1]} if archivo else set()

        def es_del_doc(f: Fragmento) -> bool:
            return f.doc_id in nombres or f.metadatos.get("archivo") in archivos

        quedan = [i for i, f in enumerate(self.fragmentos) if not es_del_doc(f)]
        dim = self.indice.d
        todos = self.indice.reconstruct_n(0, self.indice.ntotal) if self.indice.ntotal else np.zeros((0, dim), "float32")
        nuevo = faiss.IndexFlatIP(dim)
        if quedan:
            nuevo.add(np.ascontiguousarray(todos[quedan]))
        copia = AlmacenFAISS(nuevo, [self.fragmentos[i] for i in quedan], dict(self.info))
        copia._actualizar_info()
        return copia

    def buscar(self, vector: np.ndarray, k: int) -> list[tuple[Fragmento, float]]:
        k = min(k, self.indice.ntotal)
        if k <= 0:
            return []
        puntajes, posiciones = self.indice.search(vector.reshape(1, -1).astype("float32"), k)
        return [(self.fragmentos[i], float(s)) for s, i in zip(puntajes[0], posiciones[0]) if i >= 0]
