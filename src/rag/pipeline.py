"""
PIPELINE RAG
    indexar():          todos los documentos → fragmentos → embeddings → índice FAISS en disco
    indexar_archivo():  un solo documento (p. ej. un PDF recién subido); reemplaza su versión anterior
    quitar_archivo():   saca un documento del índice (y opcionalmente lo borra)
    estado_documentos(): qué archivos están indexados, pendientes o sin texto
    preguntar(): pregunta → embedding → búsqueda en FAISS → fragmentos → Claude → respuesta con fuentes
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Callable

from .almacen_vectorial import AlmacenFAISS
from .config import Config, cargar_config
from .documentos import cargar_documento, cargar_documentos, listar_archivos
from .embeddings import GeneradorEmbeddings
from .fragmentos import Fragmento, fragmentar
from .llm import GeneradorClaude, GeneradorExtractivo


@dataclass
class Respuesta:
    texto: str
    fuentes: list[dict] = field(default_factory=list)


class RAG:
    def __init__(self, config: Config | None = None, embedder: GeneradorEmbeddings | None = None, generador=None):
        self.config = config or cargar_config()
        self._embedder = embedder
        self._generador = generador
        self._almacen: AlmacenFAISS | None = None
        self._lock = threading.Lock()              # una escritura del índice a la vez

    # ── componentes (se crean al usarlos: cargar el modelo tarda unos segundos) ──
    @property
    def embedder(self) -> GeneradorEmbeddings:
        if self._embedder is None:
            modelo, motor = self.config.modelo_embeddings, self.config.motor_embeddings
            if motor == "auto" and self.hay_indice:      # «auto»: seguir con el motor con que se creó el índice
                usado = self.almacen.info.get("modelo_embeddings", "")
                if usado.startswith("fastembed:") and self.almacen.fragmentos:
                    motor, modelo = "fastembed", usado.split(":", 1)[1]
            self._embedder = GeneradorEmbeddings(modelo, motor)
        return self._embedder

    @property
    def generador(self):
        if self._generador is None:
            c = self.config
            self._generador = (GeneradorClaude(c.anthropic_api_key, c.modelo_llm, c.max_tokens, c.temperatura)
                               if c.hay_llm else GeneradorExtractivo())
        return self._generador

    @property
    def almacen(self) -> AlmacenFAISS:
        if self._almacen is None:
            self._almacen = AlmacenFAISS.cargar(self.config.carpeta_indice)
        return self._almacen

    def _mismo_modelo(self, almacen: AlmacenFAISS) -> None:
        """Buscar o agregar con un modelo distinto al del índice daría resultados sin sentido."""
        usado = almacen.info.get("modelo_embeddings")
        if almacen.fragmentos and usado != self.embedder.nombre:
            raise RuntimeError(f"El índice se creó con «{usado}» y ahora el modelo es «{self.embedder.nombre}». "
                               "Use «Reindexar todo desde cero».")

    @property
    def hay_indice(self) -> bool:
        return self._almacen is not None or os.path.exists(os.path.join(self.config.carpeta_indice, "indice.faiss"))

    @staticmethod
    def _texto_a_vectorizar(f: Fragmento) -> str:
        return f"{f.titulo}. {f.seccion}. {f.texto}"

    def asegurar_modelo(self, avance: Callable[[str], None] = print) -> bool:
        """Si el índice se creó con otro modelo de embeddings (p. ej. se pasó al modelo ligero por falta
        de memoria), lo rehace con el modelo actual. Devuelve True si reindexó."""
        if not self.hay_indice or not self.almacen.fragmentos:
            return False
        ids_por_archivo: dict[str, set] = {}
        for f in self.almacen.fragmentos:      # un mismo archivo indexado dos veces (formato viejo y nuevo)
            ids_por_archivo.setdefault(str(f.metadatos.get("archivo", "")).split("/")[-1], set()).add(f.doc_id)
        duplicado = any(len(v) > 1 for v in ids_por_archivo.values())
        if self.almacen.info.get("modelo_embeddings") == self.embedder.nombre and not duplicado:
            return False
        avance("  Hay documentos repetidos en el índice; reindexando…" if duplicado else
               f"  El índice se creó con «{self.almacen.info.get('modelo_embeddings')}»; "
               f"reindexando con «{self.embedder.nombre}»…")
        self.indexar(avance)
        return True

    # ── indexación completa ──
    def indexar(self, avance: Callable[[str], None] = lambda _: None) -> dict:
        c = self.config
        with self._lock:
            avance("Leyendo documentos…")
            docs, omitidos = cargar_documentos(c.carpeta_documentos)
            if not docs:
                raise RuntimeError(f"No hay documentos con contenido en {c.carpeta_documentos}.")
            fragmentos: list[Fragmento] = [f for d in docs for f in fragmentar(d, c.tam_fragmento, c.solape)]
            avance(f"Calculando embeddings de {len(fragmentos)} fragmentos…")
            vectores = self.embedder.documentos([self._texto_a_vectorizar(f) for f in fragmentos])
            almacen = AlmacenFAISS.construir(vectores, fragmentos, self.embedder.nombre)
            almacen.info["omitidos"] = {o.split(": ", 1)[0]: {"motivo": o.split(": ", 1)[1], "mtime": self._mtime(o.split(": ", 1)[0])}
                                        for o in omitidos}
            almacen.guardar(c.carpeta_indice)
            self._almacen = almacen
        return {"documentos": [d.titulo for d in docs], "fragmentos": len(fragmentos),
                "dimension": int(vectores.shape[1]), "omitidos": omitidos, "carpeta": c.carpeta_indice}

    def _mtime(self, archivo: str) -> float:
        ruta = os.path.join(self.config.carpeta_documentos, archivo)
        return os.path.getmtime(ruta) if os.path.exists(ruta) else 0.0

    def _almacen_o_vacio(self) -> AlmacenFAISS:
        nuevo = AlmacenFAISS.vacio(self.embedder.dimension, self.embedder.nombre)
        if self.hay_indice:
            if self.almacen.fragmentos:
                return self.almacen
            nuevo.info["omitidos"] = self.almacen.info.get("omitidos", {})   # índice sin fragmentos: se rehace
        return nuevo

    @property
    def umbral(self) -> float:
        """Similitud mínima. La escala depende del modelo: e5 da similitudes altas (0,7–0,9) aun entre textos
        sin relación; paraphrase-MiniLM, más bajas (0,1–0,7). Un umbral pensado para e5 no sirve en el otro."""
        u = self.config.similitud_minima
        return 0.35 if not self.embedder.e5 and u > 0.6 else u

    # ── un documento ──
    def indexar_archivo(self, archivo: str, avance: Callable[[str], None] = lambda _: None) -> dict:
        """Indexa (o reindexa) un archivo de la carpeta de documentos. `archivo` es la ruta relativa."""
        c = self.config
        with self._lock:
            avance(f"Leyendo {archivo}…")
            doc, motivo = cargar_documento(c.carpeta_documentos, archivo)
            almacen = self._almacen_o_vacio()
            self._mismo_modelo(almacen)
            omitidos = almacen.info.setdefault("omitidos", {})
            if doc is None:
                omitidos[archivo] = {"motivo": motivo, "mtime": self._mtime(archivo)}
                almacen = almacen.sin_documento(os.path.splitext(archivo)[0], archivo)
                almacen.info["omitidos"] = omitidos
                almacen.guardar(c.carpeta_indice)
                self._almacen = almacen
                return {"archivo": archivo, "ok": False, "motivo": motivo}
            fragmentos = fragmentar(doc, c.tam_fragmento, c.solape)
            avance(f"Calculando embeddings de {len(fragmentos)} fragmentos de «{doc.titulo}»…")
            vectores = self.embedder.documentos([self._texto_a_vectorizar(f) for f in fragmentos])
            avance("Guardando en el índice FAISS…")
            omitidos.pop(archivo, None)
            almacen = almacen.con_documento(vectores, fragmentos, doc.id, archivo)
            almacen.info["omitidos"] = omitidos
            almacen.guardar(c.carpeta_indice)
            self._almacen = almacen
        return {"archivo": archivo, "ok": True, "titulo": doc.titulo, "fragmentos": len(fragmentos)}

    def quitar_archivo(self, archivo: str, borrar: bool = False) -> None:
        c = self.config
        with self._lock:
            if self.hay_indice:
                almacen = self.almacen.sin_documento(os.path.splitext(archivo)[0], archivo)
                almacen.info.get("omitidos", {}).pop(archivo, None)
                almacen.guardar(c.carpeta_indice)
                self._almacen = almacen
            ruta = os.path.join(c.carpeta_documentos, archivo)
            if borrar and os.path.exists(ruta):
                os.remove(ruta)

    def estado_documentos(self) -> list[dict]:
        """Cada archivo de la carpeta con su estado: indexado, pendiente (nuevo o modificado) o sin texto."""
        c = self.config
        frag, titulos, mt, omit = {}, {}, {}, {}
        if self.hay_indice:
            a = self.almacen
            omit = a.info.get("omitidos", {})
            for f in a.fragmentos:
                arch = f.metadatos.get("archivo", "")
                frag[arch] = frag.get(arch, 0) + 1
                titulos[arch] = f.titulo
                mt[arch] = f.metadatos.get("mtime")
        salida = []
        for arch in listar_archivos(c.carpeta_documentos):
            ruta = os.path.join(c.carpeta_documentos, arch)
            actual = os.path.getmtime(ruta)
            d = {"archivo": arch, "tamano_kb": round(os.path.getsize(ruta) / 1024), "titulo": titulos.get(arch, ""),
                 "fragmentos": frag.get(arch, 0)}
            if arch in frag:
                d["estado"] = "pendiente" if mt.get(arch) and actual > mt[arch] + 1 else "indexado"
                d["motivo"] = "el archivo cambió desde que se indexó" if d["estado"] == "pendiente" else ""
            elif arch in omit and actual <= omit[arch].get("mtime", 0) + 1:
                d["estado"], d["motivo"] = "sin_texto", omit[arch]["motivo"]
            else:
                d["estado"], d["motivo"] = "pendiente", "todavía no se ha indexado"
            salida.append(d)
        return salida

    # ── consulta ──
    def recuperar(self, pregunta: str, k: int | None = None) -> list[tuple[Fragmento, float]]:
        self._mismo_modelo(self.almacen)
        vector = self.embedder.consulta(pregunta)
        k = k or self.config.top_k
        res = self.almacen.buscar(vector, k * 2)
        salida, vistos = [], set()
        for f, s in res:                       # sin fragmentos repetidos
            if s >= self.umbral and f.texto not in vistos:
                vistos.add(f.texto)
                salida.append((f, s))
        return salida[:k]

    def preguntar(self, pregunta: str) -> Respuesta:
        recuperados = self.recuperar(pregunta)
        texto = self.generador.responder(pregunta, recuperados)
        fuentes = [{"n": n, "titulo": f.titulo, "seccion": f.seccion, "similitud": round(s, 3),
                    "url": f.metadatos.get("url", ""), "estado": f.metadatos.get("estado", "")}
                   for n, (f, s) in enumerate(recuperados, 1)]
        return Respuesta(texto=texto, fuentes=fuentes)
